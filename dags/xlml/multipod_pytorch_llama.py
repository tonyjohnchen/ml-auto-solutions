# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from airflow import models
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.utils.state import State
from apis import gcp_config, metric_config, task, test_config
from configs import composer_env, gcs_bucket, vm_resource
from configs.xlml.pytorch import multipodteam_llama2 as multipod_config
from dags.xlml.pytorchxla_llama import DAG_ID as SINGLE_SLICE_DAG_ID

# Run Sundays at 12 pm UTC (4 am PST)
SCHEDULED_TIME = "0 12 0 * *" if composer_env.is_prod_env() else None
US_CENTRAL2_B = gcp_config.GCPConfig(
    vm_resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    vm_resource.Zone.US_CENTRAL2_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id="multipod-pytorch-llama",
    schedule=SCHEDULED_TIME,
    tags=["multipod", "pytorch", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 3),
    catchup=False,
):
  if composer_env.is_prod_env():
    # Ensure single-slice tests are passing before running multislice
    single_slice_sensor = ExternalTaskSensor(
        task_id="single-slice-sensor",
        external_dag_id=SINGLE_SLICE_DAG_ID,
    )
  else:
    single_slice_sensor = EmptyOperator(task_id="single-slice-sensor")

  gcs_bucket_prefix = f"{gcs_bucket.XLML_OUTPUT_DIR}/multipod/pytorch/nightly"
  llama_perf_1x_v4_128 = multipod_config.get_pytorch_llama2_perf_config(
      tpu_version="4",
      tpu_cores="128",
      num_slices=1,
      per_slice_batch_size=128,
      gcs_bucket_prefix=gcs_bucket_prefix,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      cluster_name=vm_resource.ClusterName.V4_128_MULTISLICE_CLUSTER.value,
      cluster_project_name=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
      config_name="13B",
  ).run()

  llama_perf_2x_v4_128 = multipod_config.get_pytorch_llama2_perf_config(
      tpu_version="4",
      tpu_cores="128",
      num_slices=2,
      per_slice_batch_size=128,
      gcs_bucket_prefix=gcs_bucket_prefix,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      cluster_name=vm_resource.ClusterName.V4_128_MULTISLICE_CLUSTER.value,
      cluster_project_name=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
      config_name="13B",
  ).run()

  single_slice_sensor >> llama_perf_1x_v4_128 >> llama_perf_2x_v4_128
