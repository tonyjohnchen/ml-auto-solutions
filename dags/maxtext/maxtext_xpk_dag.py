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

"""A DAG to run all GKE examples."""

import datetime
from airflow import models
from configs.vm_resource import TpuVersion, Project, Zone, ClusterName, DockerImage
from configs.maxtext import xpk_maxtext_config as config

with models.DAG(
    dag_id="gke_maxtext_dag",
    schedule=None,
    tags=["gke", "maxtext", "nightly"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
) as dag:
  maxtext_tpu_singleslice_v4_8 = config.get_maxtext_xpk_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      test_name="maxtext-single-slice",
      project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      cluster_name=ClusterName.V4_8_CLUSTER.value,
      docker_image=DockerImage.XPK_JAX_TEST.value,
      time_out_in_min=60,
  ).run()

  # flax_resnet_tpu_multislice_v4_128 = config.get_maxtext_xpk_config(
  #     tpu_version=TpuVersion.V4,
  #     tpu_cores=128,
  #     tpu_zone=Zone.US_CENTRAL2_B.value,
  #     test_name="resnet-multi-slice",
  #     project_name=Project.TPU_PROD_ENV_MULTIPOD.value,
  #     cluster_name=ClusterName.V4_128_MULTISLICE_CLUSTER.value,
  #     docker_image=DockerImage.XPK_JAX_TEST.value,
  #     time_out_in_min=60,
  #     num_slices=4,
  # ).run()
