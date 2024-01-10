# Dockerfile for xpk examples in xpk_example_dag.py,
# and is saved at gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest.
FROM python:3.10
RUN pip install --upgrade pip
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
# Package kubectl & gke-gcloud-auth-plugin needed for KubernetesPodOperator
RUN gcloud components install kubectl
RUN gcloud components install gke-gcloud-auth-plugin
ENV JAX_PLATFORM_NAME=TPU
ENV XLA_FLAGS="--xla_dump_to=/tmp/xla_dump/" 
RUN git clone https://github.com/google/maxtext.git 
RUN cd maxtext && bash docker_build_dependency_image.sh MODE=nightly
