FROM python:3.10-slim

# Install gcloud cli tools: gcloud and gcsfuse
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl fuse && \
    curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz && \
	mkdir -p /usr/local/gcloud && \
	tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz && \
	/usr/local/gcloud/google-cloud-sdk/install.sh && \
	rm /tmp/google-cloud-sdk.tar.gz && \
	/usr/local/gcloud/google-cloud-sdk/bin/gcloud components install kubectl && \
	/usr/local/gcloud/google-cloud-sdk/bin/gcloud components install gke-gcloud-auth-plugin && \
    curl https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.1.0/gcsfuse_1.1.0_amd64.deb -o gcsfuse_1.1.0_amd64.deb -L && \
    dpkg -i gcsfuse_1.1.0_amd64.deb && \
    rm gcsfuse_1.1.0_amd64.deb

ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Copy HF model configs from GCS
RUN mkdir /configs && gsutil -m rsync -r gs://optimus-prime-configs /configs

# Baseline dependencies
RUN apt-get update && \
    apt-get install -y git libopenblas-dev libomp5 libgomp1 iproute2 sudo ethtool
