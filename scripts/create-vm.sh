gcloud compute instances create \
deep-learning-gpu-01 \
--project=graph-learning-356920 \
--zone=europe-west1-b \
--machine-type=n1-standard-4 \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--provisioning-model=SPOT \
--instance-termination-action=STOP \
--service-account=368714207763-compute@developer.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
--accelerator=count=1,type=nvidia-tesla-t4 \
--tags=http-server,https-server \
--create-disk=auto-delete=yes,boot=yes,device-name=deep-learning-gpu-01,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20220701-debian-10,mode=rw,size=50,type=projects/graph-learning-356920/zones/europe-west1-b/diskTypes/pd-balanced \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any