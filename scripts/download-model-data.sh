# Sync data to google cloud storage
#
# Usage:
#   ./scripts/sync-data.sh

# Set variables from .env file
if [ -f .env ]; then
  echo "Loading .env file..."
  export $(cat .env | xargs)
fi

# Check if the environment variables are set
if [ -z "$GCP_BUCKET_NAME" ]; then
  echo "GCP_BUCKET_NAME is not set"
  exit 1
fi

# Sync the data to Google Cloud Storage
echo "Downloading data from Google Cloud Storage..."
gsutil -m rsync -r gs://$GCP_BUCKET_NAME/models/ data/models/
gsutil cp gs://$GCP_BUCKET_NAME/optuna.db data/optuna.db
echo "Complete."
