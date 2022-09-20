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
  echo "BUCKET_NAME is not set"
  exit 1
fi

# Sync the data to Google Cloud Storage
echo "Syncing data to Google Cloud Storage..."
# gsutil -m rsync -x 'raw.*|interim.*|graph.*' -r data gs://$GCP_BUCKET_NAME
gsutil -m rsync -x 'interim.*' -r data gs://$GCP_BUCKET_NAME
echo "Complete."
