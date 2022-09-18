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
fi

# Download data from Google Cloud Storage
echo "Downloading data from Google Cloud Storage..."
gsutil -m rsync -r gs://$GCP_BUCKET_NAME data
echo "Complete."
