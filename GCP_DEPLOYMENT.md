# Google Cloud Run Deployment Guide

Complete guide for deploying the Fraud Detection API to Google Cloud Run.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Initial Setup](#step-1-initial-setup)
- [Step 2: Prepare the Application](#step-2-prepare-the-application)
- [Step 3: Build and Push Docker Image](#step-3-build-and-push-docker-image)
- [Step 4: Deploy to Cloud Run](#step-4-deploy-to-cloud-run)
- [Step 5: Get Deployment URL](#step-5-get-deployment-url)
- [Step 6: Test the Deployment](#step-6-test-the-deployment)
- [Step 7: Configure Production Settings](#step-7-configure-production-settings-optional)
- [Step 8: Monitor and Maintain](#step-8-monitor-and-maintain)
- [Cost Estimation](#cost-estimation)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)

## Overview

**Why Cloud Run?**
- **Serverless**: No infrastructure management required
- **Auto-scaling**: Scales from 0 to 1000+ instances based on traffic
- **Pay-per-use**: Only charged when handling requests (generous free tier)
- **Fast cold starts**: Typically <1 second for this container
- **HTTPS by default**: Automatic SSL certificates

## Prerequisites

1. Google Cloud account ([free tier available](https://cloud.google.com/free))
2. Google Cloud SDK ([installation guide](https://cloud.google.com/sdk/docs/install))
3. Trained model artifacts in `models/` directory

---

## Step 1: Initial Setup

```bash
# Install Google Cloud SDK (if not already installed)
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate with Google Cloud
gcloud auth login

# Create a new project (or use existing)
gcloud projects create fraud-detection-api-prod --name="Fraud Detection API"

# Set the project as default
gcloud config set project fraud-detection-api-prod

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

---

## Step 2: Prepare the Application

```bash
# Ensure you're in the project directory
cd /path/to/e-commerce-fraud-detection

# Verify model artifacts exist
ls -lh models/*.json models/*.joblib

# Expected files:
# - models/xgb_fraud_detector.joblib (trained model - ~156KB)
# - models/transformer_config.json (feature engineering config)
# - models/threshold_config.json (prediction thresholds)
# - models/model_metadata.json (model version and metrics)
# - models/feature_lists.json (feature names and categories)
```

---

## Step 3: Build and Push Docker Image

### Option A: Using Cloud Build (Recommended)

Builds the image in Google Cloud - faster and doesn't require local Docker.

```bash
# Set your project ID
export PROJECT_ID=fraud-detection-api-prod

# Build the image using Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/fraud-detection-api

# This command:
# - Uploads your code to Google Cloud
# - Builds the Docker image in the cloud (faster, no local Docker needed)
# - Pushes to Google Container Registry automatically
```

### Option B: Using Local Docker

Alternative approach if you prefer building locally.

```bash
# Set your project ID
export PROJECT_ID=fraud-detection-api-prod

# Build the image locally
docker build -t gcr.io/$PROJECT_ID/fraud-detection-api .

# Configure Docker to use gcloud for authentication
gcloud auth configure-docker

# Push the image to Google Container Registry
docker push gcr.io/$PROJECT_ID/fraud-detection-api
```

---

## Step 4: Deploy to Cloud Run

```bash
# Deploy the container to Cloud Run
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 60 \
  --max-instances 1 \
  --min-instances 0 \
  --concurrency 80

# Explanation of flags:
# --image: The container image to deploy
# --platform managed: Use fully managed Cloud Run (serverless)
# --region: Geographic region for deployment
# --allow-unauthenticated: Allow public access (remove for private API)
# --memory: Allocate 2GB RAM (model + XGBoost requires ~1.5GB)
# --cpu: 1 vCPU for baseline performance
# --timeout: 60 second request timeout
# --max-instances: Maximum concurrent instances (cost control)
# --min-instances: Keep 0 for cost savings (scales to 0 when idle)
# --concurrency: Maximum concurrent requests per instance
```

**Expected Output:**
```
Deploying container to Cloud Run service [fraud-detection-api]...
✓ Deploying... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
Done.
Service [fraud-detection-api] revision [fraud-detection-api-00001-abc] has been deployed.
Service URL: https://fraud-detection-api-xxxxxxxxxx-uc.a.run.app
```

---

## Step 5: Get Deployment URL

```bash
# Retrieve the service URL
gcloud run services describe fraud-detection-api \
  --region us-west1 \
  --format 'value(status.url)'

# Save URL to environment variable
export SERVICE_URL=$(gcloud run services describe fraud-detection-api \
  --region us-west1 \
  --format 'value(status.url)')

echo "Service deployed at: $SERVICE_URL"
```

---

## Step 6: Test the Deployment

```bash
# Test health endpoint
curl $SERVICE_URL/health | python -m json.tool

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_version": "1.0",
#   "uptime_seconds": 12.34,
#   "timestamp": "2025-11-16T..."
# }

# Test prediction endpoint: "is_fraud": false
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "account_age_days": 180,
    "total_transactions_user": 25,
    "avg_amount_user": 250.50,
    "amount": 850.75,
    "country": "US",
    "bin_country": "US",
    "channel": "web",
    "merchant_category": "retail",
    "promo_used": 0,
    "avs_match": 1,
    "cvv_result": 1,
    "three_ds_flag": 1,
    "shipping_distance_km": 12.5,
    "transaction_time": "2024-01-15 14:30:00"
  }' | python -m json.tool

# Test prediction endpoint: "is_fraud": true
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "account_age_days": 180,
    "total_transactions_user": 25,
    "avg_amount_user": 250.50,
    "amount": 850.75,
    "country": "US",
    "bin_country": "US",
    "channel": "web",
    "merchant_category": "retail",
    "promo_used": 0,
    "avs_match": 0,
    "cvv_result": 0,
    "three_ds_flag": 0,
    "shipping_distance_km": 1222.5,
    "transaction_time": "2024-01-15 14:30:00"
  }' | python -m json.tool

# Access interactive API documentation
open "$SERVICE_URL/docs"
```

---

## Step 7: Configure Production Settings (Optional)

### Enable Authentication

Recommended for production deployments to control access.

```bash
# Deploy with authentication required
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --no-allow-unauthenticated

# Create a service account for API access
gcloud iam service-accounts create fraud-api-client \
  --display-name "Fraud API Client"

# Grant the service account permission to invoke the service
gcloud run services add-iam-policy-binding fraud-detection-api \
  --region us-west1 \
  --member "serviceAccount:fraud-api-client@$PROJECT_ID.iam.gserviceaccount.com" \
  --role "roles/run.invoker"

# Generate authentication token
gcloud auth print-identity-token

# Use token in requests
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/health
```

### Set Environment Variables

```bash
# Deploy with custom environment variables
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --set-env-vars "MODEL_VERSION=1.0,LOG_LEVEL=INFO"
```

### Configure Auto-scaling

```bash
# Keep 1 instance warm to eliminate cold starts (costs more)
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --min-instances 1 \
  --max-instances 10

# Scale based on concurrent requests (default: 80)
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --concurrency 50
```

---

## Step 8: Monitor and Maintain

### View Logs

```bash
# Stream logs in real-time
gcloud run services logs tail fraud-detection-api --region us-west1

# View recent logs
gcloud run services logs read fraud-detection-api \
  --region us-west1 \
  --limit 50
```

### View Metrics

```bash
# Open Cloud Console metrics dashboard
gcloud run services describe fraud-detection-api \
  --region us-west1 \
  --format 'value(status.url)' | \
  sed 's|https://||' | \
  xargs -I {} open "https://console.cloud.google.com/run/detail/us-west1/fraud-detection-api/metrics?project=$PROJECT_ID"

# Or view service details in terminal (includes basic metrics)
gcloud run services describe fraud-detection-api \
  --region us-west1 \
  --format yaml

# View recent request logs with activity information
gcloud run services logs read fraud-detection-api \
  --region us-west1 \
  --limit 50
```

### Update Deployment

```bash
# Rebuild and redeploy after code changes
gcloud builds submit --tag gcr.io/$PROJECT_ID/fraud-detection-api

gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1

# Cloud Run will:
# - Create a new revision
# - Gradually shift traffic to the new revision
# - Keep old revision for rollback if needed
```

### Rollback to Previous Version

```bash
# List all revisions
gcloud run revisions list --service fraud-detection-api --region us-west1

# Rollback to specific revision
gcloud run services update-traffic fraud-detection-api \
  --region us-west1 \
  --to-revisions fraud-detection-api-00001-abc=100
```

---

## Cost Estimation

**Free Tier (Monthly - as of 2025):**
- 2 million requests
- 360,000 GB-seconds of memory
- 180,000 vCPU-seconds
- 1 GB egress (North America)

**Your API (2GB RAM, 1 vCPU):**
- ~400ms average request time (including cold start)
- **Free tier covers ~5,000 requests/month** (assuming average 0.4s per request)
- After free tier: ~$0.00002 per request

**Example Monthly Costs:**
- 10,000 requests: **FREE** (within free tier)
- 100,000 requests: ~$2/month
- 1,000,000 requests: ~$20/month

**Cost Optimization Tips:**
- Use `--min-instances 0` to scale to zero when idle (default)
- Set `--max-instances` to prevent unexpected costs
- Monitor usage in [Cloud Console](https://console.cloud.google.com/billing)

---

## Troubleshooting

### Issue: "Container failed to start"

```bash
# Check build logs
gcloud builds log $(gcloud builds list --limit 1 --format 'value(id)')

# Check Cloud Run logs
gcloud run services logs read fraud-detection-api --region us-west1 --limit 100

# Common causes:
# - Missing model files in models/ directory
# - Port mismatch (Dockerfile EXPOSE must match uvicorn --port)
# - Insufficient memory (increase --memory to 4Gi)
```

### Issue: "Out of memory"

```bash
# Increase memory allocation
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --memory 4Gi
```

### Issue: "Cold start too slow"

```bash
# Keep minimum instances warm
gcloud run deploy fraud-detection-api \
  --image gcr.io/$PROJECT_ID/fraud-detection-api \
  --region us-west1 \
  --min-instances 1

# Note: This increases cost but eliminates cold starts
```

---

## Cleanup

**Delete Resources:**

```bash
# Delete the Cloud Run service
gcloud run services delete fraud-detection-api --region us-west1

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/fraud-detection-api

# Delete the entire project (WARNING: irreversible)
gcloud projects delete $PROJECT_ID
```

---

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Best Practices](https://cloud.google.com/run/docs/best-practices)
- [Monitoring Cloud Run Services](https://cloud.google.com/run/docs/monitoring)
