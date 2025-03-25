# distMatrix Backend - Railway Deployment Guide

This guide explains how to deploy the distMatrix backend application to Railway.

## Prerequisites

- A Railway account (sign up at [railway.app](https://railway.app) if you don't have one)
- Railway CLI (optional, for CLI deployment)
- Git repository with your distMatrix backend code

## Deployment Options

### Option 1: Deploy via GitHub

1. Push your backend code to a GitHub repository (you can create a separate repo just for the backend)
2. Log in to [Railway Dashboard](https://railway.app/dashboard)
3. Click "New Project" > "Deploy from GitHub repo"
4. Select your distMatrix backend repository
5. Railway will automatically detect it's a Python app and deploy it
6. Once deployed, go to the "Settings" tab > "Networking" section
7. Click "Generate Domain" to create a public URL for your backend API

### Option 2: Deploy via Railway CLI

1. Install the Railway CLI:
   ```
   npm i -g @railway/cli
   ```

2. Login to Railway:
   ```
   railway login
   ```

3. Navigate to your backend directory and initialize a Railway project:
   ```
   cd backend
   railway init
   ```

4. Deploy the application:
   ```
   railway up
   ```

5. Generate a public domain:
   ```
   railway domain
   ```

## Environment Variables

Make sure to set the following environment variables in Railway:

- `GOOGLE_MAPS_API_KEY`: Your Google Maps API key
- `PORT`: Railway will set this automatically, but you can override it if needed
- `RAILWAY_ENVIRONMENT`: Set to "production"

## Frontend Integration

After deploying your backend to Railway, you'll need to update your frontend configuration to point to the new backend URL:

1. Update the API endpoint URLs in your frontend code to use the Railway-generated domain
2. Make sure CORS is properly configured in the backend to allow requests from your frontend domain

## Project Structure

The deployment is configured using:
- `nixpacks.toml`: Tells Railway how to build and run the app
- `Procfile`: Alternative way to specify the startup command
- Modified `app.py`: Handles both development and production environments

## Troubleshooting

- Check Railway logs if the app fails to deploy
- Ensure all dependencies are in `requirements.txt`
- Verify that the Google Maps API key is set in Railway environment variables
