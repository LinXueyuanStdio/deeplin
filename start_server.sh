#!/bin/bash

# Start the FastAPI server

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with the following variables:"
    echo "HITHINK_APP_URL=your_app_url"
    echo "HITHINK_APP_ID=your_app_id"
    echo "HITHINK_APP_SECRET=your_app_secret"
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -e .

# Start the server
echo "Starting FastAPI server..."
python -m deeplin.inference_engine.hexin_server
