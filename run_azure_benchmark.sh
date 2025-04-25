#!/bin/bash

# Set Azure AI Search credentials
# Replace these with your actual Azure AI Search credentials
export AZURE_SEARCH_ENDPOINT=""
export AZURE_SEARCH_KEY=""
export AZURE_SEARCH_INDEX_NAME=""

# Run the benchmark
python VectorDBBench/azure_search_real.py
