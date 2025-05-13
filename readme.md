# Planogram Detection

A FastAPI application with Streamlit interface for planogram detection using YOLO, CLIP, and Supabase.

## Features

- Product detection using YOLO
- Product identification using multiple approaches:
  - Database matching with CLIP embeddings and Zero-shot classification with CLIP
- Planogram compliance analysis
- RESTful API endpoints
- Image processing and analysis
- User-friendly Streamlit web interface

## Setup

1. Install Poetry
2. Install dependencies:
   
bash
   poetry install

3. Run the application (API server only):
   
bash
   poetry run start

4. Run the Streamlit interface (in a separate terminal):
   
bash
   poetry run streamlit

5. Run both API server and Streamlit interface together:
   
bash
   poetry run app


## Environment Variables

Create a .env file with the following variables:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_KEY: Your Supabase API key
- API_URL: URL for the FastAPI server (default: http://localhost:8000)
- SAVE_ANNOTATED_IMAGES=true

## Streamlit Interface

The application includes a user-friendly Streamlit interface with the following features:

### Product Detection
- Upload images for product detection using database matching and zero-shot classification
- Configure detection parameters
- View annotated results with detected products

### Product Management
- Add new products 
- the product will crop using YOLO and embedding using CLIP and store to vector database
- Add variants to existing products
- View all products and variants in the database

### Planogram Analysis
- Upload actual planogram images
- Define expected planogram layouts
- Analyze compliance between actual and expected layouts
- View detailed compliance reports

## API Endpoints

### Product Detection

- POST /detect/: Detect products using database matching using database embedding and zero shot

### Product Management

- POST /products/: Add a new product
- GET /products/: Get all products

## Zero-Shot Classification

The application now supports zero-shot classification for product identification without requiring pre-stored examples. This allows detection of new products by providing their names at inference time.

## Planogram Analysis

The planogram analysis feature compares detected products with an expected layout to:
- Verify correct product placement
- Calculate compliance scores
- Identify missing or misplaced products

## Variant Detection

Products can be detected with their specific variants using three methods:
- **Embedding-based**: Uses CLIP to compare images with variant descriptions