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

## Planogram Image Matching Improvements

The planogram image matching system has been enhanced with the following features:

### 1. Data Augmentation Pipeline
- Automatically generates multiple variations of product images (rotation, scaling, brightness/contrast adjustments)
- Creates embeddings for each variation to improve robustness in matching
- Supports configurable number of augmentations per image

### 2. Enhanced Metadata and Database Schema
- Added support for rich product metadata (category, brand, color, dimensions, etc.)
- Created separate table for storing augmented embeddings
- Updated database functions for efficient filtering and search

### 3. Hybrid Re-ranking System
- Vector similarity search combined with metadata filtering
- OCR text extraction for text-based matching
- Spatial context awareness for better placement analysis
- Weighted combination of multiple signals for more accurate matching

### 4. Multiple Image Upload Support
- API endpoints for uploading multiple product images
- Background processing for efficient embedding generation
- Optimized augmentation strategy for multiple images

### 5. Improved Visualization
- Enhanced annotations with OCR text display
- Color-coded status indicators (correct, misplaced, missing, extra)
- Better error handling and detailed logging

## Usage Examples

### Uploading a Product with Metadata
```bash
curl -X POST "http://localhost:8000/products/" \
  -H "Content-Type: multipart/form-data" \
  -F "name=Product Name" \
  -F "variant=Regular" \
  -F "category=Beverages" \
  -F "brand=SampleBrand" \
  -F "color=Red" \
  -F "barcode=1234567890" \
  -F "tags=drink,soda,cola" \
  -F "image_upload=@/path/to/product_image.jpg"
```

### Uploading Multiple Images for a Product
```bash
curl -X POST "http://localhost:8000/products/upload-images/" \
  -H "Content-Type: multipart/form-data" \
  -F "product_id=1" \
  -F "images=@/path/to/image1.jpg" \
  -F "images=@/path/to/image2.jpg" \
  -F "images=@/path/to/image3.jpg"
```

### Comparing a Planogram with Metadata Filtering
```bash
curl -X POST "http://localhost:8000/planogram/compare/" \
  -H "Content-Type: application/json" \
  -d '{
    "actual_image": "base64_encoded_image",
    "expected_layout": [
      {"product_id": 1, "name": "Product 1", "expected_box": [10, 10, 100, 100]},
      {"product_id": 2, "name": "Product 2", "expected_box": [120, 10, 220, 100]}
    ],
    "metadata_filters": {
      "category": "Beverages",
      "brand": "SampleBrand"
    }
  }'
```