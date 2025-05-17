from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class ProductBase(BaseModel):
    name: str
    variant: Optional[str] = None
    image_url: Optional[str] = None
    # Additional metadata fields
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    barcode: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None  # e.g., {"width": 10.0, "height": 15.0, "depth": 5.0}
    tags: Optional[List[str]] = None
    
class ProductCreate(ProductBase):
    pass

class ProductEmbedding(BaseModel):
    id: int
    product_id: int
    embedding_vector: List[float]
    augmentation_type: Optional[str] = None  # e.g., "original", "rotation", "scale", etc.
    created_at: datetime
    
    class Config:
        orm_mode = True

class Product(ProductBase):
    id: int
    clip_embedding: Optional[List[float]] = None
    # List of augmented embeddings will be accessible via a separate endpoint
    augmented_embeddings_count: Optional[int] = Field(0, description="Number of augmented embeddings available")
    created_at: datetime
    similarity: Optional[float] = None # Added for results from embedding search

    class Config:
        orm_mode = True
        
class ProductWithEmbeddings(Product):
    embeddings: List[ProductEmbedding] = []

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    
class DetailedDetectionRequest(DetectionRequest):
    filter_by_metadata: Optional[Dict[str, str]] = None  # Optional metadata filters

class DetectionResponse(BaseModel):
    products: List[Product]
    annotated_image: Optional[str] = None # Base64 encoded image with boxes

class PlanogramComparisonRequest(BaseModel):
    actual_image: str  # Base64 encoded image of the current shelf
    expected_layout: List[dict] # List of expected products with their locations
    metadata_filters: Optional[Dict[str, str]] = None  # Optional metadata to filter candidates

class PlanogramComparisonResponse(BaseModel):
    compliance_score: float
    misplaced_products: List[dict]
    missing_products: List[dict]
    annotated_image: Optional[str] = None # Base64 encoded image
    
class ProductImageUpload(BaseModel):
    product_id: int
    image: str  # Base64 encoded image
    generate_augmentations: bool = True
    num_augmentations: int = 5

class OCRResult(BaseModel):
    text: str
    confidence: float
    box: List[int]  # x1, y1, x2, y2 