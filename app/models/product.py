from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProductBase(BaseModel):
    name: str
    variant: Optional[str] = None
    image_url: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    id: int
    clip_embedding: Optional[List[float]] = None
    created_at: datetime
    similarity: Optional[float] = None # Added for results from embedding search

    class Config:
        orm_mode = True

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image

class DetectionResponse(BaseModel):
    products: List[Product]
    annotated_image: Optional[str] = None # Base64 encoded image with boxes

class PlanogramComparisonRequest(BaseModel):
    actual_image: str  # Base64 encoded image of the current shelf
    expected_layout: List[dict] # List of expected products with their locations

class PlanogramComparisonResponse(BaseModel):
    compliance_score: float
    misplaced_products: List[dict]
    missing_products: List[dict]
    annotated_image: Optional[str] = None # Base64 encoded image 