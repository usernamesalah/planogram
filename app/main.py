from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends, Form
from typing import List, Optional, Dict, Any
import io # For reading image bytes
import base64 # For decoding image from request
from PIL import Image
import asyncio # Added for asyncio.to_thread

from app.models.product import (Product, ProductCreate, DetectionRequest, 
                                DetailedDetectionRequest, ProductImageUpload,
                                DetectionResponse, PlanogramComparisonRequest, 
                                PlanogramComparisonResponse)
from app.services import supabase as supabase_service
from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import planogram as planogram_service
from app.workers import tasks as worker_tasks

app = FastAPI(title="Planogram Detection API")

# Default confidence for identifying a detected object via embedding
IDENTIFICATION_THRESHOLD_DETECT = 0.8  # Lowered from 0.7 to 0.6 to capture more similar products

@app.post("/detect/", response_model=DetectionResponse)
async def detect_products_api(request: DetectionRequest):
    print(f"Received image for detection (first 100 chars): {request.image[:100]}")
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image provided for detection.")
        
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        detected_raw_objects = await asyncio.to_thread(detection_service.detect_objects, image_bytes)
        
        identified_products_list: List[Product] = []
        unique_identified_product_ids = set()
        identified_annotations_for_drawing: List[dict] = []

        if detected_raw_objects:
            print(f"/detect: Found {len(detected_raw_objects)} raw objects.")
            for det_obj in detected_raw_objects:
                actual_box = det_obj['box']
                try:
                    # Get the cropped object from the detection
                    cropped_pil_image = detection_service.crop_object(pil_image, actual_box)
                    
                    # Try alternate cropping methods if the first match isn't good enough
                    alternate_crops = []
                    
                    # 1. Try smart cropping (edges/contours based)
                    try:
                        smart_cropped, _ = detection_service.smart_crop(cropped_pil_image.copy())
                        alternate_crops.append(("smart", smart_cropped))
                    except Exception as e:
                        print(f"/detect: Smart cropping failed: {e}")
                    
                    # 2. Try color-based cropping (for colorful packaging)
                    try:
                        color_cropped, _ = detection_service.color_based_crop(cropped_pil_image.copy())
                        alternate_crops.append(("color", color_cropped))
                    except Exception as e:
                        print(f"/detect: Color cropping failed: {e}")
                    
                    # 3. Create color-enhanced version
                    try:
                        from app.services.augmentation import adjust_saturation, adjust_contrast
                        enhanced = adjust_saturation(cropped_pil_image.copy(), 1.5)
                        enhanced = adjust_contrast(enhanced, 1.2)
                        alternate_crops.append(("enhanced", enhanced))
                    except Exception as e:
                        print(f"/detect: Enhancement failed: {e}")
                    
                    # Original crop is always included
                    alternate_crops.append(("original", cropped_pil_image))
                    
                    # Generate embedding and find best match for each crop variant
                    best_match = None
                    best_match_score = 0
                    best_match_type = None
                    
                    for crop_type, crop_image in alternate_crops:
                        try:
                            # Get embedding for this crop
                            crop_embedding = await asyncio.to_thread(
                                embedding_service.get_image_embedding, 
                                crop_image
                            )
                            
                            # Get product matches
                            matches = await supabase_service.find_products_by_embedding(
                                embedding=crop_embedding,
                                match_threshold=IDENTIFICATION_THRESHOLD_DETECT * 0.85,
                                match_count=5,
                                use_augmentations=True
                            )
                            
                            if matches and len(matches) > 0:
                                top_match = matches[0]
                                match_score = getattr(top_match, 'similarity', 0)
                                
                                # Keep track of best match across all crop types
                                if match_score > best_match_score:
                                    best_match = top_match
                                    best_match_score = match_score
                                    best_match_type = crop_type
                                    print(f"/detect: Found better match with {crop_type} crop: {match_score:.3f}")
                        except Exception as e:
                            print(f"/detect: Error processing {crop_type} crop: {e}")
                    
                    # If we found a good match with any crop type
                    if best_match and best_match_score >= IDENTIFICATION_THRESHOLD_DETECT:
                        # OCR functionality disabled
                        # Previously OCR was used here to:
                        # 1. Extract text from images
                        # 2. Apply boosting for name/brand matches
                        # 3. Improve detection confidence
                        
                        print(f"/detect: Best match found with {best_match_type} crop: {best_match.name} (score: {best_match_score:.3f})")
                        
                        # Add to drawing list
                        identified_annotations_for_drawing.append({
                            'box': actual_box,
                            'label': f"{best_match.name} ({best_match_score:.2f})",
                            'confidence': best_match_score,
                            'color': 'green',
                            'product_id': best_match.id  # Add product_id for consistent coloring
                        })
                        
                        # Add to product list if not already included
                        if best_match.id not in unique_identified_product_ids:
                            identified_products_list.append(best_match)
                            unique_identified_product_ids.add(best_match.id)
                    else:
                        print(f"/detect: No good match found for object at {actual_box}")
                        
                except Exception as e_identify:
                    print(f"/detect: Error identifying object at {actual_box}: {e_identify}")
        else:
            print("/detect: No raw objects detected.")

        # 3. Annotate image ONLY with successfully identified products
        annotated_image_b64 = request.image
        if identified_annotations_for_drawing:
            annotated_image_bytes = await asyncio.to_thread(
                detection_service.draw_boxes_on_image, 
                image_bytes, 
                identified_annotations_for_drawing
            )
            annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')

        if not identified_products_list:
            print("/detect: No products were identified and confirmed for annotation.")

        return DetectionResponse(products=identified_products_list, annotated_image=annotated_image_b64)

    except Exception as e:
        print(f"Error in /detect/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing detection: {str(e)}")

@app.post("/detect/detailed/", response_model=DetectionResponse)
async def detect_products_with_filters_api(request: DetailedDetectionRequest):
    """Enhanced detection endpoint that supports metadata filtering"""
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image provided for detection.")
        
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        detected_raw_objects = await asyncio.to_thread(detection_service.detect_objects, image_bytes)
        
        identified_products_list: List[Product] = []
        unique_identified_product_ids = set()
        identified_annotations_for_drawing: List[dict] = []

        if detected_raw_objects:
            print(f"/detect/detailed: Found {len(detected_raw_objects)} raw objects.")
            for det_obj in detected_raw_objects:
                actual_box = det_obj['box']
                try:
                    cropped_pil_image = detection_service.crop_object(pil_image, actual_box)
                    img_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, cropped_pil_image)
                    
                    # Use metadata filters if provided
                    matched_db_products = await supabase_service.find_products_by_embedding(
                        embedding=img_embedding, 
                        match_threshold=IDENTIFICATION_THRESHOLD_DETECT * 0.9,  # Slightly lower threshold 
                        match_count=10,  # Get more candidates for re-ranking
                        metadata_filters=request.filter_by_metadata,
                        use_augmentations=True
                    )
                    
                    if matched_db_products:
                        # OCR functionality disabled
                        # Previously OCR was used here to extract text for re-ranking
                        
                        # Re-rank candidates using hybrid approach
                        reranked_products = await asyncio.to_thread(
                            planogram_service.hybrid_rerank,
                            candidates=matched_db_products,
                            query_box=actual_box,
                            query_embedding=img_embedding,
                            ocr_text=None
                        )
                        
                        if reranked_products:
                            best_match_product, best_match_score = reranked_products[0]
                            
                            if best_match_score >= IDENTIFICATION_THRESHOLD_DETECT:
                                # Add to drawing list
                                identified_annotations_for_drawing.append({
                                    'box': actual_box,
                                    'label': f"{best_match_product.name} ({best_match_score:.2f})",
                                    'confidence': best_match_score,
                                    'color': 'green',
                                    'product_id': best_match_product.id  # Add product_id for consistent coloring
                                })
                                
                                # Add to response list if unique
                                if best_match_product.id not in unique_identified_product_ids:
                                    identified_products_list.append(best_match_product)
                                    unique_identified_product_ids.add(best_match_product.id)
                except Exception as e_identify:
                    print(f"/detect/detailed: Error processing object at {actual_box}: {e_identify}")

        # Annotate image
        annotated_image_b64 = request.image
        if identified_annotations_for_drawing:
            annotated_image_bytes = await asyncio.to_thread(
                detection_service.draw_boxes_on_image, 
                image_bytes, 
                identified_annotations_for_drawing
            )
            annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')

        return DetectionResponse(products=identified_products_list, annotated_image=annotated_image_b64)

    except Exception as e:
        print(f"Error in /detect/detailed/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing detailed detection: {str(e)}")

@app.post("/products/", response_model=Product, status_code=201)
async def add_product_api(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    variant: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None), # User can provide an image URL
    image_upload: Optional[UploadFile] = File(None), # Or upload an image file
    category: Optional[str] = Form(None),
    brand: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    barcode: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)  # Comma-separated tags
):
    product_data = ProductCreate(
        name=name, 
        variant=variant, 
        image_url=image_url,
        category=category,
        brand=brand,
        color=color,
        barcode=barcode,
        tags=tags.split(",") if tags else None
    )
    
    try:
        # Add product to DB first to get an ID
        created_product = await supabase_service.add_product_to_db(product_data)
        if not created_product:
            raise HTTPException(status_code=500, detail="Failed to create product in database.")

        # If an image file is uploaded, prioritize it for embedding
        if image_upload:
            image_bytes = await image_upload.read()
            if image_bytes:
                print(f"Scheduling background task for embedding uploaded image for product ID: {created_product.id}")
                background_tasks.add_task(
                    worker_tasks.embed_and_update_product_task, 
                    created_product.id, 
                    image_bytes,
                    generate_augmentations=True,
                    crop_method="color"  # Use color-based cropping for products like Kuaci Rebo
                )
            else:
                print(f"Uploaded image for product {created_product.id} is empty, skipping embedding task.")
        elif image_url:
            print(f"Image URL provided for product {created_product.id}, but background task currently requires image bytes.")

        return created_product
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating product: {str(e)}")

@app.post("/products/upload-images/")
async def upload_product_images_api(
    background_tasks: BackgroundTasks,
    product_id: int = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Upload multiple images for a single product.
    Each image will be processed and result in embeddings (original + augmented).
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided for upload.")
    
    try:
        # Check if product exists
        product = await supabase_service.get_product_by_id_from_db(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found.")
        
        # Read all image files
        image_bytes_list = []
        for img in images:
            img_bytes = await img.read()
            if img_bytes:
                image_bytes_list.append(img_bytes)
        
        if not image_bytes_list:
            raise HTTPException(status_code=400, detail="No valid images found in the upload.")
        
        # Process all images in the background
        background_tasks.add_task(
            worker_tasks.process_multiple_product_images,
            product_id,
            image_bytes_list,
            generate_augmentations=True
        )
        
        return {"status": "success", "message": f"Processing {len(image_bytes_list)} images for product ID: {product_id}"}
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multiple images: {str(e)}")

@app.get("/products/", response_model=List[Product])
async def get_products_api():
    try:
        products = await supabase_service.get_all_products_from_db()
        return products
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error fetching products: {str(e)}")

@app.post("/planogram/compare/", response_model=PlanogramComparisonResponse)
async def compare_planogram_api(request: PlanogramComparisonRequest):
    if not request.actual_image:
        raise HTTPException(status_code=400, detail="Actual image for planogram comparison is required.")
    if not request.expected_layout:
        raise HTTPException(status_code=400, detail="Expected layout for planogram comparison is required.")

    try:
        actual_image_bytes = base64.b64decode(request.actual_image)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 string for actual_image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding actual_image: {str(e)}")

    try:
        print(f"Planogram Compare API: Received {len(request.expected_layout)} expected items.")

        analysis_result = await planogram_service.compare_planogram_layouts(
            actual_image_bytes=actual_image_bytes,
            expected_layout=request.expected_layout,
            metadata_filters=request.metadata_filters,
            use_augmentations=True
        )

        return PlanogramComparisonResponse(
            compliance_score=analysis_result.get("compliance_score", 0.0),
            misplaced_products=analysis_result.get("misplaced_products", []),
            missing_products=analysis_result.get("missing_products", []),
            annotated_image=analysis_result.get("annotated_image_b64")
        )
    except Exception as e:
        print(f"Error during planogram comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error during planogram analysis: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
