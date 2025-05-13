from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends, Form
from typing import List, Optional
import io # For reading image bytes
import base64 # For decoding image from request
from PIL import Image
import asyncio # Added for asyncio.to_thread

from app.models.product import (Product, ProductCreate, DetectionRequest, 
                                DetectionResponse, PlanogramComparisonRequest, 
                                PlanogramComparisonResponse)
from app.services import supabase as supabase_service
from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import planogram as planogram_service # Uncommented
from app.workers import tasks as worker_tasks

app = FastAPI(title="Planogram Detection API")

IDENTIFICATION_THRESHOLD_DETECT = 0.7 # Default confidence for identifying a detected object via embedding

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
        identified_annotations_for_drawing: List[dict] = [] # New list for drawing

        if detected_raw_objects:
            print(f"/detect: Found {len(detected_raw_objects)} raw objects.")
            for det_obj in detected_raw_objects:
                actual_box = det_obj['box']
                try:
                    cropped_pil_image = detection_service.crop_object(pil_image, actual_box)
                    img_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, cropped_pil_image)
                    
                    matched_db_products = await supabase_service.find_products_by_embedding(
                        embedding=img_embedding, 
                        match_threshold=IDENTIFICATION_THRESHOLD_DETECT, 
                        match_count=1
                    )
                    
                    if matched_db_products:
                        best_match_product = matched_db_products[0]
                        product_similarity = getattr(best_match_product, 'similarity', None)

                        # Always add to drawing list if identified, regardless of uniqueness for the API response list
                        identified_annotations_for_drawing.append({
                            'box': actual_box,
                            'label': best_match_product.name, # Use product name from DB
                            'confidence': product_similarity, # Use similarity score for display
                            'color': 'green' # Identified products are green
                        })
                        print(f"/detect: Matched raw box {actual_box} to DB product: {best_match_product.name} (ID: {best_match_product.id}) with similarity: {product_similarity if product_similarity is not None else 'N/A'}")

                        # Add to the main API response list only if it's a new unique product ID
                        if best_match_product.id not in unique_identified_product_ids:
                            identified_products_list.append(best_match_product)
                            unique_identified_product_ids.add(best_match_product.id)
                            # The detailed print for "Identified: ..." will now be less frequent, 
                            # as the matching itself is logged above for every drawn box.
                        # else:
                             # No specific log needed here now, as the box is drawn and matching is logged.
                except Exception as e_identify:
                    print(f"/detect: Error identifying object at {actual_box}: {e_identify}")
        else:
            print("/detect: No raw objects detected.")

        # 3. Annotate image ONLY with successfully identified products
        annotated_image_b64 = request.image # Default to original if no identified annotations
        if identified_annotations_for_drawing: # Only draw if there's something identified and to be drawn
            annotated_image_bytes = await asyncio.to_thread(
                detection_service.draw_boxes_on_image, 
                image_bytes, 
                identified_annotations_for_drawing # Pass the new list of identified items
            )
            annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')
        else:
            print("/detect: No products were identified and confirmed for annotation.")


        return DetectionResponse(products=identified_products_list, annotated_image=annotated_image_b64)

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image string.")
    except Exception as e:
        print(f"Error in /detect/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing detection: {str(e)}")

@app.post("/products/", response_model=Product, status_code=201)
async def add_product_api(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    variant: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None), # User can provide an image URL
    image_upload: Optional[UploadFile] = File(None) # Or upload an image file
):
    product_data = ProductCreate(name=name, variant=variant, image_url=image_url)
    
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
                    image_bytes
                )
            else:
                print(f"Uploaded image for product {created_product.id} is empty, skipping embedding task.")
        elif image_url:
            # If no file is uploaded but an image_url is provided, 
            # the background task could potentially fetch and process this URL.
            # For now, embed_and_update_product_task expects bytes. This part can be enhanced.
            print(f"Image URL provided for product {created_product.id}, but background task currently requires image bytes. URL: {image_url}")
            # If you want to process URL in background, the task needs to be adapted to fetch it.
            # For now, only uploaded images trigger the embedding task.

        return created_product
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=400, detail=f"Error creating product: {str(e)}")

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
        # Ensure expected_layout is in the format required by the service:
        # List of {'product_id': int, 'name': str, 'expected_box': [x1,y1,x2,y2]}
        # This might require transformation if the input from client is different.
        # For now, we assume it's correct.
        print(f"Planogram Compare API: Received {len(request.expected_layout)} expected items.")
        # Example: first item expected: {request.expected_layout[0] if request.expected_layout else 'None'}

        analysis_result = await planogram_service.compare_planogram_layouts(
            actual_image_bytes=actual_image_bytes,
            expected_layout=request.expected_layout
            # Can also pass iou_threshold and identification_threshold if needed from request
        )

        return PlanogramComparisonResponse(
            compliance_score=analysis_result.get("compliance_score", 0.0),
            misplaced_products=analysis_result.get("misplaced_products", []),
            missing_products=analysis_result.get("missing_products", []),
            annotated_image=analysis_result.get("annotated_image_b64")
        )
    except Exception as e:
        print(f"Error during planogram comparison: {e}")
        # Consider logging the full traceback for e
        raise HTTPException(status_code=500, detail=f"Error during planogram analysis: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
