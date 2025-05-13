import asyncio
from typing import Optional, List
from PIL import Image
import io
import os # Added for environment variable
from pathlib import Path # Added for path manipulation and directory creation

# These will be imported properly once the project structure is fully in place
# and we ensure services are available in the PYTHONPATH for the worker context.
from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import supabase as supabase_service

# --- Environment Variable for Saving Product Embedding Input Images ---
SAVE_PRODUCT_EMBEDDING_IMAGES = os.getenv("SAVE_PRODUCT_EMBEDDING_IMAGES", "true").lower() == "true"
LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS = Path("product_embedding_inputs")
if SAVE_PRODUCT_EMBEDDING_IMAGES:
    LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    print(f"Worker: Will save product images used for embeddings to {LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS.resolve()}")

async def embed_and_update_product_task(product_id: int, image_bytes: Optional[bytes]):
    """
    Background task to generate embedding for a product image and update the database.
    Optionally crops the image using YOLO before embedding.
    Saves the image used for embedding if SAVE_PRODUCT_EMBEDDING_IMAGES is true.
    """
    if not image_bytes:
        print(f"Task: No image bytes provided for product ID: {product_id}. Skipping embedding.")
        return

    print(f"Task: Starting embedding process for product ID: {product_id}")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_image_for_embedding = pil_image # Default to original image
        image_type_for_log = "original" # For logging

        # 1. Optional: YOLO crop
        try:
            detections = await asyncio.to_thread(detection_service.detect_objects, image_bytes)
            if detections:
                print(f"Task: Detected {len(detections)} objects for product ID: {product_id}")
                # Select the largest detected object by area
                largest_detection = max(detections, key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
                best_box = largest_detection['box']
                
                cropped_image = await asyncio.to_thread(detection_service.crop_object, pil_image, best_box)
                target_image_for_embedding = cropped_image
                image_type_for_log = "YOLO-cropped"
                print(f"Task: Cropped image for product ID: {product_id} using box {best_box}")
            else:
                print(f"Task: No objects detected by YOLO for product ID: {product_id}. Using original image.")
        except Exception as e_crop:
            print(f"Task: Error during YOLO detection/cropping for product ID {product_id}: {e_crop}. Using original image.")

        # --- Save the image that will be used for CLIP embedding ---
        if SAVE_PRODUCT_EMBEDDING_IMAGES:
            try:
                save_filename = LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS / f"product_{product_id}_{image_type_for_log}_clip_input.jpg"
                await asyncio.to_thread(target_image_for_embedding.save, save_filename, "JPEG")
                print(f"Task: Saved {image_type_for_log} image for product ID {product_id} to {save_filename}")
            except Exception as e_save:
                print(f"Task: Error saving {image_type_for_log} image for product ID {product_id}: {e_save}")
        # -------------------------------------------------------------

        # 2. Generate CLIP embedding
        clip_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, target_image_for_embedding)
        if not clip_embedding:
            print(f"Task: Failed to generate CLIP embedding for product ID: {product_id}")
            return
        
        print(f"Task: Generated CLIP embedding for product ID: {product_id}")

        # 3. Update Supabase with the embedding
        await supabase_service.update_product_embedding(product_id, clip_embedding)
        print(f"Task: Successfully updated product ID: {product_id} with new embedding.")

    except Exception as e:
        print(f"Task: Error in embed_and_update_product_task for product ID {product_id}: {e}")
        # Add more robust error handling/logging as needed 