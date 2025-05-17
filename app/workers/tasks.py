import asyncio
from typing import Optional, List, Dict, Any
from PIL import Image
import io
import os # Added for environment variable
from pathlib import Path # Added for path manipulation and directory creation
import time
import uuid

# These will be imported properly once the project structure is fully in place
# and we ensure services are available in the PYTHONPATH for the worker context.
from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import supabase as supabase_service
from app.services import augmentation as augmentation_service

# --- Environment Variable for Saving Product Embedding Input Images ---
SAVE_PRODUCT_EMBEDDING_IMAGES = os.getenv("SAVE_PRODUCT_EMBEDDING_IMAGES", "true").lower() == "true"
LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS = Path("product_embedding_inputs")
if SAVE_PRODUCT_EMBEDDING_IMAGES:
    LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    print(f"Worker: Will save product images used for embeddings to {LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS.resolve()}")

async def embed_and_update_product_task(product_id: int, image_bytes: Optional[bytes], generate_augmentations: bool = True, num_augmentations: int = 10, skip_yolo_crop: bool = True):
    """
    Background task to generate embedding for a product image and update the database.
    Process the image with color enhancement for better detection of colorful packaging.
    Generates augmented versions of the image and stores their embeddings if requested.
    
    Args:
        product_id: The product ID to update
        image_bytes: The image bytes to process
        generate_augmentations: Whether to generate augmented versions of the image
        num_augmentations: Number of augmented versions to generate
        skip_yolo_crop: Whether to skip YOLO cropping (defaults to True)
    """
    if not image_bytes:
        print(f"Task: No image bytes provided for product ID: {product_id}. Skipping embedding.")
        return

    print(f"Task: Starting embedding process for product ID: {product_id}")
    timestamp = time.time()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Always process the original image
        original_image = pil_image
        
        # 1. Process original image
        print(f"Task: Processing original image for product ID: {product_id}")
        
        # Save original image if enabled
        if SAVE_PRODUCT_EMBEDDING_IMAGES:
            try:
                save_filename = LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS / f"product_{product_id}_original_clip_input.jpg"
                await asyncio.to_thread(original_image.save, save_filename, "JPEG")
                print(f"Task: Saved original image for product ID {product_id} to {save_filename}")
            except Exception as e_save:
                print(f"Task: Error saving original image for product ID {product_id}: {e_save}")
        
        # Generate CLIP embedding for original image
        original_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, original_image)
        if not original_embedding:
            print(f"Task: Failed to generate CLIP embedding for original image, product ID: {product_id}")
            return
        
        # Update Supabase with the original embedding
        await supabase_service.update_product_embedding(product_id, original_embedding)
        print(f"Task: Updated product ID: {product_id} with original image embedding.")
        
        # 2. Generate color-enhanced version for better detection of colorful packaging
        try:
            # Create a vibrant version by boosting saturation
            from app.services.augmentation import adjust_saturation
            enhanced_image = adjust_saturation(original_image, 1.4)  # Boost saturation for vibrant colors
            
            # Save enhanced image if enabled
            if SAVE_PRODUCT_EMBEDDING_IMAGES:
                try:
                    save_filename = LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS / f"product_{product_id}_enhanced_clip_input.jpg"
                    await asyncio.to_thread(enhanced_image.save, save_filename, "JPEG")
                    print(f"Task: Saved enhanced image for product ID {product_id} to {save_filename}")
                except Exception as e_save:
                    print(f"Task: Error saving enhanced image for product ID {product_id}: {e_save}")
            
            # Generate CLIP embedding for enhanced image
            enhanced_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, enhanced_image)
            if enhanced_embedding:
                # Add the enhanced image embedding as a special augmentation type
                await supabase_service.add_augmented_embedding(
                    product_id=product_id,
                    embedding=enhanced_embedding,
                    augmentation_type="enhanced_color"
                )
                print(f"Task: Added color-enhanced embedding for product ID: {product_id}")
            else:
                print(f"Task: Failed to generate CLIP embedding for enhanced image, product ID: {product_id}")
        except Exception as e_enhance:
            print(f"Task: Error creating color-enhanced version for product ID {product_id}: {e_enhance}")
        
        # Generate augmentations from original image
        if generate_augmentations:
            await process_augmentations(
                product_id=product_id,
                image=original_image,
                num_augmentations=num_augmentations,
                prefix="original"
            )
        
        # 3. Process YOLO-cropped image if not skipped (disabled by default)
        if not skip_yolo_crop:
            try:
                detections = await asyncio.to_thread(detection_service.detect_objects, image_bytes)
                if detections:
                    print(f"Task: Detected {len(detections)} objects for product ID: {product_id}")
                    # Select the largest detected object by area
                    largest_detection = max(detections, key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
                    best_box = largest_detection['box']
                    
                    cropped_image = await asyncio.to_thread(detection_service.crop_object, pil_image, best_box)
                    
                    # Save cropped image if enabled
                    if SAVE_PRODUCT_EMBEDDING_IMAGES:
                        try:
                            save_filename = LOCAL_SAVE_PATH_PRODUCT_EMBEDDINGS / f"product_{product_id}_yolo_cropped_clip_input.jpg"
                            await asyncio.to_thread(cropped_image.save, save_filename, "JPEG")
                            print(f"Task: Saved YOLO-cropped image for product ID {product_id} to {save_filename}")
                        except Exception as e_save:
                            print(f"Task: Error saving YOLO-cropped image for product ID {product_id}: {e_save}")
                    
                    # Generate CLIP embedding for cropped image and add as augmentation
                    cropped_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, cropped_image)
                    if cropped_embedding:
                        # Add the cropped image embedding as a special augmentation type
                        await supabase_service.add_augmented_embedding(
                            product_id=product_id,
                            embedding=cropped_embedding,
                            augmentation_type="yolo_cropped"
                        )
                        print(f"Task: Added YOLO-cropped embedding for product ID: {product_id}")
                        
                        # Generate augmentations from cropped image too
                        if generate_augmentations:
                            await process_augmentations(
                                product_id=product_id,
                                image=cropped_image,
                                num_augmentations=num_augmentations // 2,  # Split augmentations between original and cropped
                                prefix="yolo_cropped"
                            )
                    else:
                        print(f"Task: Failed to generate CLIP embedding for YOLO-cropped image, product ID: {product_id}")
                else:
                    print(f"Task: No objects detected by YOLO for product ID: {product_id}. Using original image only.")
            except Exception as e_crop:
                print(f"Task: Error during YOLO detection/cropping for product ID {product_id}: {e_crop}")
        else:
            print(f"Task: YOLO cropping skipped for product ID: {product_id}.")
            
        # Update the total augmentation count on the product
        augmented_embeddings_count = await supabase_service.get_augmented_embeddings_count(product_id)
        await supabase_service.update_product_augmentation_count(product_id, augmented_embeddings_count)
        print(f"Task: Updated augmentation count for product ID: {product_id} to {augmented_embeddings_count}")
        
        total_time = time.time() - timestamp
        print(f"Task: Completed embedding process for product ID: {product_id} in {total_time:.2f} seconds.")

    except Exception as e:
        print(f"Task: Error in embed_and_update_product_task for product ID {product_id}: {e}")

async def process_augmentations(product_id: int, image: Image.Image, num_augmentations: int, prefix: str = ""):
    """
    Helper function to generate and process augmentations for an image.
    
    Args:
        product_id: The product ID
        image: The PIL image to augment
        num_augmentations: Number of augmentations to generate
        prefix: Prefix for the augmentation type
    """
    try:
        # Generate augmented images
        augmented_images = await asyncio.to_thread(
            augmentation_service.generate_augmented_images,
            image, 
            product_id,
            num_augmentations,
            include_original=False
        )
        
        print(f"Task: Generated {len(augmented_images)} {prefix} augmented images for product ID: {product_id}")
        
        # Process each augmented image
        augmentation_types = ["rotation", "scale", "brightness", "contrast", "saturation", "perspective", "blur"]
        for i, aug_image in enumerate(augmented_images):
            try:
                # Generate embedding for this augmented image
                aug_embedding = await asyncio.to_thread(embedding_service.get_image_embedding, aug_image)
                
                # Determine augmentation type for metadata
                aug_type = augmentation_types[i % len(augmentation_types)]
                aug_index = i // len(augmentation_types)
                aug_id = f"{prefix}_{aug_type}_{aug_index}" if prefix else f"{aug_type}_{aug_index}"
                
                # Store the augmented embedding in the database
                await supabase_service.add_augmented_embedding(
                    product_id=product_id,
                    embedding=aug_embedding,
                    augmentation_type=aug_id
                )
                
                print(f"Task: Added augmented embedding ({aug_id}) for product ID: {product_id}")
                
            except Exception as e_aug:
                print(f"Task: Error processing augmented image {i} for product ID {product_id}: {e_aug}")
    except Exception as e_augs:
        print(f"Task: Error generating augmentations for product ID {product_id}: {e_augs}")

async def process_multiple_product_images(product_id: int, image_bytes_list: List[bytes], generate_augmentations: bool = True):
    """
    Process multiple images for the same product.
    Each image will be processed both in its original form and with YOLO cropping.
    
    Args:
        product_id: The product ID to update
        image_bytes_list: List of image bytes to process
        generate_augmentations: Whether to generate augmented versions for each image
    """
    print(f"Task: Processing {len(image_bytes_list)} images for product ID: {product_id}")
    
    for i, img_bytes in enumerate(image_bytes_list):
        try:
            # We'll generate fewer augmentations per image when processing multiple images
            # but still process both original and YOLO-cropped versions
            num_augmentations = 4 if generate_augmentations else 0
            await embed_and_update_product_task(
                product_id=product_id, 
                image_bytes=img_bytes,
                generate_augmentations=generate_augmentations,
                num_augmentations=num_augmentations,
                skip_yolo_crop=False  # Always process both original and YOLO-cropped
            )
            print(f"Task: Completed processing image {i+1}/{len(image_bytes_list)} for product ID: {product_id}")
        except Exception as e:
            print(f"Task: Error processing image {i+1}/{len(image_bytes_list)} for product ID {product_id}: {e}") 