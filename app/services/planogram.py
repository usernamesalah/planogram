from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import os # For environment variable and path
import time # For unique filenames
from pathlib import Path # For creating directory

from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import supabase as supabase_service
from app.models.product import Product # For type hinting identified products

# --- Environment Variable for Saving Images ---
SAVE_IMAGES_LOCALLY = os.getenv("SAVE_ANNOTATED_IMAGES", "false").lower() == "true"
LOCAL_SAVE_PATH_PLANOGRAM = Path("annotated_outputs/planogram")
if SAVE_IMAGES_LOCALLY:
    LOCAL_SAVE_PATH_PLANOGRAM.mkdir(parents=True, exist_ok=True)
    print(f"Planogram service: Will save annotated images to {LOCAL_SAVE_PATH_PLANOGRAM.resolve()}")

# --- Helper for IoU --- #
def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Add epsilon for stability
    return iou

# --- Main Planogram Comparison Logic --- #
async def compare_planogram_layouts(
    actual_image_bytes: bytes,
    expected_layout: List[Dict[str, Any]], # Each dict: {'product_id': int, 'name': str, 'expected_box': [x1,y1,x2,y2]}
    iou_threshold: float = 0.3, # IoU threshold to consider a product correctly placed
    identification_threshold: float = 0.7 # Confidence for matching detected object to DB product via embedding
) -> Dict[str, Any]:
    """
    Compares the actual layout of products on a shelf (from an image)
    with an expected planogram layout.
    """
    actual_pil_image = Image.open(io.BytesIO(actual_image_bytes)).convert("RGB")

    # 1. Detect objects on the actual shelf
    detected_objects_on_shelf = await detection_service.detect_objects(actual_image_bytes)
    print(f"Planogram Service: Detected {len(detected_objects_on_shelf)} objects on the shelf.")

    identified_actual_products: List[Dict[str, Any]] = [] # Store identified products and their actual boxes

    # 2. Identify detected objects by comparing their embeddings with DB products
    for det_obj in detected_objects_on_shelf:
        actual_box = det_obj['box']
        try:
            cropped_pil_image = detection_service.crop_object(actual_pil_image, actual_box)
            # Ensure get_image_embedding is thread-safe or run in executor if it becomes a bottleneck
            img_embedding = embedding_service.get_image_embedding(cropped_pil_image)
            
            # Find best matching product in DB
            # find_products_by_embedding returns a list; we might take the top one if similarity is high
            matched_db_products = await supabase_service.find_products_by_embedding(
                img_embedding, 
                match_threshold=identification_threshold, # Use this as the similarity score from DB function
                match_count=1
            )
            
            if matched_db_products:
                best_match_product: Product = matched_db_products[0] # Product model from Pydantic
                # The RPC function 'match_products' should return 'similarity' score
                similarity_score = getattr(best_match_product, 'similarity', identification_threshold) 

                if similarity_score >= identification_threshold:
                    identified_actual_products.append({
                        "product_id": best_match_product.id,
                        "name": best_match_product.name,
                        "variant": best_match_product.variant,
                        "actual_box": actual_box,
                        "confidence": det_obj['confidence'], # Detection confidence
                        "identification_similarity": similarity_score
                    })
                    print(f"Identified: {best_match_product.name} (ID: {best_match_product.id}) at {actual_box}")
                else:
                    print(f"Low similarity ({similarity_score}) for object at {actual_box}. Not confidently identified.")
            else:
                print(f"No DB match for object at {actual_box}")
        except Exception as e:
            print(f"Error identifying object at {actual_box}: {e}")
    
    print(f"Planogram Service: Identified {len(identified_actual_products)} products on the shelf.")

    # 3. Compare with expected layout
    correctly_placed_count = 0
    misplaced_products: List[Dict[str, Any]] = []
    missing_products: List[Dict[str, Any]] = []
    
    expected_product_ids_found = set()

    for expected_item in expected_layout:
        expected_product_id = expected_item.get("product_id")
        expected_name = expected_item.get("name", f"ID:{expected_product_id}")
        expected_box = expected_item["expected_box"]
        found_and_correctly_placed = False

        for actual_item in identified_actual_products:
            if actual_item["product_id"] == expected_product_id:
                expected_product_ids_found.add(expected_product_id) # Mark as found
                iou = calculate_iou(actual_item["actual_box"], expected_box)
                if iou >= iou_threshold:
                    correctly_placed_count += 1
                    found_and_correctly_placed = True
                    actual_item['status'] = 'correct' # For drawing
                    actual_item['expected_box'] = expected_box # For drawing
                    break # This expected item is found and correctly placed
                else:
                    # Found the correct product ID, but it's in the wrong place
                    misplaced_products.append({
                        "product_id": expected_product_id,
                        "name": expected_name,
                        "expected_box": expected_box,
                        "actual_box": actual_item["actual_box"],
                        "iou": iou,
                        "reason": "Found, but wrong location"
                    })
                    actual_item['status'] = 'misplaced' # For drawing
                    actual_item['expected_box'] = expected_box # For drawing
                    found_and_correctly_placed = True # Considered handled, even if misplaced
                    break 
        
        if not found_and_correctly_placed:
            # This expected item was not found among the identified products at all
            # or if it was found, it was already marked as misplaced for another expected slot (complex case not handled here)
            missing_products.append({
                "product_id": expected_product_id,
                "name": expected_name,
                "expected_box": expected_box,
                "reason": "Not found or not identified on shelf"
            })

    # Check for any identified products on shelf that were not part of the expected layout (extra products)
    # Or products that were identified but didn't match any expected_item (could also be "misplaced" if not strictly slot-based)
    # This part can be refined. For now, focusing on missing/misplaced from expected list.

    if not expected_layout: # Avoid division by zero
        compliance_score = 1.0 if not identified_actual_products else 0.0 # Perfect if nothing expected and nothing found
    else:
        compliance_score = correctly_placed_count / len(expected_layout)

    # 4. Annotate image (basic)
    annotated_image_bytes = _draw_planogram_annotations(actual_pil_image, identified_actual_products, expected_layout, missing_products, misplaced_products, iou_threshold)
    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode("utf-8")

    return {
        "compliance_score": compliance_score,
        "correctly_placed_count": correctly_placed_count,
        "total_expected_products": len(expected_layout),
        "identified_on_shelf_count": len(identified_actual_products),
        "misplaced_products": misplaced_products,
        "missing_products": missing_products,
        "annotated_image_b64": annotated_image_b64
    }

def _draw_planogram_annotations(
    image: Image.Image, 
    identified_products: List[Dict[str, Any]], 
    expected_layout: List[Dict[str, Any]],
    missing_products: List[Dict[str,Any]],
    misplaced_details: List[Dict[str, Any]], # For getting expected box of misplaced items
    iou_threshold: float
) -> bytes:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw identified products
    for item in identified_products:
        box = item["actual_box"]
        name = item.get("name", f"ID:{item['product_id']}")
        status = item.get('status') # 'correct', 'misplaced'
        
        color = "yellow" # Default for identified but not matched to an expectation explicitly
        if status == 'correct':
            color = "green"
        elif status == 'misplaced':
            color = "orange"
        
        draw.rectangle(box, outline=color, width=5)
        draw.text((box[0], box[1] - 15 if box[1] - 15 > 0 else box[1] + 2), f"{name[:20]} (ID'd)", fill=color, font=font)

        # If misplaced, also draw its expected box
        if status == 'misplaced' and 'expected_box' in item:
            exp_b = item['expected_box']
            draw.rectangle(exp_b, outline="blue", width=3, dash=[5,5])
            draw.text((exp_b[0], exp_b[1] - 15 if exp_b[1] - 15 > 0 else exp_b[1] + 2), f"Exp: {name[:15]}", fill="blue", font=font)


    # Draw missing products (their expected locations)
    for item in missing_products:
        box = item["expected_box"]
        name = item.get("name", f"ID:{item['product_id']}")
        draw.rectangle(box, outline="red", width=5, dash=[5,5])
        draw.text((box[0], box[1] - 15 if box[1] - 15 > 0 else box[1] + 2), f"MISSING: {name[:15]}", fill="red", font=font)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    output_bytes = buffered.getvalue()

    if SAVE_IMAGES_LOCALLY:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = LOCAL_SAVE_PATH_PLANOGRAM / f"planogram_analyzed_{timestamp}.jpg"
            with open(filename, "wb") as f:
                f.write(output_bytes)
            print(f"Saved planogram annotated image to: {filename}")
        except Exception as e_save:
            print(f"Error saving planogram annotated image: {e_save}")
            
    return output_bytes 