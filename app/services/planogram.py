from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import os # For environment variable and path
import time # For unique filenames
from pathlib import Path # For creating directory
import numpy as np
from scipy.spatial.distance import cosine
import pytesseract  # For OCR
import cv2  # For image preprocessing for OCR

from app.services import detection as detection_service
from app.services import embedding as embedding_service
from app.services import supabase as supabase_service
from app.models.product import Product, OCRResult # For type hinting identified products

# --- Environment Variable for Saving Images ---
SAVE_IMAGES_LOCALLY = os.getenv("SAVE_ANNOTATED_IMAGES", "false").lower() == "true"
LOCAL_SAVE_PATH_PLANOGRAM = Path("annotated_outputs/planogram")
if SAVE_IMAGES_LOCALLY:
    LOCAL_SAVE_PATH_PLANOGRAM.mkdir(parents=True, exist_ok=True)
    print(f"Planogram service: Will save annotated images to {LOCAL_SAVE_PATH_PLANOGRAM.resolve()}")

# Try to detect Tesseract executable path for OCR
try:
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except:
    # Try to set the Tesseract path if not in PATH
    try:
        # Common paths for Windows
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    except:
        print("Warning: Tesseract OCR is not available. OCR features will be disabled.")
        OCR_AVAILABLE = False

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

# --- OCR Text Extraction --- #
def extract_text_from_image(image: Image.Image, box: Optional[List[int]] = None) -> List[OCRResult]:
    """
    Extract text from an image or a region of an image using OCR.
    
    Args:
        image: PIL Image to extract text from
        box: Optional bounding box [x1, y1, x2, y2] to crop before extraction
        
    Returns:
        List of OCRResult objects with text, confidence, and bounding box
    """
    if not OCR_AVAILABLE:
        return []
        
    try:
        # Convert PIL Image to OpenCV format for preprocessing
        if box:
            # Crop to region of interest if box provided
            cropped = image.crop(box)
            img_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            x_offset, y_offset = box[0], box[1]
        else:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            x_offset, y_offset = 0, 0
            
        # Preprocess image for better OCR results
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get a binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Use Tesseract OCR with configuration for detection with confidence
        ocr_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -" -c page_separator=""'
        ocr_data = pytesseract.image_to_data(binary, config=ocr_config, output_type=pytesseract.Output.DICT)
        
        results = []
        # Process OCR results
        for i in range(len(ocr_data['text'])):
            # Filter out empty text and low confidence results
            if int(ocr_data['conf'][i]) > 60 and ocr_data['text'][i].strip():
                # Get bounding box coordinates
                x = ocr_data['left'][i] + x_offset
                y = ocr_data['top'][i] + y_offset
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                results.append(OCRResult(
                    text=ocr_data['text'][i],
                    confidence=float(ocr_data['conf'][i]) / 100.0,
                    box=[x, y, x + w, y + h]
                ))
                
        return results
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return []

# --- Spatial Context Calculation --- #
def calculate_spatial_proximity(box1: List[int], box2: List[int]) -> float:
    """
    Calculate how close two boxes are to each other spatially.
    Returns a value between 0 and 1, where 1 means boxes are identical.
    """
    # Calculate centers of boxes
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    
    # Calculate Euclidean distance between centers
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    # Normalize distance - closer boxes have higher score
    # Use average box diagonal length for normalization
    box1_diag = ((box1[2] - box1[0]) ** 2 + (box1[3] - box1[1]) ** 2) ** 0.5
    box2_diag = ((box2[2] - box2[0]) ** 2 + (box2[3] - box2[1]) ** 2) ** 0.5
    avg_diag = (box1_diag + box2_diag) / 2
    
    # Convert distance to similarity score (1 when distance is 0, approaching 0 as distance increases)
    proximity = max(0, 1 - distance / (avg_diag * 2))
    return proximity

# --- Text Similarity Calculation --- #
def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    Returns a value between 0 and 1, where 1 means identical texts.
    """
    if not text1 or not text2:
        return 0.0
        
    # Simple implementation using character-level Jaccard similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

# --- Hybrid Re-ranking Function --- #
def hybrid_rerank(
    candidates: List[Product], 
    query_box: List[int],
    query_embedding: List[float],
    ocr_text: Optional[str] = None,
    expected_box: Optional[List[int]] = None,
    metadata_weight: float = 0.3,
    ocr_weight: float = 0.2,
    spatial_weight: float = 0.1,
    embedding_weight: float = 0.4
) -> List[Tuple[Product, float]]:
    """
    Re-rank candidate products using a hybrid approach combining:
    - Embedding similarity (from vector search)
    - Metadata relevance
    - OCR text matching
    - Spatial context
    
    Args:
        candidates: List of product candidates
        query_box: Bounding box of the query object [x1, y1, x2, y2]
        query_embedding: CLIP embedding of the query object
        ocr_text: OCR text extracted from the query object (if available)
        expected_box: Expected location from planogram (if available)
        weights: Weights for each component in the final score
        
    Returns:
        List of (product, score) tuples, sorted by score in descending order
    """
    if not candidates:
        return []
        
    # Calculate scores for each candidate
    scored_candidates = []
    
    for product in candidates:
        # 1. Embedding similarity score (already provided by vector search)
        embedding_score = getattr(product, 'similarity', 0.7)
        
        # 2. Metadata relevance score - simplified implementation
        metadata_score = 0.5  # Default middle value
        
        # 3. OCR text matching score
        ocr_score = 0.0
        if ocr_text and product.name:
            ocr_score = calculate_text_similarity(ocr_text, product.name)
            # Also check variant if available
            if product.variant:
                variant_score = calculate_text_similarity(ocr_text, product.variant)
                ocr_score = max(ocr_score, variant_score)
        
        # 4. Spatial context score
        spatial_score = 0.0
        if expected_box and expected_box != query_box:
            spatial_score = calculate_spatial_proximity(query_box, expected_box)
        
        # Calculate final score as weighted combination
        final_score = (
            embedding_weight * embedding_score +
            metadata_weight * metadata_score +
            ocr_weight * ocr_score +
            spatial_weight * spatial_score
        )
        
        scored_candidates.append((product, final_score))
    
    # Sort by score in descending order
    return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

# --- Main Planogram Comparison Logic --- #
async def compare_planogram_layouts(
    actual_image_bytes: bytes,
    expected_layout: List[Dict[str, Any]], # Each dict: {'product_id': int, 'name': str, 'expected_box': [x1,y1,x2,y2]}
    iou_threshold: float = 0.3, # IoU threshold to consider a product correctly placed
    identification_threshold: float = 0.7, # Confidence for matching detected object to DB product via embedding
    metadata_filters: Optional[Dict[str, Any]] = None, # Optional metadata filters
    use_augmentations: bool = True # Whether to use augmented embeddings
) -> Dict[str, Any]:
    """
    Compares the actual layout of products on a shelf (from an image)
    with an expected planogram layout.
    
    Args:
        actual_image_bytes: Image of the actual shelf
        expected_layout: List of expected products with their locations
        iou_threshold: Threshold for IoU to consider correct placement
        identification_threshold: Threshold for product identification
        metadata_filters: Optional filters to narrow down product search
        use_augmentations: Whether to use augmented embeddings
    """
    actual_pil_image = Image.open(io.BytesIO(actual_image_bytes)).convert("RGB")

    # 1. Detect objects on the actual shelf
    detected_objects_on_shelf = detection_service.detect_objects(actual_image_bytes)
    print(f"Planogram Service: Detected {len(detected_objects_on_shelf)} objects on the shelf.")

    identified_actual_products: List[Dict[str, Any]] = [] # Store identified products and their actual boxes

    # 2. Identify detected objects with the improved approach
    for det_obj in detected_objects_on_shelf:
        actual_box = det_obj['box']
        try:
            # Crop the detected object
            cropped_pil_image = detection_service.crop_object(actual_pil_image, actual_box)
            
            # OCR functionality disabled
            ocr_text = None
            
            # Generate embedding
            img_embedding = embedding_service.get_image_embedding(cropped_pil_image)
            
            # Find expected box that overlaps most with this detected box (for spatial context)
            expected_box = None
            max_iou = 0
            if expected_layout:
                for exp_item in expected_layout:
                    if "expected_box" in exp_item:
                        iou = calculate_iou(actual_box, exp_item["expected_box"])
                        if iou > max_iou:
                            max_iou = iou
                            expected_box = exp_item["expected_box"]
            
            # Find matching products with potential metadata filtering
            matched_db_products = await supabase_service.find_products_by_embedding(
                embedding=img_embedding, 
                match_threshold=identification_threshold * 0.9,  # Slightly lower threshold since we'll re-rank
                match_count=10,  # Get more candidates for re-ranking
                metadata_filters=metadata_filters,
                use_augmentations=use_augmentations
            )
            
            if matched_db_products:
                # Apply hybrid re-ranking
                reranked_products = hybrid_rerank(
                    candidates=matched_db_products,
                    query_box=actual_box,
                    query_embedding=img_embedding,
                    ocr_text=ocr_text,
                    expected_box=expected_box
                )
                
                if reranked_products:
                    best_match_product, best_match_score = reranked_products[0]
                    
                    if best_match_score >= identification_threshold:
                        identified_actual_products.append({
                            "product_id": best_match_product.id,
                            "name": best_match_product.name,
                            "variant": best_match_product.variant,
                            "category": best_match_product.category,
                            "brand": best_match_product.brand,
                            "actual_box": actual_box,
                            "confidence": det_obj['confidence'],
                            "identification_similarity": best_match_score,
                            "ocr_text": ocr_text
                        })
                        print(f"Identified: {best_match_product.name} (ID: {best_match_product.id}) at {actual_box} with score {best_match_score:.2f}")
                    else:
                        print(f"Low score ({best_match_score:.2f}) for object at {actual_box}. Not confidently identified.")
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
            # This expected item was not found among the identified products
            missing_products.append({
                "product_id": expected_product_id,
                "name": expected_name,
                "expected_box": expected_box,
                "reason": "Not found or not identified on shelf"
            })

    # Check for extra products (identified but not in expected layout)
    extra_products = []
    for actual_item in identified_actual_products:
        if 'status' not in actual_item:
            actual_item['status'] = 'extra'
            extra_products.append({
                "product_id": actual_item["product_id"],
                "name": actual_item["name"],
                "actual_box": actual_item["actual_box"],
                "reason": "Found on shelf but not in expected layout"
            })

    if not expected_layout: # Avoid division by zero
        compliance_score = 1.0 if not identified_actual_products else 0.0
    else:
        compliance_score = correctly_placed_count / len(expected_layout)

    # 4. Annotate image with enhanced information
    annotated_image_bytes = _draw_planogram_annotations(
        actual_pil_image, 
        identified_actual_products, 
        expected_layout, 
        missing_products, 
        misplaced_products,
        iou_threshold
    )
    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode("utf-8")

    return {
        "compliance_score": compliance_score,
        "correctly_placed_count": correctly_placed_count,
        "total_expected_products": len(expected_layout),
        "identified_on_shelf_count": len(identified_actual_products),
        "misplaced_products": misplaced_products,
        "missing_products": missing_products,
        "extra_products": extra_products,
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
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Product ID to color mapping for consistency
    product_colors = {
        "correct": "green",
        "misplaced": "orange",
        "extra": "purple",
        "missing": "red"
    }
    
    # Draw identified products
    for item in identified_products:
        box = item["actual_box"]
        name = item.get("name", f"ID:{item['product_id']}")
        status = item.get('status') # 'correct', 'misplaced', 'extra'
        
        color = product_colors.get(status, "yellow")
        
        # Draw bounding box with thicker border
        draw.rectangle(box, outline=color, width=5)
        
        # Draw eye-catching text with background
        display_text = name[:20]
        if 'ocr_text' in item and item['ocr_text']:
            display_text += f" OCR:{item['ocr_text'][:15]}"
        
        # Position and draw text with background
        text_position = (box[0] + 2, box[1] - 20 if box[1] - 20 > 0 else box[1] + 2)
        
        # Measure text for background
        try:
            text_width, text_height = draw.textsize(display_text, font=font)
        except AttributeError:
            # Fallback if textsize not available
            text_height = 20
            text_width = len(display_text) * 8
        
        # Draw background behind text
        text_bg = [
            text_position[0], text_position[1],
            text_position[0] + text_width, text_position[1] + text_height
        ]
        draw.rectangle(text_bg, fill=color)
        
        # Draw white text on colored background
        draw.text(text_position, display_text, fill="white", font=font)

        # If misplaced, also draw its expected box
        if status == 'misplaced' and 'expected_box' in item:
            exp_b = item['expected_box']
            draw.rectangle(exp_b, outline="blue", width=3, dash=[5,5])
            
            # Draw expected position text with background
            expected_text = f"Exp: {name[:15]}"
            exp_text_pos = (exp_b[0] + 2, exp_b[1] - 20 if exp_b[1] - 20 > 0 else exp_b[1] + 2)
            
            # Measure text
            try:
                exp_text_width, exp_text_height = draw.textsize(expected_text, font=font)
            except AttributeError:
                exp_text_height = 20
                exp_text_width = len(expected_text) * 8
                
            # Draw background
            exp_text_bg = [
                exp_text_pos[0], exp_text_pos[1],
                exp_text_pos[0] + exp_text_width, exp_text_pos[1] + exp_text_height
            ]
            draw.rectangle(exp_text_bg, fill="blue")
            
            # Draw text
            draw.text(exp_text_pos, expected_text, fill="white", font=font)

    # Draw missing products (their expected locations)
    for item in missing_products:
        box = item["expected_box"]
        name = item.get("name", f"ID:{item['product_id']}")
        
        # Draw dashed box
        draw.rectangle(box, outline=product_colors["missing"], width=5, dash=[5,5])
        
        # Draw missing text with background
        missing_text = f"MISSING: {name[:15]}"
        missing_text_pos = (box[0] + 2, box[1] - 20 if box[1] - 20 > 0 else box[1] + 2)
        
        # Measure text
        try:
            missing_text_width, missing_text_height = draw.textsize(missing_text, font=font)
        except AttributeError:
            missing_text_height = 20
            missing_text_width = len(missing_text) * 8
            
        # Draw background
        missing_text_bg = [
            missing_text_pos[0], missing_text_pos[1],
            missing_text_pos[0] + missing_text_width, missing_text_pos[1] + missing_text_height
        ]
        draw.rectangle(missing_text_bg, fill=product_colors["missing"])
        
        # Draw text
        draw.text(missing_text_pos, missing_text, fill="white", font=font)

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