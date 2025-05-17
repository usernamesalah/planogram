from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import io
import base64
import os # For environment variable and path
import time # For unique filenames
from pathlib import Path # For creating directory

# Load the YOLO model (e.g., YOLOv8n for a small and fast model)
# This model is loaded once when the module is imported.
MODEL_NAME = "best-yolo11m.pt" # You can choose other models like yolov8s.pt, yolov8m.pt, etc.
try:
    yolo_model = YOLO(MODEL_NAME)
    print(f"YOLO model ({MODEL_NAME}) loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model ({MODEL_NAME}): {e}")
    yolo_model = None

# --- Environment Variable for Saving Images ---
SAVE_IMAGES_LOCALLY = os.getenv("SAVE_ANNOTATED_IMAGES", "false").lower() == "true"
LOCAL_SAVE_PATH = Path("annotated_outputs/detection")
if SAVE_IMAGES_LOCALLY:
    LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Detection service: Will save annotated images to {LOCAL_SAVE_PATH.resolve()}")

def detect_objects(image_bytes: bytes, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Detects objects in an image using the YOLO model.

    Args:
        image_bytes: Bytes of the image file (e.g., from FastAPI UploadFile.read()).
        confidence_threshold: Minimum confidence score for a detection to be considered.

    Returns:
        A list of dictionaries, where each dictionary represents a detected object
        and contains 'box' (x1, y1, x2, y2), 'label' (class name), and 'confidence'.
    """
    if not yolo_model:
        raise RuntimeError("YOLO model is not loaded. Cannot perform detection.")

    try:
        # Convert image bytes to PIL Image, then to NumPy array for OpenCV compatibility if needed
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Perform detection
        # The results object contains detailed information about detections
        results = yolo_model(pil_image, conf=confidence_threshold)
        
        detected_objects = []
        # `results[0].boxes.data` provides tensor with [x1, y1, x2, y2, conf, cls_id]
        # `results[0].names` is a dict mapping class_id to class_name
        if results and results[0].boxes is not None:
            for box_data in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls_id = box_data.tolist()
                label = yolo_model.names[int(cls_id)] if yolo_model.names else f"class_{int(cls_id)}"
                
                detected_objects.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)], # (x_min, y_min, x_max, y_max)
                    "label": label,
                    "confidence": float(conf)
                })
        return detected_objects
    except Exception as e:
        print(f"Error during object detection: {e}")
        raise

def crop_object(image: Image.Image, box: List[int]) -> Image.Image:
    """Crops an object from an image using its bounding box."""
    return image.crop(box) # box is (left, upper, right, lower)

def smart_crop(image: Image.Image) -> Tuple[Image.Image, List[int]]:
    """
    Intelligently crops a product image using CV2-based object detection.
    Works well for product packaging against contrasting backgrounds.
    
    Args:
        image: PIL Image of the product
        
    Returns:
        Tuple of (cropped_image, bounding_box)
    """
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try adaptive thresholding first (works well for varied lighting)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no good contours, try Otsu's thresholding
        if not contours or max(contours, key=cv2.contourArea, default=0) == 0:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fall back to Canny edge detection if still no good contours
        if not contours or max(contours, key=cv2.contourArea, default=0) == 0:
            edges = cv2.Canny(blurred, 30, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding (10%)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            # Ensure coordinates stay within image boundaries
            img_width, img_height = image.size
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(img_width, x + w + padding_x)
            y2 = min(img_height, y + h + padding_y)
            
            # Crop the image
            cropped_img = image.crop((x1, y1, x2, y2))
            return cropped_img, [x1, y1, x2, y2]
        
        # If no contours found, return the original image and its full dimensions
        return image, [0, 0, image.width, image.height]
        
    except Exception as e:
        print(f"Error in smart_crop: {e}")
        # Return original image in case of error
        return image, [0, 0, image.width, image.height]

def color_based_crop(image: Image.Image, target_color_range=None) -> Tuple[Image.Image, List[int]]:
    """
    Crops a product image based on color segmentation.
    Works well for products with distinctive colors like Kuaci Rebo's yellow/red packaging.
    
    Args:
        image: PIL Image of the product
        target_color_range: Optional tuple ((h_min,s_min,v_min), (h_max,s_max,v_max)) 
                           in HSV color space. If None, uses default yellow/red range.
    
    Returns:
        Tuple of (cropped_image, bounding_box)
    """
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Default to yellow-red range (common in snack packaging) if not specified
        if target_color_range is None:
            # Yellow-orange-red masks (Kuaci Rebo has yellow and red packaging)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([40, 255, 255])
            
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([179, 255, 255])
            
            # Create yellow and red masks
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_or(red_mask1, red_mask2))
        else:
            # Use provided color range
            lower_bound, upper_bound = target_color_range
            combined_mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding (10%)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            # Ensure coordinates stay within image boundaries
            img_width, img_height = image.size
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(img_width, x + w + padding_x)
            y2 = min(img_height, y + h + padding_y)
            
            # Crop the image
            cropped_img = image.crop((x1, y1, x2, y2))
            return cropped_img, [x1, y1, x2, y2]
        
        # If no contours found, return the original image
        return image, [0, 0, image.width, image.height]
        
    except Exception as e:
        print(f"Error in color_based_crop: {e}")
        # Return original image in case of error
        return image, [0, 0, image.width, image.height]

def draw_boxes_on_image(image_bytes: bytes, annotations_to_draw: List[Dict[str, Any]]) -> bytes:
    """Draws bounding boxes and labels on an image based on provided annotations.
    Each annotation in annotations_to_draw should be a dict with:
    'box': [x1, y1, x2, y2]
    'label': str (product name or other label)
    'confidence': float (similarity score or other confidence, can be None)
    'color': str (e.g., 'green', 'red')
    'product_id': int (optional, for consistent coloring)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Color mapping to ensure same products get same colors
        # Using vibrant, distinguishable colors
        product_colors = [
            "red", "blue", "green", "orange", "purple", 
            "deeppink", "teal", "darkviolet", "crimson", "forestgreen",
            "darkorange", "navy", "firebrick", "darkslategray", "goldenrod"
        ]
        
        # Keep track of product_id to color mapping
        product_color_map = {}
        color_index = 0
        
        for ann in annotations_to_draw:
            box = ann['box']
            label = ann['label']
            confidence = ann.get('confidence') # Use .get() for safety if confidence might be missing
            product_id = ann.get('product_id')
            
            # Determine color based on product_id if available
            if product_id is not None:
                if product_id not in product_color_map:
                    # Assign a new color for this product
                    product_color_map[product_id] = product_colors[color_index % len(product_colors)]
                    color_index += 1
                color = product_color_map[product_id]
            else:
                # Use provided color or default to red
                color = ann.get('color', 'red')  

            # Format text: include confidence if it exists and is a float
            text = label
            if confidence is not None and isinstance(confidence, float):
                text = f"{label} ({confidence:.2f})"
            
            # Draw rectangle with thicker border (width=5)
            draw.rectangle(box, outline=color, width=5)
            
            # Create more visible text with background
            text_position = (box[0] + 2, box[1] + 2)
            text_color = "white"  # White text is more visible against dark backgrounds
            
            # Measure text size to create background
            text_width, text_height = 0, 0
            try:
                # Try to use a larger, more visible font
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                    text_width, text_height = draw.textsize(text, font=font)
                except (AttributeError, IOError):
                    # Fallback if textsize or font not available
                    font = ImageFont.load_default()
                    text_height = 15
                    text_width = len(text) * 8  # Approximate width
                
                # Draw darker background behind text for better visibility
                draw.rectangle(
                    [text_position[0], text_position[1], 
                     text_position[0] + text_width, text_position[1] + text_height],
                    fill=color
                )
                
                # Draw text on the colored background
                draw.text(text_position, text, fill=text_color, font=font)
            except Exception as text_error:
                # Ultimate fallback - basic text without background
                draw.text(text_position, text, fill=color)
                print(f"Error drawing enhanced text: {text_error}")
                 
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        output_bytes = buffered.getvalue()

        if SAVE_IMAGES_LOCALLY:
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Update filename to reflect that these are identified annotations
                filename = LOCAL_SAVE_PATH / f"identified_annotated_{timestamp}_{len(annotations_to_draw)}obj.jpg"
                with open(filename, "wb") as f:
                    f.write(output_bytes)
                print(f"Saved IDENTIFIED annotated image to: {filename}")
            except Exception as e_save:
                print(f"Error saving identified annotated image: {e_save}")
        
        return output_bytes
    except Exception as e:
        print(f"Error drawing boxes on image: {e}")
        raise

# --- Helper to convert PIL Image to Base64 String (useful for API responses) ---
def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Example usage (optional, for testing)
if __name__ == "__main__":
    if yolo_model:
        # Create a dummy image for testing
        try:
            dummy_pil_image = Image.new('RGB', (640, 480), color = 'blue')
            # Convert PIL image to bytes as the detect_objects function expects
            buffered = io.BytesIO()
            dummy_pil_image.save(buffered, format="JPEG")
            image_bytes_for_detection = buffered.getvalue()

            detections = detect_objects(image_bytes_for_detection)
            print(f"Detected objects on dummy image: {detections}")

            if detections:
                # Crop first detected object
                first_box = detections[0]["box"]
                cropped_img = crop_object(dummy_pil_image, first_box)
                # cropped_img.show() # Uncomment to display if in a suitable environment
                print(f"Cropped first object. Size: {cropped_img.size}")

                # Draw boxes
                annotated_image_bytes = draw_boxes_on_image(image_bytes_for_detection, detections)
                annotated_pil_image = Image.open(io.BytesIO(annotated_image_bytes))
                # annotated_pil_image.show() # Uncomment to display
                print("Drew boxes on dummy image.")
                b64_annotated = pil_image_to_base64(annotated_pil_image)
                print(f"Base64 annotated image (first 100 chars): {b64_annotated[:100]}")

        except RuntimeError as e:
            print(f"Runtime error during YOLO example: {e}")
        except Exception as e:
            print(f"General error during YOLO example: {e}")
    else:
        print("YOLO model not loaded, skipping examples.") 