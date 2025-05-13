from ultralytics import YOLO
from PIL import Image, ImageDraw
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

def draw_boxes_on_image(image_bytes: bytes, annotations_to_draw: List[Dict[str, Any]]) -> bytes:
    """Draws bounding boxes and labels on an image based on provided annotations.
    Each annotation in annotations_to_draw should be a dict with:
    'box': [x1, y1, x2, y2]
    'label': str (product name or other label)
    'confidence': float (similarity score or other confidence, can be None)
    'color': str (e.g., 'green', 'red')
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for ann in annotations_to_draw: # Changed variable name from det to ann
            box = ann['box']
            label = ann['label']
            confidence = ann.get('confidence') # Use .get() for safety if confidence might be missing
            color = ann.get('color', 'red')  # Default to red if color not provided

            # Format text: include confidence if it exists and is a float
            text = label
            if confidence is not None and isinstance(confidence, float):
                text = f"{label} ({confidence:.2f})"
            
            draw.rectangle(box, outline=color, width=2)
            text_position = (box[0] + 2, box[1] + 2)
            try:
                # For simplicity, using default font. Consider ImageFont.truetype for better control.
                draw.text(text_position, text, fill=color) # Use the annotation color for text too
            except IOError:
                 # Fallback if any text drawing issue occurs (e.g. font related if custom font was used)
                 draw.text(text_position, text, fill="red") # Fallback to red text
                 
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