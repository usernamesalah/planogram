from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Tuple, Dict, Any, Optional, Callable
import io
import os
from pathlib import Path
import uuid

# Constants for augmentation parameters
ROTATION_ANGLES = [-15, -10, -5, 0, 5, 10, 15]
SCALE_FACTORS = [0.9, 0.95, 1.0, 1.05, 1.1]
BRIGHTNESS_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
CONTRAST_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
# Add more color/hue variation to better handle colorful packaging
SATURATION_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
# Add perspective transforms to handle viewing angle variations
PERSPECTIVE_FACTORS = [0.05, 0.1, 0.15]

# Directory for saving augmented images (optional)
SAVE_AUGMENTED_IMAGES = os.getenv("SAVE_AUGMENTED_IMAGES", "true").lower() == "true"
AUGMENTED_IMAGES_PATH = Path("product_embedding_inputs/augmented")
if SAVE_AUGMENTED_IMAGES:
    AUGMENTED_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Augmentation service: Will save augmented images to {AUGMENTED_IMAGES_PATH.resolve()}")

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate an image by the specified angle in degrees."""
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)

def scale_image(image: Image.Image, factor: float) -> Image.Image:
    """Scale an image by the specified factor."""
    width, height = image.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    return image.resize((new_width, new_height), Image.BICUBIC)

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust the brightness of an image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust the contrast of an image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    """Adjust the saturation of an image by the specified factor."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def color_jitter(image: Image.Image, 
                brightness_factor: float = 1.0, 
                contrast_factor: float = 1.0, 
                saturation_factor: float = 1.0) -> Image.Image:
    """Apply color jittering to an image."""
    # Apply brightness adjustment
    image = adjust_brightness(image, brightness_factor)
    # Apply contrast adjustment
    image = adjust_contrast(image, contrast_factor)
    # Apply saturation adjustment
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)
    return image

def normalize_image_size(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize an image to a target size while preserving aspect ratio."""
    return image.resize(target_size, Image.BICUBIC)

def apply_perspective_transform(image: Image.Image, distortion: float) -> Image.Image:
    """Apply a perspective transform to simulate different viewing angles."""
    width, height = image.size
    
    # Define the source and destination points for the perspective transform
    # The distortion parameter controls how much the corners are displaced
    x_shift = int(width * distortion)
    y_shift = int(height * distortion)
    
    # Four corners of the image
    coeffs = [
        # Top left
        0, 0, x_shift, y_shift,
        # Top right
        width, 0, width - x_shift, y_shift,
        # Bottom right
        width, height, width - x_shift, height - y_shift,
        # Bottom left
        0, height, x_shift, height - y_shift
    ]
    
    return image.transform(
        (width, height),
        Image.QUAD,
        coeffs,
        Image.BICUBIC
    )

def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply a slight blur to simulate different focus conditions."""
    return image.filter(ImageFilter.GaussianBlur(radius))

def generate_augmented_images(image: Image.Image, 
                             product_id: Optional[int] = None,
                             num_augmentations: int = 5,
                             include_original: bool = True) -> List[Image.Image]:
    """
    Generate a list of augmented images from the original image.
    
    Args:
        image: The original PIL Image
        product_id: Product ID for saving images (if enabled)
        num_augmentations: Number of augmented versions to generate
        include_original: Whether to include the original image in the result
    
    Returns:
        List of augmented PIL Images
    """
    augmented_images = []
    
    # Include the original image if requested
    if include_original:
        augmented_images.append(image)
        if SAVE_AUGMENTED_IMAGES and product_id is not None:
            image_path = AUGMENTED_IMAGES_PATH / f"product_{product_id}_original.jpg"
            image.save(image_path, "JPEG")
    
    # List of all available augmentation functions with their parameters
    augmentation_functions = [
        # Original augmentations
        (rotate_image, ROTATION_ANGLES),
        (scale_image, SCALE_FACTORS),
        (adjust_brightness, BRIGHTNESS_FACTORS),
        (adjust_contrast, CONTRAST_FACTORS),
        # New augmentations
        # (adjust_saturation, SATURATION_FACTORS),
        # (apply_perspective_transform, PERSPECTIVE_FACTORS),
        # (apply_blur, [0.5, 1.0, 1.5])
    ]
    
    # Generate requested number of augmented images
    for i in range(num_augmentations):
        # Create a copy of the original image
        augmented = image.copy()
        
        # Apply multiple augmentations in sequence
        # For each augmentation function, randomly select parameters
        for func, params in augmentation_functions:
            # 60% chance to apply each augmentation (more diverse augmentations)
            if np.random.random() < 0.6:
                param = params[np.random.randint(0, len(params))]
                try:
                    augmented = func(augmented, param)
                except Exception as e:
                    print(f"Error applying {func.__name__} with param {param}: {e}")
        
        augmented_images.append(augmented)
        
        # Save augmented image if enabled
        if SAVE_AUGMENTED_IMAGES and product_id is not None:
            aug_id = uuid.uuid4().hex[:8]
            image_path = AUGMENTED_IMAGES_PATH / f"product_{product_id}_aug_{aug_id}.jpg"
            augmented.save(image_path, "JPEG")
    
    return augmented_images

def get_torchvision_transforms() -> Callable:
    """
    Get a composition of standard torchvision transforms for more advanced augmentation.
    This can be useful for pre-processing before passing to neural networks.
    """
    return T.Compose([
        T.RandomRotation(15),
        T.RandomResizedCrop(224, scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Example usage
if __name__ == "__main__":
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # Generate augmented images
    augmented = generate_augmented_images(test_image, count=3)
    
    print(f"Generated {len(augmented)} augmented images (including original)") 