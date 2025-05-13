import torch
import clip
from PIL import Image
from typing import List

# Determine the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessor
# We load them once when the module is imported to avoid reloading on every call
MODEL_NAME = "ViT-B/32" # You can choose other CLIP models like "ViT-L/14"

try:
    model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
    model.eval() # Set model to evaluation mode
    print(f"CLIP model ({MODEL_NAME}) loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    # Fallback or raise an error if the model is critical
    model = None
    preprocess = None

def get_image_embedding(image: Image.Image) -> List[float]:
    """Generates CLIP embedding for a given PIL Image."""
    if not model or not preprocess:
        raise RuntimeError("CLIP model is not loaded. Cannot generate image embeddings.")
    
    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        raise

def get_text_embedding(text: str) -> List[float]:
    """Generates CLIP embedding for a given text string."""
    if not model:
        raise RuntimeError("CLIP model is not loaded. Cannot generate text embeddings.")

    try:
        text_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        # Normalize features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        raise

def get_embedding_dimension() -> int:
    """Returns the dimension of the loaded CLIP model's embeddings."""
    if not model:
        raise RuntimeError("CLIP model is not loaded.")
    # The embedding dimension can usually be found from the model's visual projection or text projection layer
    # For ViT-B/32, it's 512. For ViT-L/14, it's 768.
    # This is a common way to get it, but might need adjustment if using a custom model
    try:
        return model.text_projection.shape[1] if hasattr(model, 'text_projection') and model.text_projection is not None else model.visual.output_dim
    except AttributeError:
        # Fallback for models where text_projection might not be directly accessible or named differently
        # For standard OpenAI CLIP models, this should generally work or the specific dimension is known.
        if MODEL_NAME == "ViT-B/32":
            return 512
        elif MODEL_NAME == "ViT-L/14":
            return 768
        else:
            # You might need to manually set this or find a more robust way for other models
            print(f"Warning: Could not automatically determine embedding dimension for {MODEL_NAME}. Returning default 512.")
            return 512 # Defaulting, ensure this is correct for your model

# Example usage (optional, for testing)
if __name__ == "__main__":
    if model and preprocess:
        print(f"Embedding dimension: {get_embedding_dimension()}")
        # Test with a dummy image
        try:
            dummy_image = Image.new('RGB', (224, 224), color = 'red')
            img_emb = get_image_embedding(dummy_image)
            print(f"Dummy image embedding (first 5 features): {img_emb[:5]}")
        except Exception as e:
            print(f"Error in image embedding example: {e}")

        # Test with dummy text
        try:
            text_emb = get_text_embedding("a red square")
            print(f"Dummy text embedding (first 5 features): {text_emb[:5]}")
        except Exception as e:
            print(f"Error in text embedding example: {e}")
    else:
        print("CLIP model not loaded, skipping examples.") 