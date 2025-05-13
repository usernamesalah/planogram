import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from app.models.product import ProductCreate, Product # Assuming Product model is defined
import json # Added for parsing embedding string

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("Supabase URL and Key must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper Function ---
def _parse_embedding_str(embedding_str: Optional[str]) -> Optional[List[float]]:
    if embedding_str is None:
        return None
    try:
        # Assuming the string from pgvector is a JSON-like array string, e.g., "[0.1, 0.2, ...]"
        return json.loads(embedding_str)
    except json.JSONDecodeError:
        # Handle cases where it might not be a valid JSON string, or already a list (though unlikely from DB)
        # For now, return None or raise an error if it's critical
        print(f"Warning: Could not parse embedding string: {embedding_str}")
        return None

# --- Product Operations ---

async def add_product_to_db(product_data: ProductCreate, clip_embedding: Optional[List[float]] = None) -> Product:
    """Adds a new product to the Supabase database."""
    try:
        insert_data = product_data.model_dump()
        if clip_embedding:
            # pgvector expects a string representation like '[1,2,3]' or the library might handle list directly
            # For supabase-py, sending a list directly for a vector type is usually fine.
            # If it were raw SQL, you'd use a string. Let's assume the library handles it.
            insert_data['clip_embedding'] = clip_embedding

        response = supabase.table("products").insert(insert_data).execute()
        
        if response.data:
            created_product_data = response.data[0]
            # Parse embedding if it's returned as a string
            if 'clip_embedding' in created_product_data and isinstance(created_product_data['clip_embedding'], str):
                created_product_data['clip_embedding'] = _parse_embedding_str(created_product_data['clip_embedding'])
            return Product(**created_product_data)
        else:
            error_message = "Unknown error"
            if response.error and hasattr(response.error, 'message'):
                error_message = response.error.message
            elif hasattr(response, 'status_text'): # Fallback for some error responses
                error_message = response.status_text
            print(f"Error adding product: {error_message}")
            raise Exception(f"Failed to add product: {error_message}")

    except Exception as e:
        print(f"Error in add_product_to_db: {e}")
        raise

async def get_all_products_from_db(limit: int = 100) -> List[Product]:
    """Retrieves all products from the database."""
    try:
        response = supabase.table("products").select("*").limit(limit).execute()
        if response.data:
            products_list = []
            for item_data in response.data:
                if 'clip_embedding' in item_data and isinstance(item_data['clip_embedding'], str):
                    item_data['clip_embedding'] = _parse_embedding_str(item_data['clip_embedding'])
                products_list.append(Product(**item_data))
            return products_list
        return []
    except Exception as e:
        print(f"Error fetching all products: {e}")
        # The log shows Pydantic validation error here, so ensure parsing happens before Product(**item)
        # The above modification should address this.
        return []

async def find_products_by_embedding(embedding: List[float], match_threshold: float = 0.7, match_count: int = 5) -> List[Product]:
    """
    Finds products by semantic similarity using vector embeddings.
    Requires a PostgreSQL function in Supabase for vector similarity (e.g., using pgvector).
    Example function: match_products(query_embedding vector, match_threshold float, match_count int)
    """
    try:
        response = supabase.rpc(
            "match_products", 
            {"query_embedding": embedding, "match_threshold": match_threshold, "match_count": match_count}
        ).execute()

        # Check for explicit error in the response structure first
        if hasattr(response, 'error') and response.error:
            if hasattr(response.error, 'message'):
                print(f"Error from match_products RPC (response.error.message): {response.error.message}")
            else:
                print(f"Error from match_products RPC (response.error): {response.error}")
            return []

        # Check if data is present and process it
        if hasattr(response, 'data') and response.data:
            products_list = []
            for item_data in response.data:
                if 'clip_embedding' in item_data and isinstance(item_data['clip_embedding'], str):
                    item_data['clip_embedding'] = _parse_embedding_str(item_data['clip_embedding'])
                products_list.append(Product(**item_data))
            return products_list
        
        # If no data and no explicit error, it could be a successful call with no matches,
        # or an implicit error (e.g. non-2xx HTTP status code not caught as response.error)
        # print(f"match_products RPC returned no data and no explicit error. Response: {response}") # Optional: for debugging
        return []

    except Exception as e:
        # This block will catch exceptions during the .execute() call or if response object itself indicates an error not caught above.
        error_message = str(e)
        if hasattr(e, 'message'): # Common for Supabase APIError objects
             error_message = e.message
        elif hasattr(e, 'details'): # Sometimes errors are in 'details'
             error_message = f"{error_message} - Details: {e.details}"
        
        print(f"Exception in find_products_by_embedding. Type: {type(e)}. Message: {error_message}")
        if hasattr(e, 'status_code'):
             print(f"Underlying status code (if available): {e.status_code}")
        
        return []

async def get_product_by_id_from_db(product_id: int) -> Optional[Product]:
    """Retrieves a single product by its ID."""
    try:
        response = supabase.table("products").select("*").eq("id", product_id).maybe_single().execute()
        if response.data:
            item_data = response.data
            if 'clip_embedding' in item_data and isinstance(item_data['clip_embedding'], str):
                item_data['clip_embedding'] = _parse_embedding_str(item_data['clip_embedding'])
            return Product(**item_data)
        # No explicit error check here, maybe_single() handles not found gracefully by returning data as None.
        return None
    except Exception as e:
        print(f"Error fetching product by ID {product_id}: {e}")
        return None

async def update_product_embedding(product_id: int, embedding: List[float]) -> bool:
    """Updates the CLIP embedding for an existing product in the database."""
    try:
        # Assuming supabase-py handles Python list of floats to vector string conversion.
        response = supabase.table("products")\
            .update({"clip_embedding": embedding})\
            .eq("id", product_id)\
            .execute()

        if response.data: # Update usually returns the updated records in data
            print(f"Successfully updated embedding for product ID: {product_id}")
            return True
        # Check for error if no data (though successful update with no data return is also possible if returning='minimal')
        elif response.error and hasattr(response.error, 'message'):
            print(f"Error updating embedding for product ID {product_id}: {response.error.message}")
            return False
        elif response.error:
            print(f"Error updating embedding for product ID {product_id}: {response.error}")
            return False
        else:
            print(f"No data returned or error encountered when updating embedding for product ID {product_id}. Product may not exist or no change made.")
            # This could mean the product_id didn't match, which isn't necessarily an error for the update operation itself.
            return False # Consider what to return: True if "no error, no update needed" is success.

    except Exception as e:
        print(f"Exception updating embedding for product ID {product_id}: {e}")
        return False

# Placeholder for database schema migration info
# You'll need to run SQL in Supabase to create the 'products' table and the 'match_products' function.
#
# Example SQL for 'products' table:
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE products (
#     id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
#     name TEXT NOT NULL,
#     variant TEXT,
#     image_url TEXT,
#     clip_embedding VECTOR(512), -- Or 768, or whatever your CLIP model's dimension is
#     created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
# );
# CREATE INDEX ON products USING ivfflat (clip_embedding vector_cosine_ops) WITH (lists = 100); -- Or HNSW for better recall
#
# Example SQL for 'match_products' function (using cosine similarity):
# CREATE OR REPLACE FUNCTION match_products (
#   query_embedding vector(512), -- Match embedding dimension
#   match_threshold float,
#   match_count int
# )
# RETURNS TABLE (
#   id BIGINT,
#   name TEXT,
#   variant TEXT,
#   image_url TEXT,
#   clip_embedding vector(512),
#   created_at TIMESTAMPTZ,
#   similarity float
# )
# LANGUAGE sql STABLE
# AS $$
#   SELECT
#     p.id,
#     p.name,
#     p.variant,
#     p.image_url,
#     p.clip_embedding,
#     p.created_at,
#     1 - (p.clip_embedding <=> query_embedding) AS similarity -- Cosine distance, 1 - dist = similarity
#   FROM products p
#   WHERE 1 - (p.clip_embedding <=> query_embedding) > match_threshold
#   ORDER BY similarity DESC
#   LIMIT match_count;
# $$; 