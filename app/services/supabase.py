import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Tuple
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

async def find_products_by_embedding(
    embedding: List[float], 
    match_threshold: float = 0.7, 
    match_count: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None,
    use_augmentations: bool = True
) -> List[Product]:
    """
    Finds products by semantic similarity using vector embeddings.
    Can filter candidates by metadata before vector similarity search.
    Can use augmented embeddings for better matching.
    
    Args:
        embedding: The query embedding vector
        match_threshold: Minimum similarity threshold
        match_count: Maximum number of results to return
        metadata_filters: Dict of field-value pairs to filter candidates (category, brand, color, etc.)
        use_augmentations: Whether to include augmented embeddings in the search
    """
    try:
        # If we have metadata filters and want to use them for pre-filtering
        if metadata_filters and len(metadata_filters) > 0:
            # Use the enhanced match_products_with_filters RPC
            response = supabase.rpc(
                "match_products_with_filters", 
                {
                    "query_embedding": embedding, 
                    "match_threshold": match_threshold, 
                    "match_count": match_count,
                    "metadata_filters": metadata_filters,
                    "use_augmentations": use_augmentations
                }
            ).execute()
        else:
            # Use the enhanced match_products function that can include augmentations
            response = supabase.rpc(
                "match_products", 
                {
                    "query_embedding": embedding, 
                    "match_threshold": match_threshold, 
                    "match_count": match_count,
                    "use_augmentations": use_augmentations
                }
            ).execute()

        # Check for explicit error in the response structure first
        if hasattr(response, 'error') and response.error:
            if hasattr(response.error, 'message'):
                print(f"Error from match_products RPC: {response.error.message}")
            else:
                print(f"Error from match_products RPC: {response.error}")
            return []

        # Check if data is present and process it
        if hasattr(response, 'data') and response.data:
            products_list = []
            for item_data in response.data:
                if 'clip_embedding' in item_data and isinstance(item_data['clip_embedding'], str):
                    item_data['clip_embedding'] = _parse_embedding_str(item_data['clip_embedding'])
                products_list.append(Product(**item_data))
            return products_list
        
        return []

    except Exception as e:
        print(f"Exception in find_products_by_embedding: {e}")
        if hasattr(e, 'status_code'):
             print(f"Underlying status code: {e.status_code}")
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
        return None
    except Exception as e:
        print(f"Error fetching product by ID {product_id}: {e}")
        return None

async def update_product_embedding(product_id: int, embedding: List[float]) -> bool:
    """Updates the CLIP embedding for an existing product in the database."""
    try:
        response = supabase.table("products")\
            .update({"clip_embedding": embedding})\
            .eq("id", product_id)\
            .execute()

        if response.data:
            print(f"Successfully updated embedding for product ID: {product_id}")
            return True
        elif response.error and hasattr(response.error, 'message'):
            print(f"Error updating embedding for product ID {product_id}: {response.error.message}")
            return False
        elif response.error:
            print(f"Error updating embedding for product ID {product_id}: {response.error}")
            return False
        else:
            print(f"No data returned when updating embedding for product ID {product_id}. Product may not exist.")
            return False

    except Exception as e:
        print(f"Exception updating embedding for product ID {product_id}: {e}")
        return False

async def add_augmented_embedding(
    product_id: int, 
    embedding: List[float], 
    augmentation_type: str
) -> bool:
    """
    Adds an augmented embedding for a product to the product_embeddings table.
    
    Args:
        product_id: The ID of the product
        embedding: The CLIP embedding vector
        augmentation_type: Type of augmentation (e.g., "rotation_0", "scale_1")
    """
    try:
        # Insert into product_embeddings table
        response = supabase.table("product_embeddings").insert({
            "product_id": product_id,
            "embedding_vector": embedding,
            "augmentation_type": augmentation_type
        }).execute()
        
        if response.data:
            print(f"Added augmented embedding ({augmentation_type}) for product ID: {product_id}")
            return True
        elif response.error:
            error_message = getattr(response.error, 'message', str(response.error))
            print(f"Error adding augmented embedding for product ID {product_id}: {error_message}")
            return False
        else:
            print(f"No data or explicit error when adding augmented embedding for product ID {product_id}")
            return False
            
    except Exception as e:
        print(f"Exception adding augmented embedding for product ID {product_id}: {e}")
        return False

async def update_product_augmentation_count(product_id: int, count: int) -> bool:
    """
    Updates the augmented_embeddings_count for a product.
    """
    try:
        response = supabase.table("products")\
            .update({"augmented_embeddings_count": count})\
            .eq("id", product_id)\
            .execute()
            
        if response.data:
            print(f"Updated augmentation count for product ID: {product_id} to {count}")
            return True
        elif response.error:
            error_message = getattr(response.error, 'message', str(response.error))
            print(f"Error updating augmentation count for product ID {product_id}: {error_message}")
            return False
        else:
            print(f"No data or explicit error when updating augmentation count for product ID {product_id}")
            return False
            
    except Exception as e:
        print(f"Exception updating augmentation count for product ID {product_id}: {e}")
        return False

async def get_augmented_embeddings(product_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves all augmented embeddings for a product.
    """
    try:
        response = supabase.table("product_embeddings")\
            .select("*")\
            .eq("product_id", product_id)\
            .execute()
            
        if response.data:
            embeddings_list = []
            for item in response.data:
                if 'embedding_vector' in item and isinstance(item['embedding_vector'], str):
                    item['embedding_vector'] = _parse_embedding_str(item['embedding_vector'])
                embeddings_list.append(item)
            return embeddings_list
        return []
            
    except Exception as e:
        print(f"Error fetching augmented embeddings for product ID {product_id}: {e}")
        return []

async def filter_products_by_metadata(
    metadata_filters: Dict[str, Any],
    limit: int = 100
) -> List[Product]:
    """
    Filters products based on metadata fields.
    
    Args:
        metadata_filters: Dict of field-value pairs to filter by
        limit: Maximum number of results to return
    """
    try:
        query = supabase.table("products").select("*")
        
        # Apply each filter
        for field, value in metadata_filters.items():
            if field in ["name", "variant", "category", "brand", "color", "barcode"]:
                if isinstance(value, str):
                    # For string fields, use ilike for case-insensitive partial matching
                    query = query.ilike(field, f"%{value}%")
                else:
                    # For exact matching of other types
                    query = query.eq(field, value)
            elif field == "tags" and isinstance(value, list):
                # For tags, which is an array, use the contains operator
                query = query.contains(field, value)
        
        # Execute the query with limit
        response = query.limit(limit).execute()
        
        if response.data:
            products_list = []
            for item_data in response.data:
                if 'clip_embedding' in item_data and isinstance(item_data['clip_embedding'], str):
                    item_data['clip_embedding'] = _parse_embedding_str(item_data['clip_embedding'])
                products_list.append(Product(**item_data))
            return products_list
        return []
        
    except Exception as e:
        print(f"Error filtering products by metadata: {e}")
        return []

async def get_augmented_embeddings_count(product_id: int) -> int:
    """
    Returns the count of augmented embeddings for a product.
    
    Args:
        product_id: The ID of the product to check
        
    Returns:
        The count of augmented embeddings
    """
    try:
        response = supabase.table("product_embeddings")\
            .select("id", count="exact")\
            .eq("product_id", product_id)\
            .execute()
        
        if hasattr(response, 'count') and response.count is not None:
            return response.count
        return 0
    
    except Exception as e:
        print(f"Error getting augmented embeddings count for product ID {product_id}: {e}")
        return 0

# --- SQL for database schema updates ---

# 1. Update products table to add new metadata fields
'''
ALTER TABLE products 
ADD COLUMN category TEXT,
ADD COLUMN brand TEXT,
ADD COLUMN color TEXT,
ADD COLUMN barcode TEXT,
ADD COLUMN dimensions JSONB,
ADD COLUMN tags TEXT[],
ADD COLUMN augmented_embeddings_count INTEGER DEFAULT 0;
'''

# 2. Create product_embeddings table for augmented embeddings
'''
CREATE TABLE product_embeddings (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    product_id BIGINT REFERENCES products(id) ON DELETE CASCADE,
    embedding_vector VECTOR(512), -- Match the dimension of your CLIP model
    augmentation_type TEXT,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX ON product_embeddings USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON product_embeddings (product_id);
'''

# 3. Update the match_products function to support augmentations
'''
CREATE OR REPLACE FUNCTION match_products (
  query_embedding vector(512),
  match_threshold float,
  match_count int,
  use_augmentations boolean DEFAULT false
)
RETURNS TABLE (
  id BIGINT,
  name TEXT,
  variant TEXT,
  category TEXT,
  brand TEXT,
  color TEXT,
  barcode TEXT,
  image_url TEXT,
  clip_embedding vector(512),
  augmented_embeddings_count INTEGER,
  created_at TIMESTAMPTZ,
  similarity float
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  IF use_augmentations THEN
    -- Search using both original and augmented embeddings
    RETURN QUERY
    WITH matches AS (
      -- First search original embeddings
      SELECT 
        p.id,
        1 - (p.clip_embedding <=> query_embedding) AS similarity,
        TRUE as is_original
      FROM products p
      WHERE p.clip_embedding IS NOT NULL AND 1 - (p.clip_embedding <=> query_embedding) > match_threshold
      
      UNION ALL
      
      -- Then search augmented embeddings
      SELECT 
        pe.product_id AS id,
        1 - (pe.embedding_vector <=> query_embedding) AS similarity,
        FALSE as is_original
      FROM product_embeddings pe
      WHERE 1 - (pe.embedding_vector <=> query_embedding) > match_threshold
    ),
    -- Get top match per product (could be original or augmented)
    best_matches AS (
      SELECT DISTINCT ON (id) id, similarity, is_original
      FROM matches
      ORDER BY id, similarity DESC
    )
    
    SELECT
      p.id,
      p.name,
      p.variant,
      p.category,
      p.brand,
      p.color,
      p.barcode,
      p.image_url,
      p.clip_embedding,
      p.augmented_embeddings_count,
      p.created_at,
      bm.similarity
    FROM best_matches bm
    JOIN products p ON p.id = bm.id
    ORDER BY bm.similarity DESC
    LIMIT match_count;
  ELSE
    -- Original functionality: search only in original embeddings
    RETURN QUERY
    SELECT
      p.id,
      p.name,
      p.variant,
      p.category,
      p.brand,
      p.color,
      p.barcode,
      p.image_url,
      p.clip_embedding,
      p.augmented_embeddings_count,
      p.created_at,
      1 - (p.clip_embedding <=> query_embedding) AS similarity
    FROM products p
    WHERE p.clip_embedding IS NOT NULL AND 1 - (p.clip_embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
  END IF;
END;
$$;
'''

# 4. Create a function for metadata filtering + embedding search
'''
CREATE OR REPLACE FUNCTION match_products_with_filters (
  query_embedding vector(512),
  match_threshold float,
  match_count int,
  metadata_filters jsonb,
  use_augmentations boolean DEFAULT false
)
RETURNS TABLE (
  id BIGINT,
  name TEXT,
  variant TEXT,
  category TEXT,
  brand TEXT,
  color TEXT,
  barcode TEXT,
  image_url TEXT,
  clip_embedding vector(512),
  augmented_embeddings_count INTEGER,
  created_at TIMESTAMPTZ,
  similarity float
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
  category_filter TEXT := metadata_filters->>'category';
  brand_filter TEXT := metadata_filters->>'brand';
  color_filter TEXT := metadata_filters->>'color';
  barcode_filter TEXT := metadata_filters->>'barcode';
  name_filter TEXT := metadata_filters->>'name';
  variant_filter TEXT := metadata_filters->>'variant';
BEGIN
  IF use_augmentations THEN
    -- First filter by metadata, then search with both original and augmented embeddings
    RETURN QUERY
    WITH filtered_products AS (
      SELECT p.id
      FROM products p
      WHERE 
        (category_filter IS NULL OR p.category ILIKE concat('%', category_filter, '%')) AND
        (brand_filter IS NULL OR p.brand ILIKE concat('%', brand_filter, '%')) AND
        (color_filter IS NULL OR p.color ILIKE concat('%', color_filter, '%')) AND
        (barcode_filter IS NULL OR p.barcode ILIKE concat('%', barcode_filter, '%')) AND
        (name_filter IS NULL OR p.name ILIKE concat('%', name_filter, '%')) AND
        (variant_filter IS NULL OR p.variant ILIKE concat('%', variant_filter, '%'))
    ),
    matches AS (
      -- Search original embeddings for filtered products
      SELECT 
        p.id,
        1 - (p.clip_embedding <=> query_embedding) AS similarity,
        TRUE as is_original
      FROM products p
      JOIN filtered_products fp ON p.id = fp.id
      WHERE p.clip_embedding IS NOT NULL AND 1 - (p.clip_embedding <=> query_embedding) > match_threshold
      
      UNION ALL
      
      -- Search augmented embeddings for filtered products
      SELECT 
        pe.product_id AS id,
        1 - (pe.embedding_vector <=> query_embedding) AS similarity,
        FALSE as is_original
      FROM product_embeddings pe
      JOIN filtered_products fp ON pe.product_id = fp.id
      WHERE 1 - (pe.embedding_vector <=> query_embedding) > match_threshold
    ),
    -- Get top match per product
    best_matches AS (
      SELECT DISTINCT ON (id) id, similarity, is_original
      FROM matches
      ORDER BY id, similarity DESC
    )
    
    SELECT
      p.id,
      p.name,
      p.variant,
      p.category,
      p.brand,
      p.color,
      p.barcode,
      p.image_url,
      p.clip_embedding,
      p.augmented_embeddings_count,
      p.created_at,
      bm.similarity
    FROM best_matches bm
    JOIN products p ON p.id = bm.id
    ORDER BY bm.similarity DESC
    LIMIT match_count;
  ELSE
    -- Filter by metadata and search only in original embeddings
    RETURN QUERY
    SELECT
      p.id,
      p.name,
      p.variant,
      p.category,
      p.brand,
      p.color,
      p.barcode,
      p.image_url,
      p.clip_embedding,
      p.augmented_embeddings_count,
      p.created_at,
      1 - (p.clip_embedding <=> query_embedding) AS similarity
    FROM products p
    WHERE 
      p.clip_embedding IS NOT NULL AND 
      1 - (p.clip_embedding <=> query_embedding) > match_threshold AND
      (category_filter IS NULL OR p.category ILIKE concat('%', category_filter, '%')) AND
      (brand_filter IS NULL OR p.brand ILIKE concat('%', brand_filter, '%')) AND
      (color_filter IS NULL OR p.color ILIKE concat('%', color_filter, '%')) AND
      (barcode_filter IS NULL OR p.barcode ILIKE concat('%', barcode_filter, '%')) AND
      (name_filter IS NULL OR p.name ILIKE concat('%', name_filter, '%')) AND
      (variant_filter IS NULL OR p.variant ILIKE concat('%', variant_filter, '%'))
    ORDER BY similarity DESC
    LIMIT match_count;
  END IF;
END;
$$;
''' 