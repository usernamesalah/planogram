-- Fix for ambiguous column references in match_products and match_products_with_filters functions

-- 1. Drop existing functions
DROP FUNCTION IF EXISTS match_products(vector(512), float, int, boolean);
DROP FUNCTION IF EXISTS match_products_with_filters(vector(512), float, int, jsonb, boolean);

-- 2. Recreate match_products function
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
        p.id AS product_id,
        1 - (p.clip_embedding <=> query_embedding) AS match_similarity,
        TRUE as is_original
      FROM products p
      WHERE p.clip_embedding IS NOT NULL AND 1 - (p.clip_embedding <=> query_embedding) > match_threshold
      
      UNION ALL
      
      -- Then search augmented embeddings
      SELECT 
        pe.product_id,
        1 - (pe.embedding_vector <=> query_embedding) AS match_similarity,
        FALSE as is_original
      FROM product_embeddings pe
      WHERE 1 - (pe.embedding_vector <=> query_embedding) > match_threshold
    ),
    -- Get top match per product (could be original or augmented)
    best_matches AS (
      SELECT DISTINCT ON (product_id) 
        product_id, 
        match_similarity, 
        is_original
      FROM matches
      ORDER BY product_id, match_similarity DESC
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
      bm.match_similarity AS similarity
    FROM best_matches bm
    JOIN products p ON p.id = bm.product_id
    ORDER BY bm.match_similarity DESC
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

-- 3. Recreate match_products_with_filters function
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
      SELECT p.id AS product_id
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
        p.id AS product_id,
        1 - (p.clip_embedding <=> query_embedding) AS match_similarity,
        TRUE as is_original
      FROM products p
      JOIN filtered_products fp ON p.id = fp.product_id
      WHERE p.clip_embedding IS NOT NULL AND 1 - (p.clip_embedding <=> query_embedding) > match_threshold
      
      UNION ALL
      
      -- Search augmented embeddings for filtered products
      SELECT 
        pe.product_id,
        1 - (pe.embedding_vector <=> query_embedding) AS match_similarity,
        FALSE as is_original
      FROM product_embeddings pe
      JOIN filtered_products fp ON pe.product_id = fp.product_id
      WHERE 1 - (pe.embedding_vector <=> query_embedding) > match_threshold
    ),
    -- Get top match per product
    best_matches AS (
      SELECT DISTINCT ON (product_id) 
        product_id, 
        match_similarity, 
        is_original
      FROM matches
      ORDER BY product_id, match_similarity DESC
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
      bm.match_similarity AS similarity
    FROM best_matches bm
    JOIN products p ON p.id = bm.product_id
    ORDER BY bm.match_similarity DESC
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