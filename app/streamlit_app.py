import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv
import io
from PIL import Image

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Planogram Detection")

st.title("Planogram Compliance & Product Management")

# --- Helper Functions ---
def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Product Detection", "Product Management", "Planogram Analysis"])

# --- Product Detection Page ---
if app_mode == "Product Detection":
    st.header("Product Detection")
    
    # Add tabs for different detection modes
    detection_mode = st.radio("Detection Mode", ["Standard", "Enhanced (with metadata filtering)"])
    
    uploaded_image = st.file_uploader("Upload an image for product detection", type=["jpg", "jpeg", "png"])
    
    # Add metadata filtering options for enhanced detection
    metadata_filters = {}
    if detection_mode == "Enhanced (with metadata filtering)":
        st.subheader("Metadata Filters (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input("Category")
            brand = st.text_input("Brand")
            color = st.text_input("Color")
        with col2:
            barcode = st.text_input("Barcode")
            name_filter = st.text_input("Product Name")
            variant_filter = st.text_input("Variant")
        
        # Only add non-empty filters
        if category: metadata_filters["category"] = category
        if brand: metadata_filters["brand"] = brand
        if color: metadata_filters["color"] = color
        if barcode: metadata_filters["barcode"] = barcode
        if name_filter: metadata_filters["name"] = name_filter
        if variant_filter: metadata_filters["variant"] = variant_filter

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Products"):
            with st.spinner("Detecting products..."):
                b64_image = image_to_base64(uploaded_image)
                try:
                    # Use the appropriate endpoint based on detection mode
                    if detection_mode == "Standard":
                        response = requests.post(f"{API_URL}/detect/", json={"image": b64_image})
                    else:
                        response = requests.post(f"{API_URL}/detect/detailed/", 
                                               json={"image": b64_image, "filter_by_metadata": metadata_filters})
                    
                    response.raise_for_status()
                    detection_data = response.json()
                    st.success("Detection successful!")
                    
                    if detection_data.get("annotated_image"):
                        annotated_img_bytes = base64.b64decode(detection_data["annotated_image"])
                        st.image(annotated_img_bytes, caption="Detected Products", use_column_width=True)
                    
                    st.subheader("Detected Products:")
                    if detection_data.get("products"):
                        for prod in detection_data["products"]:
                            # Display more metadata if available
                            metadata_str = ""
                            if prod.get("category"): metadata_str += f"Category: {prod['category']}, "
                            if prod.get("brand"): metadata_str += f"Brand: {prod['brand']}, "
                            if prod.get("color"): metadata_str += f"Color: {prod['color']}"
                            
                            st.write(f"- {prod.get('name')} (Variant: {prod.get('variant', 'N/A')})")
                            if metadata_str:
                                st.write(f"  {metadata_str}")
                            
                            # Show augmentation count if available
                            if prod.get("augmented_embeddings_count", 0) > 0:
                                st.write(f"  Augmentations: {prod.get('augmented_embeddings_count')}")
                    else:
                        st.write("No products detected or no product list in response.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling detection API: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

# --- Product Management Page ---
elif app_mode == "Product Management":
    st.header("Product Management")
    
    # Add New Product
    st.subheader("Add New Product")
    with st.form("new_product_form", clear_on_submit=True):
        product_name = st.text_input("Product Name")
        product_variant = st.text_input("Product Variant (Optional)")
        
        # Add metadata fields
        st.subheader("Product Metadata")
        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input("Category")
            brand = st.text_input("Brand")
            color = st.text_input("Color")
        with col2:
            barcode = st.text_input("Barcode")
            tags = st.text_input("Tags (comma-separated)")
        
        # Image options
        product_image_url_input = st.text_input("Product Image URL (Optional - if not uploading file)")
        st.text("Upload Product Images (Recommended - can select multiple)")
        uploaded_product_images = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        st.info("""**Enhanced Image Processing:**
        1. Each uploaded image will be processed in two ways:
           - Original image: Full product with packaging and background
           - Color-enhanced: Specialized saturation enhancement for colorful packaging
        2. Both versions will be used to create embeddings
        3. Multiple augmentations focusing on color variations will be generated
        4. This specifically improves detection of bright, colorful products like Kuaci Rebo snacks""")
        
        generate_augmentations = st.checkbox("Generate image augmentations", value=True)
        
        submitted = st.form_submit_button("Add Product")

        if submitted and product_name:
            with st.spinner("Adding product..."):
                # Prepare form data
                form_data = {
                    "name": (None, product_name),
                    "variant": (None, product_variant if product_variant else ""),
                    "image_url": (None, product_image_url_input if product_image_url_input else ""),
                    "category": (None, category if category else ""),
                    "brand": (None, brand if brand else ""),
                    "color": (None, color if color else ""),
                    "barcode": (None, barcode if barcode else ""),
                    "tags": (None, tags if tags else "")
                }
                
                files_data = {}
                
                try:
                    # First, create the product
                    if uploaded_product_images and len(uploaded_product_images) > 0:
                        # Add the first image during product creation
                        first_image = uploaded_product_images[0]
                        files_data["image_upload"] = (first_image.name, first_image.getvalue(), first_image.type)
                        response = requests.post(f"{API_URL}/products/", data=form_data, files=files_data)
                    else:
                        # No files, just form data
                        simple_data_payload = {k: v[1] for k, v in form_data.items()}
                        response = requests.post(f"{API_URL}/products/", data=simple_data_payload)
                    
                    response.raise_for_status()
                    product_data = response.json()
                    product_id = product_data.get("id")
                    
                    # If there are additional images, upload them
                    if uploaded_product_images and len(uploaded_product_images) > 1:
                        additional_images = uploaded_product_images[1:]
                        multi_files = []
                        
                        for img in additional_images:
                            multi_files.append(('images', (img.name, img.getvalue(), img.type)))
                        
                        multi_response = requests.post(
                            f"{API_URL}/products/upload-images/",
                            data={"product_id": product_id},
                            files=multi_files
                        )
                        multi_response.raise_for_status()
                        
                    st.success(f"Product '{product_name}' added successfully with {len(uploaded_product_images)} images!")
                    
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Error adding product: {http_err} - {http_err.response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error adding product: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        elif submitted and not product_name:
            st.error("Product Name is required.")

    # View Products
    st.subheader("Existing Products")
    if st.button("Refresh Product List"):
        try:
            response = requests.get(f"{API_URL}/products/")
            response.raise_for_status()
            products = response.json()
            
            if products:
                for prod in products:
                    # Create expandable section for each product
                    with st.expander(f"{prod.get('name')} (ID: {prod.get('id')})"):
                        # Basic info
                        st.write(f"**Variant:** {prod.get('variant', 'N/A')}")
                        
                        # Metadata
                        metadata_col1, metadata_col2 = st.columns(2)
                        with metadata_col1:
                            st.write(f"**Category:** {prod.get('category', 'N/A')}")
                            st.write(f"**Brand:** {prod.get('brand', 'N/A')}")
                            st.write(f"**Color:** {prod.get('color', 'N/A')}")
                        with metadata_col2:
                            st.write(f"**Barcode:** {prod.get('barcode', 'N/A')}")
                            st.write(f"**Tags:** {', '.join(prod.get('tags', []))}")
                        
                        # Augmentation info
                        st.write(f"**Augmentations:** {prod.get('augmented_embeddings_count', 0)}")
                        
                        # Image if available
                        if prod.get('image_url'):
                            st.image(prod.get('image_url'), width=200)
            else:
                st.write("No products found.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching products: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Planogram Analysis Page ---
elif app_mode == "Planogram Analysis":
    st.header("Planogram Compliance Analysis")
    actual_planogram_image = st.file_uploader("Upload actual planogram image", type=["jpg", "jpeg", "png"])
    
    # Add metadata filtering for planogram analysis
    st.subheader("Metadata Filters (Optional)")
    st.write("These filters help narrow down the product search during analysis")
    
    metadata_col1, metadata_col2 = st.columns(2)
    with metadata_col1:
        category_filter = st.text_input("Category Filter")
        brand_filter = st.text_input("Brand Filter")
    with metadata_col2:
        color_filter = st.text_input("Color Filter")
        name_filter = st.text_input("Name Filter")
    
    # Prepare metadata filters
    metadata_filters = {}
    if category_filter: metadata_filters["category"] = category_filter
    if brand_filter: metadata_filters["brand"] = brand_filter
    if color_filter: metadata_filters["color"] = color_filter
    if name_filter: metadata_filters["name"] = name_filter
    
    st.subheader("Define Expected Planogram Layout")

    # Fetch products for dropdown selection
    available_products = []
    try:
        response = requests.get(f"{API_URL}/products/")
        response.raise_for_status()
        fetched_prods_data = response.json()
        if fetched_prods_data:
            available_products = fetched_prods_data
    except Exception as e:
        st.error(f"Error fetching product list: {e}. Please ensure API is running and products exist.")

    if not available_products:
        st.warning("No products found in the database. Please add products first via Product Management.")

    # Initialize expected_layout in session state if it doesn't exist
    if 'expected_layout_items' not in st.session_state:
        st.session_state.expected_layout_items = []

    # Display existing expected items
    for i, item in enumerate(st.session_state.expected_layout_items):
        pid = item.get("product_id", "N/A")
        pname = item.get("name", "Unknown Product")
        box_str = str(item.get("expected_box", "Not set"))
        st.text(f"Item {i+1}: {pname} (ID: {pid}) at Box: {box_str}")

    with st.form("add_expected_item_form"):
        st.write("**Add New Expected Product Slot:**")
        if available_products:
            # Create a list of product names for the selectbox, mapping back to ID
            product_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in available_products}
            selected_product_display_name = st.selectbox(
                "Select Product", 
                options=list(product_options.keys()), 
                key=f"exp_prod_select"
            )
            selected_product_id = product_options.get(selected_product_display_name)
            selected_product_name = selected_product_display_name # Store full display name for later retrieval
        else:
            st.text("No products available to select.")
            selected_product_id = None
            selected_product_name = None

        st.text("Expected Bounding Box (x1, y1, x2, y2 - top-left and bottom-right coordinates):")
        cols_box = st.columns(4)
        x1 = cols_box[0].number_input("X1", min_value=0, step=1, key="exp_x1")
        y1 = cols_box[1].number_input("Y1", min_value=0, step=1, key="exp_y1")
        x2 = cols_box[2].number_input("X2", min_value=0, step=1, key="exp_x2")
        y2 = cols_box[3].number_input("Y2", min_value=0, step=1, key="exp_y2")
        
        add_item_submitted = st.form_submit_button("Add Expected Item")

        if add_item_submitted and selected_product_id and x2 > x1 and y2 > y1:
            # Find the original product name from available_products using selected_product_id
            product_details = next((p for p in available_products if p['id'] == selected_product_id), None)
            display_name_for_item = product_details['name'] if product_details else "Unknown Product"

            st.session_state.expected_layout_items.append({
                "product_id": selected_product_id,
                "name": display_name_for_item,
                "expected_box": [int(x1), int(y1), int(x2), int(y2)]
            })
            st.success(f"Added {display_name_for_item} to expected layout.")
            st.rerun()
        elif add_item_submitted:
            st.error("Please select a product and ensure X2 > X1 and Y2 > Y1 for the box.")

    if st.button("Clear All Expected Items"):
        st.session_state.expected_layout_items = []
        st.rerun()

    if actual_planogram_image and st.session_state.expected_layout_items:
        st.image(actual_planogram_image, caption="Actual Planogram", use_column_width=True)
        if st.button("Analyze Planogram Compliance"):
            with st.spinner("Analyzing planogram..."):
                b64_actual_image = image_to_base64(actual_planogram_image)
                api_expected_layout = st.session_state.expected_layout_items
                
                request_data = {
                    "actual_image": b64_actual_image,
                    "expected_layout": api_expected_layout,
                    "metadata_filters": metadata_filters if metadata_filters else None
                }
                
                try:
                    response = requests.post(f"{API_URL}/planogram/compare/", json=request_data)
                    response.raise_for_status()
                    analysis_data = response.json()
                    st.success("Planogram analysis complete!")
                    
                    st.metric("Compliance Score", f"{analysis_data.get('compliance_score', 0.0) * 100:.2f}%")
                    
                    if analysis_data.get("annotated_image"):
                        annotated_img_bytes = base64.b64decode(analysis_data["annotated_image"])
                        st.image(annotated_img_bytes, caption="Analysis Result", use_column_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Misplaced Products:")
                        if analysis_data.get("misplaced_products"):
                            for p in analysis_data["misplaced_products"]:
                                st.markdown(f"- 🟠 **{p.get('name')}**")
                                st.markdown(f"  - Expected: {p.get('expected_box')}")
                                st.markdown(f"  - Actual: {p.get('actual_box')}")
                                st.markdown(f"  - IoU: {p.get('iou', 0):.2f}")
                        else:
                            st.write("No misplaced products.")
                    
                    with col2:
                        st.subheader("Missing Products:")
                        if analysis_data.get("missing_products"):
                            for p in analysis_data["missing_products"]:
                                st.markdown(f"- 🔴 **{p.get('name')}**")
                                st.markdown(f"  - Expected position: {p.get('expected_box')}")
                                st.markdown(f"  - Reason: {p.get('reason', 'Unknown')}")
                        else:
                            st.write("No missing products.")
                    
                    # Add section for extra products (new feature)
                    st.subheader("Extra Products:")
                    if analysis_data.get("extra_products"):
                        for p in analysis_data["extra_products"]:
                            st.markdown(f"- 🟣 **{p.get('name')}**")
                            st.markdown(f"  - Position: {p.get('actual_box')}")
                            st.markdown(f"  - Reason: {p.get('reason', 'Unknown')}")
                    else:
                        st.write("No extra products found.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling planogram API: {e} - {response.text if 'response' in locals() else 'No response'}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("----")
st.sidebar.info(
    "This is a demo application for Planogram Detection. "
    "Enhanced with data augmentation, metadata filtering, and OCR."
) 