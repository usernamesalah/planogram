import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv

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
    uploaded_image = st.file_uploader("Upload an image for product detection", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Products"):
            with st.spinner("Detecting products..."):
                b64_image = image_to_base64(uploaded_image)
                try:
                    response = requests.post(f"{API_URL}/detect/", json={"image": b64_image})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    detection_data = response.json()
                    st.success("Detection successful!")
                    if detection_data.get("annotated_image"):
                        annotated_img_bytes = base64.b64decode(detection_data["annotated_image"])
                        st.image(annotated_img_bytes, caption="Detected Products", use_column_width=True)
                    st.subheader("Detected Products:")
                    if detection_data.get("products"):
                        for prod in detection_data["products"]:
                            st.write(f"- {prod.get('name')} (Variant: {prod.get('variant', 'N/A')})")
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
        # We will now prioritize file upload. The image_url field can remain for cases where an external URL is preferred
        # and the backend task is adapted to fetch from URL (currently it expects bytes).
        product_image_url_input = st.text_input("Product Image URL (Optional - if not uploading file)")
        uploaded_product_image_file = st.file_uploader("Upload Product Image (Recommended)", type=["jpg", "jpeg", "png"])
        
        submitted = st.form_submit_button("Add Product")

        if submitted and product_name:
            with st.spinner("Adding product..."):
                # Prepare form data
                form_data = {
                    "name": (None, product_name),
                    "variant": (None, product_variant if product_variant else ""),
                    "image_url": (None, product_image_url_input if product_image_url_input else "")
                }
                files_data = {}
                if uploaded_product_image_file:
                    # Add file to the files dictionary for multipart/form-data
                    files_data["image_upload"] = (uploaded_product_image_file.name, uploaded_product_image_file.getvalue(), uploaded_product_image_file.type)
                
                try:
                    # When sending files, requests typically doesn't use the `json` parameter.
                    # It uses `files` for file parts and `data` for other form fields.
                    if files_data: # If there's a file to upload
                        response = requests.post(f"{API_URL}/products/", data=form_data, files=files_data)
                    else: # If no file, just form data (e.g., only image_url provided)
                        # The API's /products/ endpoint expects Form data, not JSON when no file is sent.
                        # So, we should convert form_data to simple dict for the `data` param if no files.
                        simple_data_payload = {k: v[1] for k, v in form_data.items()}                        
                        response = requests.post(f"{API_URL}/products/", data=simple_data_payload)
                        
                    response.raise_for_status()
                    st.success(f"Product '{product_name}' added successfully! Processing image in background if provided.")
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
                    st.write(f"- {prod.get('name')} (Variant: {prod.get('variant', 'N/A')}, ID: {prod.get('id')})")
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
    
    st.subheader("Define Expected Planogram Layout")

    # Fetch products for dropdown selection
    available_products = []
    try:
        response = requests.get(f"{API_URL}/products/")
        response.raise_for_status()
        fetched_prods_data = response.json()
        if fetched_prods_data:
            # Create a list of tuples or objects suitable for st.selectbox options
            # [{'id': 1, 'name': 'Coke', ...}, ...]
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
            # This is a bit redundant if we store selected_product_name already, but safer.
            product_details = next((p for p in available_products if p['id'] == selected_product_id), None)
            display_name_for_item = product_details['name'] if product_details else "Unknown Product"

            st.session_state.expected_layout_items.append({
                "product_id": selected_product_id,
                "name": display_name_for_item, # Store name for display and potential API use
                "expected_box": [int(x1), int(y1), int(x2), int(y2)]
            })
            st.success(f"Added {display_name_for_item} to expected layout.")
            # Fields will clear due to form submission if not for session state trickiness
            # Forcing a rerun to update the displayed list above the form
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
                # Prepare the layout for the API (it already expects product_id and expected_box)
                api_expected_layout = st.session_state.expected_layout_items
                
                request_data = {
                    "actual_image": b64_actual_image,
                    "expected_layout": api_expected_layout
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

                    st.subheader("Misplaced Products:")
                    if analysis_data.get("misplaced_products"):
                        for p in analysis_data["misplaced_products"]:
                            st.write(f"- {p.get('name')}: Expected at {p.get('expected_location')}, Found at {p.get('actual_location')}")
                    else:
                        st.write("No misplaced products.")

                    st.subheader("Missing Products:")
                    if analysis_data.get("missing_products"):
                        for p in analysis_data["missing_products"]:
                            st.write(f"- {p.get('name')}: Expected at {p.get('expected_location')}")
                    else:
                        st.write("No missing products.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling planogram API: {e} - {response.text if 'response' in locals() else 'No response'}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("----")
st.sidebar.info(
    "This is a demo application for Planogram Detection. "
    "Not all features are fully implemented."
) 