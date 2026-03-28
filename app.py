import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

st.set_page_config(page_title="Glaucoma Detection Dashboard", layout="wide")

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_model():
    # Ensure your model file is in the same directory
    return tf.keras.models.load_model("glaucoma_model.h5")

def preprocess_image(image):
    # 1. Ensure image is in RGB
    image = image.convert("RGB")
    
    # 2. --- NEW: CENTER CROP ---
    # This cuts off the black camera borders (the red ring)
    # so the AI focuses on the retina and optic disc.
    width, height = image.size
    crop_size = min(width, height) * 0.80  # Keep only the center 80%
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    # ---------------------------

    # 3. Resize to your model's input size
    image = image.resize((224, 224))
    
    # 4. Normalize and add batch dimension
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def colored_progress_bar(label, value, color):
    st.write(label)
    st.markdown(
        f"""
        <div style="background-color:#e0e0e0; border-radius:5px; width:100%; height:20px;">
            <div style="background-color:{color}; width:{value*100}%; height:100%; border-radius:5px;"></div>
        </div>
        <p style='font-weight:bold;'>{value*100:.2f}%</p>
        """,
        unsafe_allow_html=True
    )

def make_gradcam_heatmap(img_array, gradcam_model, prediction_value):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_array)
        
        # If prediction is low (< 0.5), it's 'Normal'. 
        # We target (1 - predictions) to see what makes it 'Normal'.
        if prediction_value < 0.5:
            score = 1.0 - predictions[:, 0]
        else:
            score = predictions[:, 0]

    # Calculate gradients for the specific 'winning' class
    grads = tape.gradient(score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to remove the background noise/rings
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize
    if tf.math.reduce_max(heatmap) != 0:
        heatmap /= tf.math.reduce_max(heatmap)
        
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image with red-hot coloring and transparent background.
    """
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    
    # Create red-hot colormap manually (transparent background)
    heatmap_rgb = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
    heatmap_rgb[..., 0] = (heatmap * 255).astype(np.uint8)  # Red channel only
    
    # Convert original image to numpy array
    img_np = np.array(img.convert("RGB"))
    
    # Blend: alpha = intensity of heatmap
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)
    
    return Image.fromarray(overlay)

# ----------------- App Sidebar -----------------
st.sidebar.header("Settings")
model = load_model()
heatmap_opacity = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.4, 0.05)

with st.expander("🔍 View Model Architecture"):
    for i, layer in enumerate(model.layers):
        try:
            # Try to get output_shape, if not, use the output property
            shape = layer.output_shape
        except AttributeError:
            shape = layer.output.shape
            
        st.write(f"Index {i} | Layer: **{layer.name}** | Shape: {shape}")

# ----------------- Main App -----------------
st.title("👁️ Glaucoma Detection Dashboard")
st.markdown("### Retinal Fundus Image Screening")
st.write("Upload a retinal image to predict Glaucoma and visualize decision regions.")

uploaded_file = st.file_uploader("Upload Retinal Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    
    # Perform Prediction
    prediction_raw = model.predict(processed_image)[0][0]
    glaucoma_prob = float(prediction_raw)
    normal_prob = float(1 - glaucoma_prob)
    label = "Glaucoma" if glaucoma_prob > 0.5 else "Normal"

    # Display Results in Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis")
        if label == "Glaucoma":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
        
        colored_progress_bar("Glaucoma Probability", glaucoma_prob, "#d9534f")
        colored_progress_bar("Normal Probability", normal_prob, "#5cb85c")

    # ----------------- Grad-CAM Engine -----------------
    # 1. Identify Target Conv Layer
    conv_layers = [l.name for l in model.layers if "conv" in l.name.lower() or isinstance(l, tf.keras.layers.Conv2D)]
    
    if not conv_layers:
        st.warning("No convolutional layers found for visualization.")
    else:
        last_conv_layer_name = conv_layers[-1]
        
        # 2. Re-thread Sequential model to Functional for Gradient Tracking
        img_input = tf.keras.Input(shape=(224, 224, 3))
        curr_output = img_input
        target_layer_output = None
        
        for layer in model.layers:
            curr_output = layer(curr_output)
            if layer.name == last_conv_layer_name:
                target_layer_output = curr_output

        gradcam_model = Model(inputs=img_input, outputs=[target_layer_output, curr_output])

        # 3. Generate Heatmap
        heatmap = make_gradcam_heatmap(processed_image, gradcam_model, glaucoma_prob)
        gradcam_result = overlay_heatmap_on_image(image, heatmap, alpha=heatmap_opacity)

        # 4. Display Visualization
        st.divider()
        st.subheader("Grad-CAM Interpretability")
        st.image(gradcam_result, caption=f"Focusing on: {last_conv_layer_name}", use_container_width=True)
        st.info("💡 **Clinical Note:** Red areas indicate where the model detected features relevant to the prediction.")

        # ----------------- Export Report -----------------
        buf = io.BytesIO()
        gradcam_result.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Heatmap Report",
            data=byte_im,
            file_name=f"glaucoma_report_{label.lower()}.jpg",
            mime="image/jpeg"
        )

st.divider()
st.caption("Disclaimer: This tool is for educational purposes only and should not replace professional medical diagnosis.")