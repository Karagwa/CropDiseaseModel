import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2

# 🚀 Load model
model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)

# 🏷️ Class labels
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]


st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a **clear image** of a plant leaf to detect diseases (or health status).")


uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((224, 224))
        st.image(image_resized, caption="📷 Uploaded Image", use_container_width=False)

        # Preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array)[0]
        predicted_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_idx]
        confidence = prediction[predicted_idx] * 100

        # Show result
        st.markdown(f"### 🧠 Prediction: `{predicted_label}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        if confidence < 30:
            st.warning("⚠️ I'm not very confident about this. Try using a clearer or closer leaf image.")

        # Top 3 predictions
        st.subheader("🔍 Top 3 Predictions")
        top3_indices = np.argsort(prediction)[-3:][::-1]
        for idx in top3_indices:
            st.write(f"**{class_names[idx]}** — {prediction[idx] * 100:.2f}%")

        st.markdown("---")
        st.caption("📌 This tool is a proof-of-concept model. For real-world agricultural decisions, consult a professional.")

    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")

st.sidebar.header("🛠️ Instructions & Tips")        
st.sidebar.markdown("Instructions:")
st.sidebar.markdown("""
1. Upload a clear image of a plant leaf.
2. The model will predict the disease or health status of the plant.
3. Review the confidence level and top predictions.
""")

st.sidebar.markdown("Tips:")
st.sidebar.markdown("""
- Use high-quality images with good lighting.
- Ensure the leaf is clearly visible and not obscured.
- If the prediction confidence is low, try a different image.
""")
       
st.sidebar.markdown("About")
st.sidebar.markdown("""
This app uses a pre-trained model to classify plant diseases based on leaf images.
It is designed for educational purposes and may not be suitable for professional agricultural use.

For more information, visit the [GitHub repository](https://github.com/Karagwa/Crop-Disease-model).


""")
