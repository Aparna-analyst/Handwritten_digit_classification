
import streamlit as st
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog

# Title
st.title("MNIST Digit Classifier 🎨")
st.write("Upload a handwritten digit image to classify it using HOG + Logistic Regression.")

# Load model and scaler
with open("mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# HOG feature extractor
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_np = np.array(image)
    features = hog(image_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features_scaled = scaler.transform([features])
    return features_scaled

# Upload section
uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    try:
        features = preprocess_image(image)
        prediction = model.predict(features)[0]
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted Digit: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

