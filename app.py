import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("brain_tumor_model.keras")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image")

uploaded_file = st.file_uploader("Choose MRI image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128,128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

# If binary classification with sigmoid
if prediction.shape[1] == 1:
    prob = prediction[0][0]
    
    if prob > 0.5:
        confidence = prob * 100
        st.error(f"🚨 Tumor Detected")
    else:
        confidence = (1 - prob) * 100
        st.success(f"✅ No Tumor Detected")

# If multi-class with softmax
else:
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    if class_index == 0:
        st.error("🚨 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

# Show confidence bar
st.write(f"### Confidence: {confidence:.2f}%")
st.progress(int(confidence))