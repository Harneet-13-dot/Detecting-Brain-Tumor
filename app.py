import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("brain_tumor_model.keras")

st.title("🧠 Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB").resize((128,128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Since your model uses softmax with 2 outputs
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    if class_index == 0:
        st.error(f"🚨 Tumor Detected ({confidence:.2f}%)")
    else:
        st.success(f"✅ No Tumor Detected ({confidence:.2f}%)")
