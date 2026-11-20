
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Plant Leaf Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Leaf Disease Detector")
st.write(
    "Upload a clear image of a plant leaf to detect "
    "possible diseases using Transfer Learning."
)

# -------------------------------
# 2. LOAD TRAINED MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("leaf_disease_model_v3.keras")

try:
    model = load_model()

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

except Exception as e:
    st.error("Unable to load the trained model or class labels.")
    st.error(str(e))
    st.stop()

# -------------------------------
# 3. CLEAN CLASS NAMES
# -------------------------------

def clean(name):
    crop, _, disease = name.partition("___")

    crop = crop.replace("_", " ").replace("(", "").replace(")", "").strip()
    disease = disease.replace("_", " ").strip() or "unknown"

    return crop, disease


crops = sorted(set(clean(c)[0] for c in class_names))
SUPPORTED = ", ".join(crops)

# -------------------------------
# 4. WARNING THRESHOLDS
# -------------------------------

HIGH = 0.85
LOW = 0.60
MARGIN = 0.25

# -------------------------------
# 5. IMAGE UPLOAD
# -------------------------------

st.subheader("Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# 6. PREDICTION
# -------------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("Detect Disease", type="primary"):

        with st.spinner("Analyzing your leaf image..."):

            # Preprocessing: same as original Colab notebook
            processed_img = img.resize((224, 224))

            arr = np.expand_dims(
                np.array(processed_img, dtype="float32"),
                axis=0
            )

            probs = model.predict(arr, verbose=0)[0]

            order = np.argsort(probs)[::-1]

            top = float(probs[order[0]])
            second = float(probs[order[1]])

            crop, disease = clean(class_names[int(order[0])])

            margin = top - second

        st.divider()
        st.subheader("Detection Result")

        # -------------------------------
        # 7. SMART WARNING SYSTEM
        # -------------------------------

        if top < LOW:

            st.error("❌ Cannot recognize this leaf")

            st.write(
                f"Confidence is only **{top*100:.0f}%**."
            )

            st.write(
                "This may be an unsupported plant or an unclear image."
            )

            st.info(
                "Try uploading one clear leaf image in daylight "
                "with a plain background."
            )

            st.write("**Supported crops:**", SUPPORTED)

        elif top < HIGH or margin < MARGIN:

            st.warning("⚠️ Not sure about this result")

            st.write(
                f"Best guess: **{crop} - {disease}**"
            )

            st.write(
                f"Confidence: **{top*100:.0f}%**"
            )

            st.info(
                "The model is not sufficiently certain. "
                "Try a clearer image or consult an agriculture expert."
            )

            st.write("**Supported crops:**", SUPPORTED)

        else:

            if disease.lower() == "healthy":

                st.success(f"✅ Likely healthy {crop} leaf")

            else:

                st.success(f"✅ Likely: {crop} - {disease}")

            st.write(
                f"Confidence: **{top*100:.0f}%**"
            )

        # -------------------------------
        # 8. TOP 3 PREDICTIONS
        # -------------------------------

        st.divider()
        st.subheader("Top 3 Predictions")

        for i in order[:3]:

            label = " - ".join(
                clean(class_names[int(i)])
            )

            confidence = float(probs[int(i)]) * 100

            st.write(f"**{label}**")

            st.progress(float(probs[int(i)]))

            st.caption(f"{confidence:.2f}% confidence")

        # -------------------------------
        # 9. DISCLAIMER
        # -------------------------------

        st.divider()

        st.caption(
            "This is a student project model. It only recognizes "
            "the crops it was trained on and may produce incorrect "
            "predictions. Confirm results with an agriculture expert "
            "before treating plants."
        )

else:

    st.info("Please upload a leaf image to begin detection.")

    st.write("**Supported crops:**", SUPPORTED)
