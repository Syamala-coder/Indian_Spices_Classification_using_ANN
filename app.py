# ============================================================
# INDIAN SPICE CLASSIFICATION - STREAMLIT APP
# ============================================================

import streamlit as st
import numpy as np
import joblib

from PIL import Image
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Spice Classifier",
    page_icon="🌶️",
    layout="centered"
)


# ============================================================
# 2. HUGGING FACE REPOSITORY
# ============================================================

HF_REPO_ID = "Syamu-1207/indian-spices-ann"

IMG_SIZE = (64, 64)


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title("🌶️ Indian Spice Classification")

st.write(
    "Upload an image or use the live webcam "
    "to identify an Indian spice using the trained ANN model."
)

st.info(
    "Supported classes: Bay Leaf, Caraway seeds, "
    "Cloves, Mace, Stone Flowers"
)


# ============================================================
# 4. LOAD MODEL, SCALER AND LABEL ENCODER
# ============================================================

@st.cache_resource
def load_artifacts():

    # Download H5 model from Hugging Face
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="spice_model_augmented.h5"
    )

    # Download scaler
    scaler_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="scaler_augmented.pkl"
    )

    # Download label encoder
    encoder_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="label_encoder.pkl"
    )

    # Load model
    model = load_model(
        model_path,
        compile=False
    )

    # Load scaler
    scaler = joblib.load(
        scaler_path
    )

    # Load label encoder
    label_encoder = joblib.load(
        encoder_path
    )

    return model, scaler, label_encoder


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    model, scaler, label_encoder = load_artifacts()

except Exception as e:

    st.error(
        "❌ Error while loading the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Flatten
    image_flat = image_array.reshape(
        1,
        -1
    )

    # Apply trained scaler
    image_scaled = scaler.transform(
        image_flat
    )

    return image_scaled


# ============================================================
# 6. CHOOSE IMAGE INPUT
# ============================================================

st.subheader("📷 Choose Image Input")

input_method = st.radio(
    "Select an option:",
    [
        "📁 Upload Image",
        "📷 Live Web Camera"
    ],
    horizontal=True
)


# ============================================================
# 7. IMAGE INPUT
# ============================================================

image_file = None


if input_method == "📁 Upload Image":

    st.write(
        "Upload a spice image from your computer."
    )

    image_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )


else:

    st.write(
        "Allow camera access in your browser "
        "and take a picture of the spice."
    )

    image_file = st.camera_input(
        "Take a picture"
    )


# ============================================================
# 8. DISPLAY IMAGE
# ============================================================

if image_file is not None:

    try:

        original_image = Image.open(
            image_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            "❌ Could not read the image."
        )

        st.exception(e)

        st.stop()


    st.subheader("🖼️ Selected Image")

    st.image(
        original_image,
        caption="Image to classify",
        use_container_width=True
    )


    # ========================================================
    # 9. PREDICT
    # ========================================================

    if st.button(
        "🔍 Predict Spice",
        type="primary",
        use_container_width=True
    ):

        try:

            # Preprocess
            image_scaled = preprocess_image(
                original_image
            )

            # Prediction
            probabilities = model.predict(
                image_scaled,
                verbose=0
            )

            # Predicted index
            predicted_index = int(
                np.argmax(
                    probabilities[0]
                )
            )

            # Confidence
            confidence = float(
                probabilities[0][
                    predicted_index
                ]
            )

            # Decode class
            predicted_class = (
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )


            # =================================================
            # 10. DISPLAY RESULT
            # =================================================

            st.subheader(
                "🌿 Prediction Result"
            )

            st.success(
                f"Predicted Spice: **{predicted_class}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


            # =================================================
            # 11. ALL PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Class Probabilities"
            )

            class_probabilities = []

            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities[0]
            ):

                class_probabilities.append(
                    (
                        class_name,
                        float(probability)
                    )
                )


            # Highest probability first
            class_probabilities.sort(
                key=lambda item: item[1],
                reverse=True
            )


            for (
                class_name,
                probability
            ) in class_probabilities:

                percentage = (
                    probability * 100
                )

                st.write(
                    f"**{class_name}** — "
                    f"{percentage:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            probability,
                            0.0
                        ),
                        1.0
                    )
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# 12. MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Model Information"
):

    st.write(
        "**Model:** Improved ANN"
    )

    st.write(
        "**Input image:** 64 × 64 × 3 RGB"
    )

    st.write(
        "**Flattened input:** 12,288 features"
    )

    st.write(
        "**Hidden layers:** 256 → 128 → 64"
    )

    st.write(
        "**Hidden activation:** ReLU"
    )

    st.write(
        "**Output activation:** Softmax"
    )

    st.write(
        "**Number of classes:** 5"
    )

    st.write(
        "**Optimizer:** Adam"
    )

    st.write(
        "**Learning rate:** 0.001"
    )

    st.write(
        "**Loss:** Sparse Categorical Crossentropy"
    )

    st.write(
        "**Dropout:** 0.30 → 0.30 → 0.20"
    )

    st.write(
        "**Preprocessing:** "
        "Resize → Normalize → Flatten → StandardScaler"
    )


# ============================================================
# 13. MODEL SOURCE
# ============================================================

with st.expander(
    "🔧 Model Source Information"
):

    st.write(
        f"**Hugging Face repository:** "
        f"`{HF_REPO_ID}`"
    )

    st.write(
        "**Model:** `spice_model_augmented.h5`"
    )

    st.write(
        "**Scaler:** `scaler_augmented.pkl`"
    )

    st.write(
        "**Label encoder:** `label_encoder.pkl`"
    )