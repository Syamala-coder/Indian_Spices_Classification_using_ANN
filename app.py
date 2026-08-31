# ============================================================
# INDIAN SPICE CLASSIFICATION - STREAMLIT APP
# ============================================================

import os

import streamlit as st
import numpy as np
import joblib

from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Spice Classifier",
    page_icon="🌶️",
    layout="centered"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Hugging Face model repository
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Syamu-1207/indian-spices-ann"

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="spice_model_augmented.keras"
)

SCALER_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="scaler_augmented.pkl"
)

ENCODER_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="label_encoder.pkl"
)

IMG_SIZE = (64, 64)


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title(
    "🌶️ Indian Spice Classification"
)

st.write(
    "Upload an image or use the live webcam "
    "to identify an Indian spice using the "
    "trained ANN model."
)

st.info(
    "Supported classes: Bay Leaf, Caraway seeds, "
    "Cloves, Mace, Stone Flowers"
)


# ============================================================
# 4. MODEL FILES
# ============================================================

# The trained model, scaler and label encoder are downloaded
# automatically from the Hugging Face repository above.

# ============================================================
# 5. LOAD MODEL, SCALER AND LABEL ENCODER
# ============================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    return (
        model,
        scaler,
        label_encoder
    )


try:

    model, scaler, label_encoder = (
        load_artifacts()
    )

except Exception as e:

    st.error(
        "❌ Error while loading the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Resize to 64 x 64
    # --------------------------------------------------------

    image = image.resize(
        IMG_SIZE
    )


    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Normalize pixel values
    # 0-255 -> 0-1
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )


    # --------------------------------------------------------
    # Flatten image
    # 64 x 64 x 3 = 12288
    # --------------------------------------------------------

    image_flat = image_array.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Apply trained StandardScaler
    # --------------------------------------------------------

    image_scaled = scaler.transform(
        image_flat
    )


    return image_scaled


# ============================================================
# 7. CHOOSE IMAGE INPUT
# ============================================================

st.subheader(
    "📷 Choose Image Input"
)

input_method = st.radio(
    "Select an option:",
    [
        "📁 Upload Image",
        "📷 Live Web Camera"
    ],
    horizontal=True
)


# ============================================================
# 8. IMAGE INPUT
# ============================================================

image_file = None


# ============================================================
# OPTION 1: UPLOAD IMAGE
# ============================================================

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


# ============================================================
# OPTION 2: LIVE WEBCAM
# ============================================================

else:

    st.write(
        "Allow camera access in your browser "
        "and take a picture of the spice."
    )

    image_file = st.camera_input(
        "Take a picture"
    )


# ============================================================
# 9. DISPLAY IMAGE
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


    st.subheader(
        "🖼️ Selected Image"
    )

    st.image(
        original_image,
        caption="Image to classify",
        use_container_width=True
    )


    # ========================================================
    # 10. PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔍 Predict Spice",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # PREPROCESS IMAGE
            # ------------------------------------------------

            image_scaled = preprocess_image(
                original_image
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probabilities = model.predict(
                image_scaled,
                verbose=0
            )


            # ------------------------------------------------
            # FIND PREDICTED CLASS
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(
                    probabilities[0]
                )
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                probabilities[0][
                    predicted_index
                ]
            )


            # ------------------------------------------------
            # DECODE CLASS
            # ------------------------------------------------

            predicted_class = (
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )


            # =================================================
            # 11. DISPLAY PREDICTION
            # =================================================

            st.subheader(
                "🌿 Prediction Result"
            )

            st.success(
                f"Predicted Spice: "
                f"**{predicted_class}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


            # =================================================
            # 12. ALL CLASS PROBABILITIES
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


            # Sort from highest to lowest probability

            class_probabilities.sort(
                key=lambda item: item[1],
                reverse=True
            )


            # Display every class

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
# 13. MODEL INFORMATION
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
# 14. MODEL SOURCE INFORMATION
# ============================================================

with st.expander(
    "🔧 Model Source Information"
):

    st.write(
        f"**Hugging Face repository:** `{HF_REPO_ID}`"
    )

    st.write(
        "**Model:** `spice_model_augmented.keras`"
    )

    st.write(
        "**Scaler:** `scaler_augmented.pkl`"
    )

    st.write(
        "**Label encoder:** `label_encoder.pkl`"
    )

# ============================================================
# END OF APP
# ============================================================
