import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Visual Question Answering")
st.write("Upload an image and ask a question about it.")

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    processor = ViltProcessor.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )

    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


try:
    with st.spinner("Loading AI Model... Please wait."):
        processor, model = load_model()

except Exception as e:
    st.error(f"Model loading failed:\n\n{e}")
    st.stop()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        question = st.text_input(
            "Ask a Question",
            placeholder="Example: What color is the car?"
        )

        if st.button("Get Answer"):

            if question.strip() == "":
                st.warning("Please enter a question.")

            else:

                with st.spinner("Analyzing Image..."):

                    encoding = processor(
                        image,
                        question,
                        return_tensors="pt"
                    )

                    encoding = {
                        k: v.to(DEVICE)
                        for k, v in encoding.items()
                    }

                    with torch.no_grad():

                        outputs = model(**encoding)

                    predicted_idx = outputs.logits.argmax(-1).item()

                    answer = model.config.id2label[predicted_idx]

                st.success(f"Answer: **{answer}**")

    except Exception as e:

        st.error(f"Error processing image:\n\n{e}")


st.markdown("---")
st.caption("Built using Streamlit + Hugging Face ViLT")
