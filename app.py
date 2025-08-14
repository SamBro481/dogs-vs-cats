import warnings
warnings.filterwarnings('ignore')

from fastai.vision.all import *
import gradio as gr
from pathlib import Path

# -------- Load the model once at startup --------
MODEL_PATH = Path(__file__).parent / "model.pkl"
learn = load_learner(MODEL_PATH)

# -------- Prediction function --------
def classify_image(img):
    """
    Predict whether an image is a dog or a cat.
    Returns a dictionary suitable for Gradio Label output.
    """
    pred, _, probs = learn.predict(img)
    # Return a dict: {class_name: probability}
    return {str(pred): float(probs.max())}

# -------- Gradio interface --------
image_input = gr.Image(type="pil", label="Upload Image")
label_output = gr.Label(num_top_classes=None, label="Prediction")

intf = gr.Interface(
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
    title="Dogs vs Cats Classifier",
    description="Upload an image to predict whether it's a dog or a cat."
)

# Launch the interface (no inline=False needed on Hugging Face Spaces)
if __name__ == "__main__":
    intf.launch()
