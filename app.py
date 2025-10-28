import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
from skimage.segmentation import mark_boundaries
from lime import lime_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
TARGET_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    return transform(image_pil).unsqueeze(0)

def denormalize_for_display(tensor):
    img_tensor = tensor.clone().squeeze(0).cpu().numpy().transpose((1, 2, 0))
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    img_to_show = std * img_tensor + mean
    img_to_show = np.clip(img_to_show, 0, 1)
    return img_to_show

@st.cache_resource
def load_model(model_path):
    try:
        model = MyModel(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_grad_cam(model, input_tensor, original_image_np):
    target_layer = model.model[3]
    cam = GradCAM(model=model, target_layers=[target_layer])
    with torch.no_grad():
        output = model(input_tensor.to(device))
        pred_idx = output.argmax(dim=1).item()
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        original_image_np, 
        grayscale_cam, 
        use_rgb=True, 
        colormap=cv2.COLORMAP_JET
    )
    return visualization

def get_lime_explanation(model, image_pil_resized):
    def predict_for_lime(numpy_images):
        images = torch.from_numpy(numpy_images).permute(0, 3, 1, 2).float().to(device)
        normalizer = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        images = normalizer(images)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    image_np = np.array(image_pil_resized) / 255.0
    explainer = lime_image.LimeImageExplainer()
    with st.spinner("Generating LIME..."):
        explanation = explainer.explain_instance(
            image_np,
            predict_for_lime,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        lime_image_with_boundaries = mark_boundaries(temp, mask)
    return lime_image_with_boundaries

st.title("🧠 Brain Tumor Classification")

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your brain MRI image",
    type=["jpg", "jpeg", "png"],
    help="Limit 200MB per file"
)

st.sidebar.markdown("---")

show_grad_cam = st.sidebar.checkbox("Show Grad-CAM", value=True)
show_lime = st.sidebar.checkbox("Show LIME Explanation", value=True)

with st.sidebar.expander("ℹ️ About this App & Tumor Types"):
    st.write("""
        This application uses a **Convolutional Neural Network (CNN)** to classify brain MRI scans.
        It identifies one of the following conditions:
        
        * **Glioma:** A tumor that originates in the glial (support) cells of the brain or spinal cord.
        * **Meningioma:** A tumor that grows from the meninges, the protective layers covering the brain.
        * **Pituitary Tumor:** An abnormal growth that develops in the pituitary gland at the base of the brain.
        * **No Tumor:** A scan identified as not containing evidence of these tumor types.
    """)

model = load_model("custom_cnn_tumor_model.pth")

image_pil = None
caption = ""

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        caption = "Uploaded MRI Scan"
    except Exception as e:
        st.error(f"Error opening uploaded file: {e}")
        image_pil = None
else:
    st.info("Please upload an image from the sidebar.")

col1, col2 = st.columns(2)

if image_pil and model:
    with col1:
        st.header("Uploaded MRI Scan")
        display_image = image_pil.resize(TARGET_SIZE)
        st.image(display_image, caption=caption, use_container_width=True)
        try:
            input_tensor = preprocess_image(image_pil)
            with torch.no_grad():
                outputs = model(input_tensor.to(device))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx_tensor = torch.max(probabilities, 1)
                predicted_idx = predicted_idx_tensor.item()
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence_percent = confidence.item() * 100
            st.subheader("Prediction")
            if predicted_class == "No Tumor":
                st.success(f"**Prediction:** {predicted_class}")
            else:
                st.error(f"**Prediction:** {predicted_class}")
            st.metric(label="Confidence", value=f"{confidence_percent:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    with col2:
        st.header("Explain Prediction")
        if 'input_tensor' not in locals():
            st.warning("Cannot generate explanations until image is processed.")
        else:
            tab1, tab2 = st.tabs(["Grad-CAM", "LIME"])
            with tab1:
                if show_grad_cam:
                    st.subheader("Grad-CAM")
                    try:
                        denormalized_img_np = denormalize_for_display(input_tensor)
                        grad_cam_img = get_grad_cam(model, input_tensor, denormalized_img_np)
                        st.image(grad_cam_img, caption="Grad-CAM Overlay", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating Grad-CAM: {e}")
                else:
                    st.info("Select 'Show Grad-CAM' in the sidebar to display.")
            with tab2:
                if show_lime:
                    st.subheader("LIME Explanation")
                    try:
                        image_pil_resized = image_pil.resize(TARGET_SIZE)
                        lime_explanation_img = get_lime_explanation(model, image_pil_resized)
                        st.image(lime_explanation_img, caption="LIME Explanation (Top 5 features)", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating LIME: {e}")
                else:
                    st.info("Select 'Show LIME Explanation' in the sidebar to display.")
elif not model:
    st.warning("Model `custom_cnn_tumor_model.pth` could not be loaded. Please check the file and definition.")
