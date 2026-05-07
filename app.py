import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import time
import os
from models import SimpleCNN, DeeperCNN, ResNetTransfer, EfficientNetTransfer

st.set_page_config(page_title="🍅 Tomato Disease Diagnosis", layout="centered")
st.title("🍅 Tomato Disease Diagnosis System (4 Models)")
st.markdown("---")

CLASS_NAMES = ['Not_Tomato', 'Tomato_Early_blight', 'Tomato_Healthy', 
               'Tomato_leaf_late_blight', 'Tomato_leaf_yellow_curl_virus', 
               'Tomato_mold_leaf', 'Tomato_septora_leaf_spot']

MODELS_INFO = {
    "SimpleCNN (Scratch)": {"class": SimpleCNN, "weight": "simplecnn_7classes.pth", "img_size": 128},
    "DeeperCNN (Scratch)": {"class": DeeperCNN, "weight": "deepercnn_7classes.pth", "img_size": 128},
    "ResNet50 (Transfer)": {"class": ResNetTransfer, "weight": "resnet50_7classes.pth", "img_size": 224},
    "EfficientNet (Transfer)": {"class": EfficientNetTransfer, "weight": "efficientnet_7classes_fixed.pth", "img_size": 224}
}

available_models = {name: info for name, info in MODELS_INFO.items() if os.path.exists(info["weight"])}
if not available_models:
    st.error("No trained models found! Please train at least one model.")
    st.stop()

@st.cache_resource
def load_model(model_name):
    info = available_models[model_name]
    model = info["class"](num_classes=7)
    model.load_state_dict(torch.load(info["weight"], map_location='cpu'))
    model.eval()
    return model, info["img_size"]

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose AI Model:", list(available_models.keys()))
model, img_size = load_model(selected_model)
st.sidebar.success(f"Loaded {selected_model} (input size {img_size}×{img_size})")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a clear tomato leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Input Image", use_container_width=True)
    
    if st.button("🔍 Diagnose Disease"):
        input_tensor = transform(image).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, idx = torch.max(probs, 0)
        elapsed = (time.time() - start) * 1000
        
        disease = CLASS_NAMES[idx.item()]
        confidence = conf.item() * 100
        
        if disease == 'Not_Tomato':
            st.error(f"**Prediction:** {disease}")
            st.warning("⚠️ This is not a tomato leaf. Please upload a valid leaf image.")
        else:
            st.success(f"**Prediction:** {disease.replace('_', ' ')}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.caption(f"⏱️ Inference time: {elapsed:.2f} ms")
        
        rec = {
            'Tomato_Early_blight': "🔸 Apply Mancozeb or Chlorothalonil.",
            'Tomato_Healthy': "✅ Healthy plant.",
            'Tomato_leaf_late_blight': "⚠️ Use copper fungicide.",
            'Tomato_leaf_yellow_curl_virus': "🦟 Control whiteflies.",
            'Tomato_mold_leaf': "🍄 Reduce humidity.",
            'Tomato_septora_leaf_spot': "🌱 Remove infected leaves."
        }
        if disease in rec:
            st.write(rec[disease])
        
        with st.expander("📊 Detailed confidence per class"):
            for i, name in enumerate(CLASS_NAMES):
                st.write(f"{name}: {probs[i].item()*100:.2f}%")
else:
    st.info("👈 Upload an image to start diagnosis.")