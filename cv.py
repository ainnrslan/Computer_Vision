import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Setup
st.set_page_config(page_title="CPU Image Classifier", layout="wide")
st.title("🚀 Enhanced Image Classification with ResNet18")

# Constants
device = torch.device("cpu")

# Cached model loading
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device).eval()
    return model

# Load model
with st.spinner("Loading model..."):
    model = load_model()
    preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of predictions", 1, 10, 5)
    show_details = st.checkbox("Show detailed analysis", value=False)

# Main content
uploaded_file = st.file_uploader(
    "📁 Upload an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image for classification"
)

if uploaded_file is not None:
    try:
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Get predictions
        with st.spinner("Analyzing image..."):
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output[0], dim=0)
                top_prob, top_indices = torch.topk(probabilities, top_k)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader(f"Top-{top_k} Predictions")
            data = []
            for i in range(top_k):
                label = models.ResNet18_Weights.DEFAULT.meta["categories"][top_indices[i]]
                prob = top_prob[i].item() * 100
                
                # Color code confidence
                color = "green" if prob > 70 else "orange" if prob > 30 else "red"
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>{label}: {prob:.2f}%</span>",
                    unsafe_allow_html=True
                )
                data.append({"Class": label, "Probability": prob})
            
            # Visualization
            df = pd.DataFrame(data)
            st.bar_chart(df.set_index("Class"))
        
        # Detailed analysis
        if show_details:
            with st.expander("📊 Detailed Analysis"):
                # Show probability distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(df["Class"], df["Probability"])
                ax.set_xlabel("Probability (%)")
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)
                
                # Show raw tensor info
                st.write("**Image Tensor Statistics:**")
                st.json({
                    "shape": list(input_tensor.shape),
                    "mean": float(input_tensor.mean().item()),
                    "std": float(input_tensor.std().item()),
                    "range": [float(input_tensor.min().item()), 
                             float(input_tensor.max().item())]
                })
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Note:** This app runs on CPU only. Inference might be slower than GPU.")