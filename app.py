import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd
import io
import os
import pickle

# Force CPU mode for all operations (disable CUDA)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_tensor_type('torch.FloatTensor')

# Monkey patch torch.cuda functions at module level to always return False/CPU
# This must be done before any model loading
_original_cuda_is_available = torch.cuda.is_available
_original_cuda_device_count = getattr(torch.cuda, 'device_count', lambda: 0)

def _force_cpu_available():
    return False

torch.cuda.is_available = _force_cpu_available
if hasattr(torch.cuda, 'device_count'):
    torch.cuda.device_count = lambda: 0

# CRITICAL: Patch PyTorch's _validate_device function BEFORE any imports that use it
# This function is called during model loading and checks CUDA availability
import torch.serialization
if hasattr(torch.serialization, '_validate_device'):
    _original_validate_device = torch.serialization._validate_device
    
    def _patched_validate_device(location, backend_name='cpu'):
        # Always return CPU device, bypassing all CUDA checks
        return torch.device('cpu')
    
    torch.serialization._validate_device = _patched_validate_device

# Page configuration
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class names for chest X-ray classification (15 classes)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@st.cache_resource
def load_efficientnet_model():
    """Load EfficientNet model from CPU-converted pickle file"""
    try:
        model_path = "efficientnet_best_model_cpu.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please ensure efficientnet_best_model_cpu.pkl is in the same directory as app.py")
            return None
        
        # Load the CPU-converted model (should load without CUDA issues)
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Set model to evaluation mode and ensure it's on CPU
        model.eval()
        model = model.to('cpu')
        
        return model
    except Exception as e:
        st.error(f"Error loading EfficientNet model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model"""
    try:
        model_path = "yoloV8.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {str(e)}")
        return None

def preprocess_image_for_classification(image):
    """Preprocess image for EfficientNet model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_classification(model, image_tensor, model_name):
    """Run inference for classification models"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Get top predictions
        threshold = 0.5
        predicted_indices = np.where(probabilities > threshold)[0]
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        results = []
        for idx in sorted_indices:
            results.append({
                'class': CLASS_NAMES[idx],
                'probability': float(probabilities[idx]),
                'predicted': idx in predicted_indices
            })
        
        return results
    except Exception as e:
        st.error(f"Error during {model_name} prediction: {str(e)}")
        return None

def predict_yolo(model, image):
    """Run YOLOv8 inference"""
    try:
        # Run inference
        results = model(image, conf=0.25)
        
        # Get the first result (single image)
        result = results[0]
        
        # Plot results on image
        annotated_image = result.plot()
        
        # Convert to PIL Image
        annotated_pil = Image.fromarray(annotated_image)
        
        # Get detection information
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    'class': model.names[cls],
                    'confidence': conf,
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })
        
        return annotated_pil, detections
    except Exception as e:
        st.error(f"Error during YOLOv8 prediction: {str(e)}")
        return None, None

def main():
    # Sidebar
    st.sidebar.title("🏥 Medical Image Analysis")
    st.sidebar.markdown("---")
    
    # Model selection info
    st.sidebar.markdown("### Available Models")
    st.sidebar.markdown("""
    - **EfficientNet**: Multi-label classification
    - **YOLOv8**: Object detection
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image (PNG/JPG)
    2. Select a model tab
    3. View predictions
    """)
    
    # Main content
    st.title("🏥 Medical Image Analysis Dashboard")
    st.markdown("Upload a chest X-ray image to analyze using different deep learning models.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG or JPG image file"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.stop()
        
        # Create tabs for different models
        tab1, tab2 = st.tabs(["🔬 EfficientNet", "🎯 YOLOv8"])
        
        # EfficientNet Tab
        with tab1:
            st.header("EfficientNet Classification")
            st.markdown("Multi-label classification using EfficientNet-B0")
            
            if st.button("Run EfficientNet Prediction", type="primary"):
                with st.spinner("Loading model and running inference..."):
                    model = load_efficientnet_model()
                    if model is not None:
                        # Preprocess image
                        image_tensor = preprocess_image_for_classification(image)
                        
                        # Run prediction
                        results = predict_classification(model, image_tensor, "EfficientNet")
                        
                        if results:
                            st.success("✅ Prediction completed!")
                            
                            # Display top predictions
                            st.subheader("📊 Predictions")
                            
                            # Create columns for better layout
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Top Predictions")
                                for i, result in enumerate(results[:5]):
                                    prob = result['probability']
                                    predicted = "✅" if result['predicted'] else "❌"
                                    color = "green" if result['predicted'] else "gray"
                                    st.markdown(
                                        f"{predicted} **{result['class']}**: {prob:.2%}",
                                        unsafe_allow_html=True
                                    )
                            
                            with col2:
                                st.markdown("### All Classes")
                                # Create a bar chart
                                df = pd.DataFrame(results)
                                df['probability'] = df['probability'] * 100
                                st.bar_chart(df.set_index('class')['probability'])
                            
                            # Show all predictions in expander
                            with st.expander("View All Predictions"):
                                for result in results:
                                    prob = result['probability']
                                    predicted = "✅ Predicted" if result['predicted'] else "❌ Not Predicted"
                                    st.write(f"**{result['class']}**: {prob:.2%} ({predicted})")
        
        # YOLOv8 Tab
        with tab2:
            st.header("YOLOv8 Object Detection")
            st.markdown("Object detection and localization using YOLOv8")
            
            # Confidence threshold slider
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Minimum confidence score for detections"
            )
            
            if st.button("Run YOLOv8 Detection", type="primary"):
                with st.spinner("Loading model and running inference..."):
                    model = load_yolo_model()
                    if model is not None:
                        # Run prediction
                        annotated_image, detections = predict_yolo(model, image)
                        
                        if annotated_image is not None:
                            st.success("✅ Detection completed!")
                            
                            # Display annotated image
                            st.subheader("🎯 Detection Results")
                            st.image(annotated_image, caption="Annotated Image with Detections", use_container_width=True)
                            
                            # Display detection information
                            if detections:
                                st.subheader("📋 Detected Objects")
                                
                                # Filter by confidence threshold
                                filtered_detections = [d for d in detections if d['confidence'] >= conf_threshold]
                                
                                if filtered_detections:
                                    # Create a table
                                    det_df = pd.DataFrame(filtered_detections)
                                    det_df['confidence'] = det_df['confidence'].apply(lambda x: f"{x:.2%}")
                                    st.dataframe(det_df[['class', 'confidence']], use_container_width=True)
                                    
                                    # Show statistics
                                    st.metric("Total Detections", len(filtered_detections))
                                else:
                                    st.info(f"No detections found above confidence threshold of {conf_threshold:.2%}")
                            else:
                                st.info("No objects detected in the image.")
    else:
        st.info("👆 Please upload an image file to get started.")
        
        # Show example usage
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload Image**: Click on the file uploader above and select a chest X-ray image (PNG or JPG format)
        2. **Select Model**: Choose one of the model tabs:
           - **EfficientNet**: For multi-label classification
           - **YOLOv8**: For object detection with bounding boxes
        3. **Run Prediction**: Click the prediction button in the selected tab
        4. **View Results**: Review the predictions and visualizations
        """)

if __name__ == "__main__":
    main()

