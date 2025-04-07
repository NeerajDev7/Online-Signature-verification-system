import streamlit as st
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import hog
from PIL import Image
from sklearn.decomposition import PCA
import joblib  # To load PCA model

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    """Load and cache models for faster execution."""
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg16.eval()
    vgg16_extractor = torch.nn.Sequential(*list(vgg16.children())[:-1]).to(device)
    pca = joblib.load("pca_model.pkl")
    classifier = joblib.load("random_forest_model.pkl")
    return vgg16_extractor, pca, classifier

# Load models
vgg16_extractor, pca, classifier = load_models()

# Image transformation for VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define HOG feature length based on training data
HOG_FEATURE_LENGTH = 149283  # Ensure this matches PCA training

def extract_features(image):
    """Extract HOG and VGG16 features from the image."""
    image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract HOG features
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
    
    # Pad or truncate HOG features
    if len(hog_features) < HOG_FEATURE_LENGTH:
        hog_features_padded = np.pad(hog_features, (0, HOG_FEATURE_LENGTH - len(hog_features)), mode='constant')
    else:
        hog_features_padded = hog_features[:HOG_FEATURE_LENGTH]
    
    # Convert to tensor for VGG16
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Extract VGG16 features
    with torch.no_grad():
        vgg16_feature = vgg16_extractor(image_tensor).cpu().numpy().flatten()
    
    # Concatenate features
    features = np.hstack((hog_features_padded, vgg16_feature))
    
    # Debugging output
    print(f"Extracted Features Shape: {features.shape}")  
    
    return features

def predict_signature(image):
    """Predicts the authenticity of the signature."""
    try:
        features = extract_features(image)

        # Ensure PCA input shape is consistent
        features_reduced = pca.transform(features.reshape(1, -1))

        # Predict
        prediction = classifier.predict(features_reduced)[0]

        # Determine if the signature is genuine or forged
        if prediction % 2 == 1:
            result = "Forged Signature"
        else:
            result = "Genuine Signature"
        
        return f"Predicted Class: ({result})"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Signature Verification using HOG + VGG16")
st.write("Upload an image of a signature to verify its authenticity.")

uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Signature", use_column_width=True)

    if st.button("Verify Signature"):
        with st.spinner("Processing... Please wait."):
            result = predict_signature(image)
        st.success(result)
