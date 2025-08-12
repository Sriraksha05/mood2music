import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model safely
def load_emotion_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load emotion model: {str(e)}")

# Get input shape safely
def _get_input_shape(model):
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None:
            shape = getattr(model.layers[0], 'input_shape', None)
        if shape is None:
            raise ValueError("Model input shape could not be determined.")
        
        # If batch dimension is present, skip it
        if len(shape) == 4:
            return shape[1], shape[2], shape[3]
        elif len(shape) == 3:
            return shape
        else:
            raise ValueError(f"Unexpected input shape format: {shape}")
    except Exception as e:
        raise RuntimeError(f"Error getting model input shape: {str(e)}")

# Predict emotion from image

def predict_emotion_from_image(model, image_file):
    """
    Predict emotion from uploaded image file
    """
    try:
        # Open image and convert to grayscale
        img = Image.open(image_file).convert('L')  # 'L' = grayscale
        
        # Resize to model's input size (likely 48x48)
        img = img.resize((48, 48))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values (0-255 → 0-1)
        img_array = img_array / 255.0
        
        # Expand dimensions to match (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=-1)  # add channel = 1
        img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
        
        # Predict
        preds = model.predict(img_array)
        
        # Get predicted label
        class_idx = np.argmax(preds[0])
        labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return labels[class_idx]
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Map numeric prediction to emotion label
def _map_label_to_emotion(label_idx):
    emotion_labels = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }
    return emotion_labels.get(label_idx, "unknown")
