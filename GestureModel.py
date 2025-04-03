import torch
import torch.nn as nn
import numpy as np
import CNN

class GestureModel:
    def __init__(self, model_path):
        """Initialize the model by loading the pre-trained weights."""
        self.model = CNN.CNNModel()  # Create model instance
        self.model.load_state_dict(torch.load(model_path))  # Load saved weights
        self.model.eval()  # Set model to evaluation mode
    
    def predict(self, image):
        """Make a prediction on the given image."""
        # Ensure the image is in the correct shape (batch_size, channels, height, width)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image)
        
        return output.numpy()  # Convert output to NumPy array for consistency

