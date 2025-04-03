import torch
import torch.nn as nn
import numpy as np
import CNN

class GestureModel:
    def __init__(self, model_path):
        """Initialize the model by loading the pre-trained weights."""
        self.model = CNN.CNNModel()  # Create model instance
        self.model.load_state_dict(torch.load(model_path, weights_only=True))  # Load saved weights
        self.model.eval()  # Set model to evaluation mode
    
    def predict(self, image):
        """Make a prediction on the given image."""
        # Ensure the image is in the correct shape (batch_size, channels, height, width)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image)
        
        return output.numpy()  # Convert output to NumPy array for consistency

if __name__ == "__main__":
    # Example usage
    model_path = 'rotated_gesture_model.pth'  # Path to the pre-trained model weights
    gesture_model = GestureModel(model_path)
    
    # Dummy image for testing (28x28 grayscale image)
    test_image = np.random.rand(28, 28).astype(np.float32)  # Replace with actual image loading code
    
    prediction = gesture_model.predict(test_image)
    print("Prediction:", prediction)