import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# Import model structure and decoding function from your original file
from LicensePlate import CRNN, decode

class PlateRecognizer:
    def __init__(self, weight_path="crnn_weights.pth"):
        """
        Initialize the recognizer and load model & weights
        """
        print("Loading model architecture...")
        self.model = CRNN()
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights file not found {weight_path}! Please run LicensePlate.py to train and save the model first.")
            
        print("Loading trained weights...")
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print("Model ready!\n" + "-"*30)

    def predict(self, image_path):
        """
        Input a single image path and return the recognition result
        """
        if not os.path.exists(image_path):
            return f"Error: Image not found {image_path}"

        try:
            raw_image = Image.open(image_path).convert('L')
            image_tensor = self.transform(raw_image).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(image_tensor)
                result = decode(preds[:, 0, :])
                
            return result

        except Exception as e:
            return f"Recognition error: {str(e)}"

# ==========================================
# Actual Usage Demo (Batch Recognition on Folder)
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate the recognizer 
    try:
        recognizer = PlateRecognizer(weight_path="crnn_weights.pth")
    except Exception as e:
        print(e)
        exit(1)

    # 2. Specify the absolute path of the folder for batch recognition
    #  Please replace with your actual test image folder path 
    test_folder ="images3"
    
    print("-" * 55)
    
    if not os.path.exists(test_folder):
        print(f"Error: Test folder not found {test_folder}")
    else:
        print(f"Scanning folder: {test_folder}\n")
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        count = 0
        
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(valid_extensions):
                count += 1
                img_path = os.path.join(test_folder, filename)
                
                plate_number = recognizer.predict(img_path)
                
                print(f"Image: {filename:<15} ->  Result: 【 {plate_number} 】")
        
        if count == 0:
            print("No supported image files found in the folder (.jpg, .png, .bmp)!")
        else:
            print(f"\nBatch recognition completed! Processed {count} images in total.")
    print("-" * 55)
