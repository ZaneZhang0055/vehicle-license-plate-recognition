import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from LicensePlate import CRNN, decode

class PlateRecognizer:
    def __init__(self, weight_path="crnn_weights.pth"):
        """
        Initialize the recognizer, load the model and weights
        """
        print(" Loading model architecture...")
        self.model = CRNN()
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f" Cannot find the weight file {weight_path}！Please run first LicensePlate.py Train and save the model。")
            
        print("正在加载训练好的权重...")
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print(" 模型准备就绪！\n" + "-"*30)

    def predict(self, image_path):
        """
        传入单张图片路径，返回识别结果
        """
        if not os.path.exists(image_path):
            return f"错误：找不到图片 {image_path}"

        try:
            raw_image = Image.open(image_path).convert('L')
            image_tensor = self.transform(raw_image).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(image_tensor)
                result = decode(preds[:, 0, :])
                
            return result

        except Exception as e:
            return f"识别出错: {str(e)}"
    # ==========================================
# 实际调用演示 (批量识别文件夹)
# ==========================================
if __name__ == "__main__":
    # 1. 实例化识别器 
    try:
        recognizer = PlateRecognizer(weight_path="crnn_weights.pth")
    except Exception as e:
        print(e)
        exit(1)

    # 2. 指定你要批量识别的文件夹绝对路径
    # 👇👇👇 请替换为你实际的测试图片文件夹路径 👇👇👇
    test_folder = r"D:\CS183\test" 
    
    print("-" * 55)
    
    if not os.path.exists(test_folder):
        print(f" 错误：找不到测试文件夹 {test_folder}")
    else:
        print(f" 正在扫描文件夹: {test_folder}\n")
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        count = 0
        
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(valid_extensions):
                count += 1
                img_path = os.path.join(test_folder, filename)
                
                plate_number = recognizer.predict(img_path)
                
                print(f" 图片: {filename:<15} ->  识别结果: 【 {plate_number} 】")
        
        if count == 0:
            print(" 文件夹里没有找到支持的图片文件(.jpg, .png, .bmp)！")
        else:
            print(f"\n 批量识别完成！共处理了 {count} 张图片。")
    print("-" * 55)
