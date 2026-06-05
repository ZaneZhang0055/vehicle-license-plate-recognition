import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

# ==========================================
# 0. Core Path Configuration
# ==========================================
IMAGE_DIR_PATH = r"d:\CS183\新建文件夹 (4)"

# ==========================================
# 1. Full Configuration & Industrial Dictionary
# ==========================================
CHARS = [
    "<Blank>",
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑",
    "沪", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新",
    "A", "B", "C", "D", "E", "F", "G", "H",
    "J", "K", "L", "M", "N", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]
char_to_id = {char: idx for idx, char in enumerate(CHARS)}
id_to_char = {idx: char for idx, char in enumerate(CHARS)}

NUM_CLASSES = len(CHARS)
BLANK_INDEX = 0

# ==========================================
# 2.CRNN = Deep Convolutional Layers + Dropout
# ==========================================
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # --------------------------
        # Correct Key Dimensions!!!
        # --------------------------
        self.rnn = nn.LSTM(input_size=1024, hidden_size=256, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * 2, NUM_CLASSES)
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w)
        conv = conv.permute(2, 0, 1)

        output, _ = self.rnn(conv)
        output = self.dropout(output)
        output = self.fc(output)
        return output

# ==========================================
# 3. Dataset
# ==========================================
class RealDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.data = []

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"🚨 Image folder not found! Path: {img_dir}")

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(valid_extensions):
                text = os.path.splitext(filename)[0]
                self.data.append((filename, text))

        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            
            # 👇👇👇 New Data Augmentation Module 👇👇👇
            # 1. Color Jitter: Random brightness, contrast, saturation (simulate different weather & light)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # 2. Random Slight Rotation: Simulate camera misalignment (within ±3 degrees)
            transforms.RandomRotation(degrees=3),
            # 👆👆👆 New Data Augmentation Module 👆👆👆
            
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, text = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        raw_image = Image.open(img_path).convert('L')
        image_tensor = self.transform(raw_image)
        
        target = [char_to_id[c] for c in text if c in char_to_id]
        
        # --- Safety Check ---
        if len(target) == 0:
            print(f"⚠️ Warning: No valid license plate characters extracted from {img_name}, please check the filename!")
        # ------------------------
        
        return image_tensor, torch.LongTensor(target), len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, targets, target_lengths

# ==========================================
# 4. Decoder
# ==========================================
def decode(preds):
    _, max_indices = torch.max(preds, dim=1)
    char_list = []
    prev_idx = None
    for idx in max_indices:
        idx = idx.item()
        if idx != BLANK_INDEX and idx != prev_idx:
            char_list.append(id_to_char[idx])
        prev_idx = idx
    return "".join(char_list)[:8]
# ==========================================
# 5. Training
# ==========================================
# ==========================================
# 5. Training
# ==========================================
if __name__ == "__main__":
    print("Initializing model and data...")
    model = CRNN()
    dataset = RealDataset(img_dir=IMAGE_DIR_PATH)
    bs = min(32, len(dataset))
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn)

    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(f"\nStart training (Current dataset contains {len(dataset)} images)...")

    for epoch in range(1, 151):
        model.train()
        total_loss = 0
        for images, targets, target_lengths in dataloader:
            optimizer.zero_grad()
            preds = model(images)
            input_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long)
            loss = criterion(F.log_softmax(preds, dim=2), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataloader):.4f}")

    # ===== Save Weights =====
    torch.save(model.state_dict(), "crnn_weights.pth")
    print("Training completed, model weights saved as crnn_weights.pth")
