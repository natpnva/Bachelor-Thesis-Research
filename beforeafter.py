import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
import pandas as pd

PARQUET_PATH = "/Users/nataliapaunova/MSP/BTR/BTR code/files/gzd5_matched.parquet"
IMAGE_ROOT = "/Users/nataliapaunova/MSP/BTR/BTR code/images_flat" # root directory that relative image paths are relative to
IMAGE_COL = "image_path" # column name containing relative image paths

# Pick any image from your dataframe
df = pd.read_parquet(PARQUET_PATH)
df[IMAGE_COL] = df[IMAGE_COL].apply(lambda x: os.path.basename(x))
img_path = os.path.join(IMAGE_ROOT, df[IMAGE_COL].iloc[0])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(180),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # RGB → 1 channel, using luminance weights (0.299, 0.587, 0.114) — torchvision does this by default
    transforms.CenterCrop(180),   # crop 180×180 from center point
    transforms.Resize((128, 128)),  # downscale
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(), # converts [0, 255] → [0, 1] automatically  
    #transforms.Normalize(mean=[0.5], std=[0.25])   #accuracy goes down by 15% when I add this, maybe because the images are already pretty normalized since they're mostly black with some bright spots, so scaling and shifting messes with that? would need to experiment more to be sure    
])

# --- WITHOUT transforms ---
original = Image.open(img_path)

# --- WITH transforms ---
transformed = train_transform(original)
# ToTensor gives shape (C, H, W) with values in [0,1]
# matplotlib needs (H, W, C) or (H, W) for grayscale
transformed_np = transformed.squeeze().numpy()  # squeeze removes the channel dim since it's grayscale → (H, W)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(original, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(transformed_np, cmap='gray')
axes[1].set_title("Transformed")
axes[1].axis('off')

plt.tight_layout()
plt.show()