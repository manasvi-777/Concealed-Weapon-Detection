import os
import shutil
import random



# Your existing variables
images_dir = "data/processed/images/all"
labels_dir = "data/processed/labels/all"
train_img_dir = "data/processed/images/train"
val_img_dir = "data/processed/images/val"
train_lbl_dir = "data/processed/labels/train"
val_lbl_dir = "data/processed/labels/val"

# Create directories if they don't exist
for directory in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(directory, exist_ok=True)

# Get all image files
images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# Process training images
for img in train_images:
    lbl = img.replace('.jpg', '.txt')
    src_img_path = os.path.join(images_dir, img)
    src_lbl_path = os.path.join(labels_dir, lbl)
    
    # Copy image
    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, os.path.join(train_img_dir, img))
    else:
        print(f"Warning: Image not found: {src_img_path}")
    
    # Copy label if it exists
    if os.path.exists(src_lbl_path):
        shutil.copy(src_lbl_path, os.path.join(train_lbl_dir, lbl))
    else:
        print(f"Warning: Label not found: {src_lbl_path}")

# Repeat for validation images
for img in val_images:
    lbl = img.replace('.jpg', '.txt')
    src_img_path = os.path.join(images_dir, img)
    src_lbl_path = os.path.join(labels_dir, lbl)
    
    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, os.path.join(val_img_dir, img))
    else:
        print(f"Warning: Image not found: {src_img_path}")
    
    if os.path.exists(src_lbl_path):
        shutil.copy(src_lbl_path, os.path.join(val_lbl_dir, lbl))
    else:
        print(f"Warning: Label not found: {src_lbl_path}")
