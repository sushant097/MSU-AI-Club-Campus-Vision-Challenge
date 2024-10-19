
# Campus Vision Challenge Dataset

## Overview
The Campus Vision Challenge is a multi-class image classification task where participants develop machine learning models to classify images of 10 different buildings on the university campus. The goal is to predict the building's class (one of the 10 classes) and the probability for each class based on an input image.

### What is Multi-Class Classification?
In a multi-class classification problem, the task is to classify an input image into one of several possible categories. Here, the 10 classes correspond to different building names on campus. The model should output a probability distribution across all classes, and the class with the highest probability is considered the prediction.

### List of Building Classes:
1. Butler Hall
2. Carpenter Hall
3. Lee Hall
4. McCain Hall
5. McCool Hall
6. Old Main
7. Simrall Hall
8. Student Union
9. Swalm Hall
10. Walker Hall

---

## Dataset Characteristics

### Image Distribution
The dataset consists of images for each building as summarized below:

- **Butler Hall**: 1167 images
- **Carpenter Hall**: 1198 images
- **Lee Hall**: 1261 images
- **McCain Hall**: 1277 images
- **McCool Hall**: 1354 images
- **Old Main**: 1362 images
- **Simrall Hall**: 1190 images
- **Student Union**: 1337 images
- **Swalm Hall**: 1361 images
- **Walker Hall**: 1260 images

The dataset is relatively balanced, with similar numbers of images for each building. Data augmentation can still be helpful to improve model robustness.

### Normalization Statistics
To enhance model performance, it is recommended to normalize the dataset using the following statistics calculated across all images:

- **Mean (RGB)**: `[0.4608, 0.4610, 0.4559]`
- **Standard Deviation (RGB)**: `[0.2312, 0.2275, 0.2638]`

### How to Normalize
To use these statistics for normalization in PyTorch:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4608, 0.4610, 0.4559], std=[0.2312, 0.2275, 0.2638])
])
```

---

## Data Augmentation Techniques

Data augmentation can help improve model generalization by artificially increasing the diversity of the training set. Here are some techniques and examples using the **Albumentations** library, which is popular for image augmentations:

### 1. **Random Horizontal and Vertical Flips**
   - Flipping images horizontally or vertically to make the model invariant to changes in orientation.
   ```python
   import albumentations as A

   transform = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
   ])
   ```

### 2. **Rotation**
   - Rotate the images by a random angle within a specified range.
   ```python
   transform = A.Compose([
       A.Rotate(limit=30, p=0.7),  # Randomly rotate images within ±30 degrees
   ])
   ```

### 3. **Random Cropping and Resizing**
   - Randomly crop a portion of the image and resize it back to the original dimensions to simulate zoom effects.
   ```python
   transform = A.Compose([
       A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0), p=0.8),
   ])
   ```

### 4. **Color Jittering**
   - Randomly change the brightness, contrast, saturation, and hue of the images.
   ```python
   transform = A.Compose([
       A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
   ])
   ```

### 5. **Gaussian Noise and Blur**
   - Add noise or blur to the images to make the model more robust to low-quality inputs.
   ```python
   transform = A.Compose([
       A.GaussianBlur(blur_limit=(3, 7), p=0.3),
       A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
   ])
   ```

### 6. **Cutout / Coarse Dropout**
   - Randomly mask out square regions of the input image to simulate occlusion.
   ```python
   transform = A.Compose([
       A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
   ])
   ```

### 7. **Mixing Multiple Techniques**
   - Use a combination of augmentations for a more diversified dataset.
   ```python
   transform = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0), p=0.8),
       A.Rotate(limit=30, p=0.7),
       A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
       A.GaussianBlur(blur_limit=(3, 7), p=0.3),
       A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
   ])
   ```

---

## Best Practices for Model Training

### 1. **Use Transfer Learning**
   - Start with a pre-trained model (e.g., ResNet, EfficientNet) and fine-tune it on the dataset. This can save training time and improve accuracy.

### 2. **Apply Data Augmentation**
   - Even though the dataset is balanced, data augmentation can still help improve generalization and prevent overfitting.

### 3. **Monitor Overfitting**
   - Use techniques like early stopping, dropout layers, and regularization to mitigate overfitting. Track the training and validation loss to ensure that the model generalizes well.

### 4. **Learning Rate Scheduling**
   - Use a learning rate scheduler (e.g., ReduceLROnPlateau) to adjust the learning rate during training based on the validation performance.

### 5. **Experiment with Batch Size**
   - Start with a smaller batch size if limited by memory, then gradually increase it for faster convergence. A batch size of 32 is a good starting point.

### 6. **Use Weight Decay**
   - Adding weight decay (L2 regularization) can help reduce overfitting by penalizing large weights.

### 7. **Perform Cross-Validation**
   - Perform k-fold cross-validation to evaluate the model's performance on different splits of the data.

### 8. **Fine-Tune All Layers**
   - After training the final layers of a pre-trained model, unfreeze all layers and fine-tune the entire network with a smaller learning rate.

---

## Using Any Model/Resources

Participants can choose any model architecture or open-source resources to train on this dataset. However, ensure that the final model does not require more than 24 GB of VRAM during testing with a batch size of 1.

---

## Image Resizing for Testing

Ensure that your testing script includes code to resize any input image to 1024x1024 before feeding it to the model for prediction.

Example using `PIL`:
```python
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((1024, 1024))
    return image
```

---

## Common Queries

### 1. **Is this a multi-class classification problem?**
   Yes, the goal is to classify each image into one of the 10 building categories.

### 2. **Can we use any open-source models?**
   Yes, you are free to use any open-source models. Ensure your model can run within 24 GB of VRAM during testing.

### 3. **What should the submission include?**
   - A detailed `README` explaining your approach.
   - A working testing script and a model download link.
   - A `requirements.txt` listing the dependencies.

### 4. **Should we split the dataset into train and test sets?**
   Yes, it's important to split the dataset to avoid data leakage during training.

### 5. **How will the model be evaluated?**
   The model will be tested once during the final evaluation. Make sure your `README` and testing script are comprehensive.

### 6. **Where can I find more information about this challenge?**
   Visit [here](https://github.com/sushant097/MSU-AI-Club-Campus-Vision-Challenge/blob/master/ProjectCompetitionDetails.md)

---

## Dataset Summary
============================================================
- **Butler Hall**: 1167 images
- **Carpenter Hall**: 1198 images
- **Lee Hall**: 1261 images
- **McCain Hall**: 1277 images
- **McCool Hall**: 1354 images
- **Old Main**: 1362 images
- **Simrall Hall**: 1190 images
- **Student Union**: 1337 images
- **Swalm Hall**: 1361 images
- **Walker Hall**: 1260 images

Normalization Statistics for the Dataset:
- **Mean (RGB)**: `[0.4608, 0.4610, 0.4559]`
- **Standard Deviation (RGB)**: `[0.2312, 0.2275, 0.2638]`

### How to Normalize:
To use these statistics for normalization in PyTorch:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4608, 0.4610, 0.4559], std=[0.2312, 0.2275, 0.2638])
])
```
---

Here's a `PyTorch` code snippet to load the resized dataset from the `CampusVisionChallengeFinal` directory, apply normalization, and prepare it for model training. The code includes detailed comments explaining each step.

```python
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to the merged dataset
data_dir = 'CampusVisionChallengeFinal'

# Normalization values obtained from the dataset (calculated earlier)
# Replace these values with your calculated mean and std from log.txt
mean = [0.485, 0.456, 0.406]  # Example values, replace with actual values
std = [0.229, 0.224, 0.225]   # Example values, replace with actual values

# Define the transformation steps for the dataset
# 1. Resize: Since images are already resized to 1024x1024, we skip this step here
# 2. ToTensor: Converts the image to a PyTorch tensor
# 3. Normalize: Normalizes the image using the provided mean and std values
transform = transforms.Compose([
    transforms.ToTensor(),                          # Convert image to PyTorch tensor
    transforms.Normalize(mean=mean, std=std)        # Normalize using mean and std
])

# Load the dataset
# ImageFolder expects the dataset directory to have subfolders for each class (building names)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # Use 80% of the data for training
val_size = len(dataset) - train_size  # Use the remaining 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
# Batch size is set to 32; you can adjust it based on your hardware capabilities
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Check the dataset details
print(f'Total images in the dataset: {len(dataset)}')
print(f'Number of training images: {len(train_dataset)}')
print(f'Number of validation images: {len(val_dataset)}')

# Example of how to iterate through the DataLoader
# This loop fetches batches of images and labels
for images, labels in train_loader:
    # images: A batch of images (shape: [batch_size, 3, 1024, 1024])
    # labels: Corresponding labels for the images
    print(f'Batch of images shape: {images.shape}')
    print(f'Batch of labels shape: {labels.shape}')
    # Code to Train model
    # ..
    # ..
    pass
```

### Explanation of Each Step:
1. **Path to Dataset**:
   - `data_dir` points to the `MergeDataset` directory containing the resized images, organized in subfolders representing each building (class).
2. **Normalization Values**:
   - Update `mean` and `std` with the actual values calculated in your dataset's `log.txt`.
3. **Transformation Steps (`transforms.Compose`)**:
   - **`ToTensor()`**: Converts PIL images to PyTorch tensors and scales pixel values to `[0, 1]`.
   - **`Normalize()`**: Normalizes the tensor using the provided mean and std values for each RGB channel.
4. **Loading the Dataset (`datasets.ImageFolder`)**:
   - Loads the images and automatically assigns labels based on subfolder names.
5. **Splitting the Dataset**:
   - Uses an 80-20 split to create training and validation sets. **I would suggest to do in advance offline and read from training and validaion directory itself.**
6. **DataLoaders**:
   - `DataLoader` provides an easy way to iterate through the dataset in mini-batches.
   - The `batch_size` can be adjusted based on your hardware.
   - `num_workers` is set to 4 for parallel data loading. Adjust it based on your system capabilities.
7. **Dataset Statistics and Batch Example**:
   - Prints the total number of images, training images, and validation images.
   - Demonstrates how to iterate through the DataLoader to fetch batches of images and labels.



## Copyright
This dataset is provided by the CSE AI Club. It cannot be distributed or shared without the permission of Mississippi State University and the CSE AI Club.

