# Competition Name:
## **Campus Vision Challenge**

**Organized By: AI CLUB. Department of Computer Science And Engineering.**
<img src="https://github.com/user-attachments/assets/5da125c2-20f8-40fb-865c-9e45e0f8524c" style="height: 200px, width: 200px">


### Objective:
Participants will build an image classification model to predict the name of a university building based on a new image. The model should be trained using images of various university buildings taken from different angles and lighting conditions (not at night).

### Timeline:
- **Announcement & Registration:** 1 week before the competition.
- **Starter Code Release (Officially):** Sep 16 through the email.
- **Q & A:** Will be organized if demanded. All issues should be submitted through Github Issues over the project repository.
- **Solution Submission Deadline:** October 7 midnight. We will circulate the link to submit a solution. Only the github repo link is accepted, including detailed readme, ipynb file, model download drive link, and instructions to test your model. For example, we have a folder of images and how we can run your model to test on that. 
- **Evaluation & Judging:** At most 7 days after submission.
- **Winners Announcement:** October 14.





# Problem Statement:
Participants will make a dataset of images representing different buildings on the university campus. The task is to create a machine-learning model that can accurately classify a new image of a building into one of the pre-defined categories (building names). 
The pre-defined 10 building names are: 
* `Butler Hall`
* `Carpenter Hall`
* `Lee Hall`
* `McCain Hall`
* `McCool Hall`
* `Old Main`
* `Simrall Hall`
* `Student Union`
* `Swalm Hall`
* `Walker Hall`


### Guess the building name below:
![dataset_image_walker_hall](https://github.com/user-attachments/assets/0cbebc4e-558f-4156-8e58-9f10d0122175)



```python
print("Walker Hall")
```

    Walker Hall


### Dataset:
- **Images:** You will create a set of labeled images for each building, taken from various angles, times of day, and lighting conditions to ensure diversity by yourself.
- **Labels:** Label each with the corresponding building name.
- **Test Set:** A separate test set of images that participants will not have access to during training but will be used by ourselves for final evaluation.




### Detailed Guide:

#### Step 1: Understanding the Problem
- **Objective:** The goal is to classify images of university buildings into the correct building category.
- **Challenge:** The diversity of angles, lighting conditions, and other variations make the classification task more complex.

#### Step 2: Data Preparation
- **Loading the Dataset:** Use the provided code to load images and their corresponding labels.
- **Preprocessing:** Normalize the images, resize them to a uniform size, and convert labels to a numerical format. You can do data augmentation for making model robust.

#### Step 3: Model Building
- **Choosing a Model:** Start with a simple Convolutional Neural Network (CNN) using PyTorch.
- **Model Architecture:** 
  - **Input Layer:** Accepts the image data.
  - **Convolutional Layers:** Extract features from images.
  - **Pooling Layers:** Reduce the spatial dimensions, which helps in reducing the complexity.
  - **Dense Layers:** Perform the final classification.

#### Step 4: Model Training
- **Train/Test Split:** Split your data into training and validation sets.
- **Training:** Train the model using the provided training data.
- **Validation:** Validate the model on the validation set and adjust hyperparameters if needed.

#### Step 5: Evaluation
- **Accuracy:** Measure how well your model performs on unseen data.
- **Confusion Matrix:** Use a confusion matrix to visualize where your model might be making errors.
- **Classification Report:** Use the `classification_report` from `sklearn` to get precision, recall, and F1 scores.

#### Step 6: Improving the Model
- **Data Augmentation:** Apply techniques like rotation, flipping, and zooming to artificially increase the size of your dataset.
- **Transfer Learning:** Consider using a pre-trained model like VGG16 or ResNet50 and fine-tuning it for your dataset.
- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and model architectures.

#### Step 7: Submission
- **Code Submission:** Participants will submit their code (both .py and Jupyter Notebook files) along with a README explaining their approach through Github Repo. List your participant's name/emails in the readme file. 
- **Model Submission:** Submit the separate download link of a trained model like gdrive/onedrive. Make sure it's accessible to anyone.
- **Instruction:** How can we test your model?
- **Report:** A short report detailing their model’s performance, challenges, and how they overcame them in a Readme file or pdf.
- 
**Failure to do so is auto disqualify from competition.**

### Tools and Libraries:
- **Programming Languages:** Python
- **Frameworks:** PyTorch, TensorFlow (optional)
- **Data Handling:** NumPy, Pandas, PIL (for image handling)
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** scikit-learn for metrics like accuracy, precision, recall, F1 score, and confusion matrix

### Support and Mentorship:
- Use the Project Github Repository Issues page to submit any issues. Within 24 hours we will try to address that issue. Also, Discord group is provided for communication. 

### Evaluation Criteria:
- **Accuracy:** The primary metric, based on performance on the test set (We will not provide train/test set). `Precision, Recall, F1 score, and Log loss` are other secondary metrics used for evaluation. 
- **Code Quality:** Readability, comments, and proper documentation.
- **Innovation:** Creative approaches to improving model performance.

**You can use any resources/code available on the internet. We only evaluate overall as your logic/test scores obtained on our test dataset.**

### Judging Panel:
- Professors and experienced AI practitioners from the Department of Computer Science.
- Potential guest judges from the industry.
- Only results will be shown to participants. 

### Prizes:
- **1st Place:** Certificate of Achievement + \$200 gift card.
- **2nd Place:** Certificate of Achievement + \$150 gift card.
- **3rd Place:** Certificate of Achievement + \$100 gift card.


### Wrap-Up:
- Host a final event to announce winners, showcase top solutions, and provide feedback. 

### Step 1: Setting Up the  Dataset
Assume you have a  dataset with 10 classes of images, each class representing a folder. For example:

```bash
/dataset/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── val/
        ├── class_1/
        ├── class_2/
        └── ...

```
Each folder (e.g., class_1, class_2) contains images belonging to that class. In our case, class means building name.



### Step 2: Extracting and Organizing the Dataset

* Organize into train and val folders if not already done
* This assumes that your extracted dataset has a structure that you can move around easily.
* You can use Python, shell script, or do it manually.

### Step 3: Detailed Starter Code with Instructions

Here's a detailed version of the starter code with instructions, along with an explanation of how to set up a dummy image dataset and the necessary preprocessing steps. This guide is intended to be beginner-friendly, with comments explaining each step of the process.

#### 1. **Data Preprocessing and Augmentation**
- **RandomResizedCrop(512):** This randomly crops the image and resizes it to 512x512 pixels. This helps the model generalize better by learning from different parts of the image.
- **RandomHorizontalFlip():** Randomly flips the image horizontally, which can help the model learn invariance to the orientation of the image.
- **ToTensor():** Converts the image from a PIL format to a PyTorch tensor, which is necessary for using it in the model.
- **Normalize():** Standardizes the pixel values to have a mean and standard deviation similar to the pre-trained model’s training set. This helps in faster convergence.

#### 2. **Loading Data**
- **ImageFolder:** A PyTorch utility that loads images from folders and automatically assigns labels based on the folder names.
- **DataLoader:** A PyTorch utility that efficiently loads batches of data for training.

#### 3. **Model Definition**
- **ResNet18:** A pre-trained deep learning model that is often used for image classification tasks. We modify the last layer to match the number of classes in our dataset.

#### 4. **Training the Model**
- **Criterion:** The loss function (CrossEntropyLoss) calculates how far off the model’s predictions are from the actual labels.
- **Optimizer:** The optimizer (SGD) updates the model’s weights to reduce the loss.

### Step 5: Prediction Code Explanation

####  **Predict with the Model**
- **Preprocessing:** 
  - We need to apply the same transformations to the test image as we did during training. This includes resizing, cropping, converting to a tensor, and normalizing.
  - **unsqueeze(0):** Adds a batch dimension because the model expects input in the shape of (batch_size, channels, height, width).
  
- **Model Evaluation:**
  - **model.eval():** Switches the model to evaluation mode, which turns off dropout and batch normalization layers.
  - **torch.no_grad():** Context manager that disables gradient calculation, which saves memory and computations during inference.
  - **Only these metrics will be used: overall accuracy, precision, recall, f1 score and log loss.**
  
- **Prediction:**
  - The model outputs raw scores (logits) for each class. The `torch.max()` function retrieves the class with the highest score.
  - Finally, we map the predicted class index to the class name using `class_names`.

### How to Use the Starter Code

1. **Train the Model:** Run the training code to train your model on the dataset.
2. **Save the Model:** After training, the best model will be saved as `best_model.pth`.
3. **Predict New Images:** Use the prediction function to classify new images. Provide the image path, and the model will output the predicted class name. You can iterate through validation/test images and calculate overall in-class accuracies.


This code is structured to help beginners understand the steps involved in building, training, and deploying an image classification model using PyTorch. The detailed instructions should make it easier to follow along and experiment with different parts of the code.


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os

# Step 1: Data Preprocessing and Augmentation
# --------------------------------------------
def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(512),        # Randomly crop the image to 512x512
            transforms.RandomHorizontalFlip(),        # Randomly flip the image horizontally
            transforms.ToTensor(),                    # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),                    # Resize the image to 512x512
            transforms.CenterCrop(512),                # Crop from the center
            transforms.ToTensor(),                     # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], # Normalize with standard values
                                 [0.229, 0.224, 0.225]) # Mean and Std for RGB channels
        ]),
    }
    return data_transforms

# Step 2: Load Data
# -----------------
def load_data(data_dir, batch_size=32):
    data_transforms = get_data_transforms()
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# Step 3: Define the Model
# ------------------------
def create_model(num_classes):
    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    
    # Get the input dimensions of the last layer (fully connected layer)
    num_ftrs = model.fc.in_features
    
    # Replace the last layer with a new one that has `num_classes` outputs
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Step 4: Train the Model
# -----------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()  # Store the best model weights
    best_acc = 0.0  # Track the best validation accuracy
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model to training mode
            else:
                model.eval()   # Set the model to evaluation mode
            
            running_loss = 0.0  # Track the loss
            running_corrects = 0  # Track the number of correct predictions
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU/CPU
                
                optimizer.zero_grad()  # Clear the gradients
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass
                    _, preds = torch.max(outputs, 1)  # Get the class with the highest score
                    loss = criterion(outputs, labels)  # Calculate the loss
                    
                    if phase == 'train':
                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update the weights
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

# Step 5: Evaluate the Model
# --------------------------
def evaluate_model(model, dataloader, criterion, class_names):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)  # Get the class with the highest score
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())    # Store predicted labels
            all_probs.extend(probs.cpu().numpy())    # Store probabilities
    
    avg_loss = running_loss / len(dataloader.dataset)  # Calculate average loss
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logloss = log_loss(all_labels, all_probs)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    
    return accuracy, precision, recall, f1, logloss

# Example Usage
# -------------
if __name__ == "__main__":
    data_dir = "path/to/dataset"
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
    
    # Evaluate the model on the validation set
    print("Validation Set Performance:")
    evaluate_model(model, dataloaders['val'], criterion, class_names)
    
    # Save the best model
    torch.save(model.state_dict(), "best_model.pth")

```

### Evaluation
Calculate overall accuracy, precision, recall, F1 score, and log loss for a multi-class classification problem using PyTorch. 

### Explanation of the Metrics Calculation

#### 1. **Accuracy:**
- **`accuracy_score`**: Calculates the overall accuracy by comparing the true labels (`all_labels`) with the predicted labels (`all_preds`).

#### 2. **Precision, Recall, F1 Score:**
- **`precision_recall_fscore_support`**: This function from `sklearn` calculates precision, recall, and F1 score for multi-class classification. We use `average='weighted'` to consider the imbalance between classes when computing the scores.

#### 3. **Log Loss:**
- **`log_loss`**: Measures the performance of the classification model when the output is a probability value between 0 and 1. It penalizes wrong predictions with high confidence more severely.

### Step-by-Step Breakdown

1. **Preprocessing & Augmentation:** The data is prepared with augmentations for the training set and just resized for the validation set.
2. **Data Loading:** The dataset is loaded using `ImageFolder` and `DataLoader`, with batch size and other parameters specified.
3. **Model Definition:** A ResNet18 model is adapted to the number of classes in the dataset.
4. **Training:** The model is trained using a loop over epochs, with the loss and accuracy being printed for each epoch.
5. **Evaluation:**
   - **Evaluate Model:** The model is set to evaluation mode. The true labels, predicted labels, and probabilities are collected for the validation set.
   - **Calculate Metrics:** Accuracy, precision, recall, F1 score, and log loss are calculated using the collected predictions and labels.
   - **Print Metrics:** The calculated metrics are printed to the console.

