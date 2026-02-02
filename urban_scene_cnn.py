"""
Week 3 Assignment: Neural Network Models for Urban Scene Classification
Author: Muhammad Hassan Naeem
Date: February 1, 2026
Description: CNN implementation for classifying urban scenes using MIT Places dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# STEP 1: Dataset Loading and Preprocessing
# ============================================================================

def load_dataset(dataset_path="./MIT_Places_Urban_Subset", batch_size=32):
    """
    Load and preprocess the MIT Places Urban Subset dataset
    
    Args:
        dataset_path: Path to the dataset directory
        batch_size: Batch size for data loaders
    
    Returns:
        train_loader, val_loader, test_loader, dataset
    """
    print("\n" + "="*60)
    print("STEP 1: Loading and Preprocessing Dataset")
    print("="*60)
    
    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n⚠️  Dataset not found at: {dataset_path}")
        print("Creating a dummy dataset structure for demonstration...")
        create_dummy_dataset(dataset_path)
    
    # Load dataset from local directory
    dataset = ImageFolder(root=dataset_path, transform=transform)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Number of classes: {len(dataset.classes)}")
    print(f"  Classes: {dataset.classes}")
    
    # Split dataset into training (70%), validation (15%), and test (15%) sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset Split:")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Validation samples: {len(val_set)}")
    print(f"  Test samples: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, dataset


def create_dummy_dataset(dataset_path, num_classes=5, samples_per_class=100):
    """
    Create a dummy dataset for demonstration purposes
    
    Args:
        dataset_path: Path where dummy dataset will be created
        num_classes: Number of urban scene classes
        samples_per_class: Number of samples per class
    """
    print("\nCreating dummy dataset for demonstration...")
    
    class_names = ["street", "highway", "building", "park", "square"][:num_classes]
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create dummy images
        for i in range(samples_per_class):
            # Create a random RGB image
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            
            # Save as PNG using PIL
            from PIL import Image
            img_pil = Image.fromarray(img)
            img_path = os.path.join(class_dir, f"{class_name}_{i:04d}.png")
            img_pil.save(img_path)
    
    print(f"✓ Dummy dataset created with {num_classes} classes and {samples_per_class} samples each")


def visualize_samples(dataset, num_samples=6):
    """
    Visualize sample images from the dataset
    
    Args:
        dataset: The dataset to visualize
        num_samples: Number of samples to display
    """
    print("\nVisualizing sample images...")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sample_image, sample_label = dataset[i]
        
        # Denormalize image for visualization
        img = sample_image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {dataset.classes[sample_label]}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches='tight')
    print("✓ Sample images saved as 'sample_images.png'")
    plt.show()


# ============================================================================
# STEP 2: CNN Model Architecture
# ============================================================================

class UrbanSceneCNN(nn.Module):
    """
    Convolutional Neural Network for Urban Scene Classification
    Features:
    - Multiple convolutional layers with batch normalization
    - Dropout for regularization
    - ReLU activations
    - Max pooling
    """
    
    def __init__(self, num_classes):
        super(UrbanSceneCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully Connected Layers
        # After 3 pooling layers: 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        
        return x


def print_model_summary(model, num_classes):
    """
    Print a summary of the model architecture
    """
    print("\n" + "="*60)
    print("STEP 2: CNN Model Architecture")
    print("="*60)
    print(f"\nModel: UrbanSceneCNN")
    print(f"Number of classes: {num_classes}")
    print("\nArchitecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")


# ============================================================================
# STEP 3: Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, 
                epochs=10, device=device):
    """
    Train the CNN model
    
    Args:
        model: The CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        criterion: Loss function
        epochs: Number of training epochs
        device: Device to train on (CPU or CUDA)
    
    Returns:
        train_losses, val_losses, val_accuracies
    """
    print("\n" + "="*60)
    print("STEP 3: Training the CNN Model")
    print("="*60)
    
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print epoch results
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Best model saved! (Val Acc: {best_val_accuracy:.4f})")
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print("="*60)
    
    return train_losses, val_losses, val_accuracies


# ============================================================================
# STEP 4: Evaluation and Visualization
# ============================================================================

def evaluate_model(model, test_loader, device=device):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained CNN model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        test_accuracy, predictions, true_labels
    """
    print("\n" + "="*60)
    print("STEP 4: Evaluating Model Performance")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = correct / total
    
    print(f"\nTest Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Correct predictions: {correct}/{total}")
    
    return test_accuracy, all_predictions, all_labels


def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    Plot training history (loss and accuracy curves)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax2.plot(epochs, val_accuracies, 'g-^', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training history saved as 'training_history.png'")
    plt.show()


def plot_final_results(test_accuracy):
    """
    Plot final test accuracy
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(['Test Accuracy'], [test_accuracy], color='#2ecc71', width=0.5)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('CNN Model Performance on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_accuracy.png', dpi=150, bbox_inches='tight')
    print("✓ Test accuracy plot saved as 'test_accuracy.png'")
    plt.show()


def plot_confusion_matrix(predictions, true_labels, class_names):
    """
    Plot confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute the complete pipeline
    """
    print("\n" + "="*60)
    print("Week 3: Urban Scene Classification with CNN")
    print("Author: Muhammad Hassan Naeem")
    print("="*60)
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    
    # Step 1: Load Dataset
    train_loader, val_loader, test_loader, dataset = load_dataset(
        batch_size=BATCH_SIZE
    )
    
    # Visualize sample images
    visualize_samples(dataset, num_samples=6)
    
    # Step 2: Initialize Model
    num_classes = len(dataset.classes)
    model = UrbanSceneCNN(num_classes)
    print_model_summary(model, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Step 3: Train Model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        epochs=EPOCHS, device=device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Step 4: Evaluate Model
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy, predictions, true_labels = evaluate_model(
        model, test_loader, device=device
    )
    
    # Plot final results
    plot_final_results(test_accuracy)
    
    # Plot confusion matrix
    plot_confusion_matrix(predictions, true_labels, dataset.classes)
    
    print("\n" + "="*60)
    print("✅ All steps completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - sample_images.png")
    print("  - training_history.png")
    print("  - test_accuracy.png")
    print("  - confusion_matrix.png")
    print("  - best_model.pth")
    print("\n🎉 Project complete! Ready for GitHub submission!")
    print("="*60)


if __name__ == "__main__":
    main()
