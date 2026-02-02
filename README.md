# Week 3 Assignment: Neural Network Models for Urban Scene Classification

**Author:** Muhammad Hassan Naeem  
**Course:** Computer Vision - Concordia University  
**Date:** February 1, 2026

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for classifying urban scenes using the MIT Places dataset. The model uses advanced techniques including **Batch Normalization** and **Dropout** for improved performance and generalization.

## 🎯 Objectives

✅ Implement a CNN for urban scene classification  
✅ Use MIT Places dataset (subset focusing on urban environments)  
✅ Train, evaluate, and optimize the model  
✅ Visualize results and model performance  
✅ Complete GitHub integration and documentation

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **scikit-learn** - Performance metrics
- **seaborn** - Enhanced visualizations

## 📁 Project Structure

```
Assignment03/Code/
│
├── urban_scene_cnn.py       # Main implementation file
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── MIT_Places_Urban_Subset/ # Dataset directory (auto-created if missing)
│   ├── street/
│   ├── highway/
│   ├── building/
│   ├── park/
│   └── square/
│
└── Output Files (generated after running):
    ├── sample_images.png         # Sample dataset images
    ├── training_history.png      # Training/validation curves
    ├── test_accuracy.png         # Final accuracy plot
    ├── confusion_matrix.png      # Confusion matrix visualization
    └── best_model.pth           # Trained model weights
```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Hassan-Naeem-code/Computer-Vision-Week-03.git
cd Computer-Vision-Week-03
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision numpy matplotlib opencv-python scikit-learn seaborn
```

### 3. Dataset Setup

The code will automatically create a dummy dataset if the MIT Places dataset is not found. For real data:

1. Download MIT Places dataset subset
2. Organize into folders by class name
3. Place in `MIT_Places_Urban_Subset/` directory

## 🏃 Running the Project

Execute the main script:

```bash
python urban_scene_cnn.py
```

This will:
1. Load and preprocess the dataset
2. Display sample images
3. Initialize the CNN model
4. Train the model for 10 epochs
5. Evaluate on test data
6. Generate visualization plots

## 🧠 Model Architecture

### UrbanSceneCNN

The CNN consists of:

**Convolutional Blocks (3):**
- Conv2D → BatchNorm2D → ReLU → MaxPool2D → Dropout2D
- Increasing filters: 32 → 64 → 128

**Fully Connected Layers:**
- Flatten → FC(512) → BatchNorm1D → ReLU → Dropout → FC(num_classes)

**Key Features:**
- Batch Normalization for stable training
- Dropout (0.25 for conv, 0.5 for FC) for regularization
- MaxPooling for spatial dimension reduction
- Total parameters: ~2.5M (varies by num_classes)

## 📊 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Train/Val/Test Split | 70% / 15% / 15% |

## 📈 Results

The model generates the following outputs:

1. **sample_images.png** - Visualization of 6 sample images from dataset
2. **training_history.png** - Loss and accuracy curves over epochs
3. **test_accuracy.png** - Final test set performance
4. **confusion_matrix.png** - Detailed per-class performance
5. **best_model.pth** - Saved model with best validation accuracy

### Expected Performance

With the dummy dataset:
- Training progresses normally
- Model learns to classify synthetic data
- Serves as template for real dataset

With real MIT Places data:
- Expected test accuracy: 70-85% (depends on data quality and size)
- Better performance with more training epochs and data augmentation

## 🔬 Key Implementation Details

### Data Preprocessing
- Images resized to 128×128
- Normalized using ImageNet statistics
- Random train/val/test split with fixed seed

### Training Features
- Automatic best model saving
- Per-epoch validation
- Training and validation metrics tracking
- GPU support (auto-detects CUDA)

### Evaluation
- Comprehensive test set evaluation
- Confusion matrix generation
- Per-class accuracy analysis

## 📝 Code Highlights

### Batch Normalization
Stabilizes training by normalizing layer inputs:
```python
self.bn1 = nn.BatchNorm2d(32)
```

### Dropout Regularization
Prevents overfitting by randomly dropping connections:
```python
self.dropout1 = nn.Dropout2d(0.25)  # For conv layers
self.dropout4 = nn.Dropout(0.5)     # For FC layers
```

### Data Augmentation (Future Enhancement)
Can add to transforms:
```python
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(10),
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

## 🎥 Video Walkthrough Topics

When recording your video (5-7 minutes), cover:

1. **Project Overview** (1 min)
   - Explain the goal and dataset
   
2. **Code Walkthrough** (2-3 min)
   - Dataset loading and preprocessing
   - CNN architecture explanation
   - Training loop details
   
3. **Results Analysis** (2 min)
   - Show training curves
   - Discuss test accuracy
   - Analyze confusion matrix
   
4. **Insights & Improvements** (1 min)
   - What worked well
   - Potential improvements
   - Real-world applications

## 🚀 Potential Improvements

1. **Data Augmentation** - Add random transforms for better generalization
2. **Learning Rate Scheduling** - Reduce LR when validation plateaus
3. **Deeper Architecture** - Add more convolutional blocks
4. **Transfer Learning** - Use pretrained ResNet/VGG as backbone
5. **Ensemble Methods** - Combine multiple models
6. **Early Stopping** - Stop when validation stops improving

## 📚 References

- [MIT Places Dataset](http://places2.csail.mit.edu/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)

## 📦 Submission Checklist

✅ GitHub repository with all code  
✅ `urban_scene_cnn.py` - Complete implementation  
✅ `requirements.txt` - All dependencies  
✅ `README.md` - This documentation  
✅ Generated visualizations (PNG files)  
✅ Video walkthrough (5-7 minutes)  
✅ PowerPoint presentation (6-7 slides)  
✅ ZIP file of repository  

## 🤝 GitHub Integration

### Initial Setup
```bash
git init
git add .
git commit -m "Initial commit: Urban Scene CNN"
git remote add origin https://github.com/Hassan-Naeem-code/Computer-Vision-Week-03.git
git push -u origin main
```

### Subsequent Updates
```bash
git add .
git commit -m "Your commit message here"
git push
```

## 📧 Contact

**Muhammad Hassan Naeem**  
Concordia University  
Computer Vision Course  

---

## 🎉 Conclusion

This project demonstrates the implementation of a CNN for urban scene classification, incorporating modern deep learning techniques like batch normalization and dropout. The modular code structure makes it easy to experiment with different architectures and hyperparameters.

**Ready for submission!** 🚀

---

*Last Updated: February 1, 2026*
