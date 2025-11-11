# Face Mask Detection using Transfer Learning (MobileNet-V1)
Efficient deep learning solution for automatic face mask detection based on transfer learning with MobileNet-V1. This project leverages pre-trained weights and custom dataset augmentation to accurately identify whether people are wearing face masks, supporting public health and safety in diverse settings.

# Overview
Face mask detection is crucial in public spaces, workplaces, and healthcare environments to monitor compliance with health guidelines. This project demonstrates how transfer learning with MobileNet-V1 can enable rapid, robust mask detection with minimal computational resources. The workflow features pre-trained weights, data augmentation, and streamlined detection for both real-time and batch applications.

# Dataset
Pre-training: MobileNet-V1 initialized with ImageNet weights for feature extraction.

Fine-tuning: Modified version of the Kaggle Face Mask dataset, which contains labeled images across two classes:

with_mask

without_mask

Images are preprocessed through resizing, normalization, and data augmentation (flipping, rotation, zoom) to bolster generalization.

# Model Architecture
Base: MobileNet-V1 (ImageNet pre-trained weights, initial layers frozen)

Custom top layers: Dense layers for final binary classification

Activations: ReLU for feature layers, Sigmoid for output

Optimizer: Adam

Loss function: Binary Cross-Entropy

# Project Structure
```
Mask-Detection-using-Transfer-Learning/
│
├── data/
│   ├── with_mask/
│   └── without_mask/
├── models/
├── notebooks/
├── utils/
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
```
- `data/`: Contains `with_mask/` and `without_mask/` subfolders for class images
- `notebooks/`: For exploratory data analysis and visualization  
- `train.py`, `evaluate.py`, `predict.py`: Scripts for running training, evaluation, and inference
bash
```
git clone https://github.com/Razanmohamad25/Mask-Detection-using-Transfer-Learning.git
cd Mask-Detection-using-Transfer-Learning
pip install -r requirements.txt
Usage
Training
bash
python train.py
Evaluation
bash
python evaluate.py
Inference
bash
python predict.py --image path_to_image.jpg
You can easily integrate the model with live video feeds or deploy it on edge devices for real-time mask detection.
```
# Results
High accuracy on validation images

Fast inference, suitable for real-time monitoring

Lightweight architecture for mobile and embedded deployment

# Future Improvements
Integration with live camera streams

Support for different mask types

Further model optimization using TensorFlow Lite

