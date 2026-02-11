# Waste-CNN-Image-Classification

## Table Of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Sources](#dataset-sources)
- [Tools and Technologies](#tools-and-technologies)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Model Performance](#model-performance)
- [Model Insights and analysis](#model-insights-and-analysis)
- [limitation](#limitation)
- [Mitigation Strategies](#mitigation-strategies)
  
### Project Overview

This project implements a Convolutional Neural Network (CNN) to automatically classify waste images into two categories:
- Organic
- Recycle
  
The aim is to improve waste sorting efficiency, support recycling systems, and reduce environmental pollution through automation.

### Problem Statement

Manual waste classification is time-consuming and error-prone. This project provides an AI-based solution to:
- Automate waste sorting
- Improve recycling accuracy
- Support smart waste management systems

### Dataset Sources&Description

The primary dataset used for this analysis is the "waste_dataset" containing images of the categories of waste products. Waste_dataset was gotten from kaggle datasets. The dataset consists of labeled waste images categorized into:
- Organic
- Recycle

### Tools and Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

### Data Cleaning and Preparation

In the initial data preparation phase, we performed the following tasks:
- Merged training and test datasets
- Data loading and inspection.
- Cleaning (duplicates removed)
- Renaming systematically
- Resizing to 50×50
- Split into 80% training and 20% testing
  

### Model Architecture

A custom CNN built from scratch(no transfer learning):
```python
input_shape = (50,50,3)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

### Training Configuration

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Output Activation: Sigmoid (Binary Classification)
- Metrics: accuracy
- Epochs: Longer training(100 epochs)
- Batch size: 32
- Data Augmentation:
- Rescaling
- Rotation
- Width and height shifts
- Zoom
- Horizontal flipping

### Model Performance

- Test Accuracy: 89%
- Macro Avg F1-score: 0.89
- Weighted Avg F1-score: 0.89
- 
#### Classification Report

|                 |Precision|Recall  |f1-score|Support|
|-----------------|---------|--------|--------|-------|
|Organic          |    0.93 |  0.88  |   0.90 |  2678 |
|Recycle          |   0.86  |    0.92|   0.89 |  2220 |
|                 |         |        |        |       |
|accuracy         |         |        |  0.89  |  4898 |
|macro avg        | 0.89    | 0.90   | 0.89   | 4898  |       
|weighted avg     | 0.90    |  0.89  |  0.89  |  4898 |

#### Confusion Matrix

[[2349     329]

 [187  2033]]
 
### Model Insights & Analysis

#### What the model learned well:
- The model shows strong performance on both classes, with:
  - High precision for organic waste (fewer false positives).
  - High recall for recycle, meaning fewer recyclable items are missed.

- Some misclassification exists between visually similar items (e.g., food containers vs biodegradable waste).
  
The model demonstrates no extreme class bias, as both classes have balanced precision and recall.

### Limitations
- Dataset size and diversity may limit generalization to real-world waste.
- Performance may drop on:
  - Low-light images
  - Blurry or occluded waste items
  - Waste categories not seen during training

#### Mitigation strategies:

- Expand dataset with more waste categories (glass, metal, paper, e-waste).
- Experiment with transfer learning (MobileNet, ResNet).
- Improve preprocessing (background removal, lighting normalization).
- Deploy as a web or mobile application for real-time usage.

