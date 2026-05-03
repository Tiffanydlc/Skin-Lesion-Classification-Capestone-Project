# Skin Lesion Classification Capstone Project

## Project Overview
This repository contains my DATA 4382 capstone project on skin lesion classification. The goal of this project is to classify skin lesion images into 9 diagnostic classes using image preprocessing, feature engineering, and machine learning models.

The project compares two machine learning approaches:
- Random Forest
- XGBoost

Both models were trained on tabular features extracted from skin lesion images and evaluated using visual and quantitative performance metrics.

## Dataset
The dataset includes 9 lesion classes:
- Pigmented benign keratosis
- Melanoma
- Basal cell carcinoma
- Nevus
- Squamous cell carcinoma
- Vascular lesion
- Actinic keratosis
- Dermatofibroma
- Seborrheic keratosis

Main dataset file:
- `1. Data/SC_Dataset_9_Classes.csv`

Sample images are stored in:
- `1. Data/sample_images/`

## Project Workflow
The overall modeling workflow is shown below.

![Modeling Pipeline](5.%20Images/ML%20Data%20Viz/Modeling%20Pipeline.png)

This pipeline includes:
- Image preprocessing
- Hair removal
- Stratified train/validation split
- Feature extraction and engineering
- Model training with Random Forest and XGBoost
- Evaluation using predictions on held-out data

## Class Distribution
The figure below shows the distribution of the original 9 lesion classes.

![Class Distribution](5.%20Images/ML%20Data%20Viz/Class%20Distribution.png)

## Repository Structure
```text
1. Data/
2. Notebooks/
3. Models/
4. Results : Presentations/
5. Images/
requirements.txt
README.md
```

## Notebooks And Files
- `2. Notebooks/Image to Tabular Pipeline.ipynb`
  - builds the image-to-tabular feature extraction pipeline
- `2. Notebooks/RF_9_classes.ipynb`
  - Random Forest training and evaluation notebook
- `2. Notebooks/xgBoost_9_classes.ipynb`
  - XGBoost training and evaluation notebook
- `2. Notebooks/preprocess.py`
  - preprocessing script for image preparation
- `3. Models/rf_9_classes_model.pkl`
  - saved Random Forest deployment model
- `3. Models/xgb_best_model.pkl`
  - saved XGBoost deployment model

## Random Forest Results
The Random Forest model was evaluated with confusion matrices, ROC curves, impurity-based feature importance, permutation importance, and SHAP-based interpretation.

### Confusion Matrix
![RF Confusion Matrix](5.%20Images/ML%20Data%20Viz/RF%20Confusion%20Matrix.png)

### ROC Curves
![RF ROC Curves](5.%20Images/ML%20Data%20Viz/RF%20-ROC%20Curves.png)

### Top 30 Feature Importances
![RF Top 30 Feature Importance](5.%20Images/ML%20Data%20Viz/RF%20Top%2030%20Feature%20Importance.png)

### Permutation Importances
![RF Permutation Importances](5.%20Images/ML%20Data%20Viz/RF%20-Permutation%20Importances.png)

### Global SHAP Importances
![RF Global SHAP Importances](5.%20Images/ML%20Data%20Viz/RF%20Global%20SHAP%20Importances.png)

## XGBoost Results
The XGBoost model was also evaluated with the same core metrics and feature importance plots.

### Confusion Matrix
![XGB Confusion Matrix](5.%20Images/ML%20Data%20Viz/XGB-%20Confusion%20Matrix.png)

### ROC Curve
![XGB ROC Curve](5.%20Images/ML%20Data%20Viz/XGB%20-%20ROC%20Curve.png)

### Top 30 Feature Importances
![XGB Top 30 Feature Importance](5.%20Images/ML%20Data%20Viz/XGB%20-%20Top%2030%20Feature%20Importance.png)

### Permutation Importance
![XGB Permutation Importance](5.%20Images/ML%20Data%20Viz/XGB%20-%20Premutation%20Importance.png)

## Evaluation Metrics
The project uses the following evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Feature importance
- Permutation importance
- SHAP interpretation

## Libraries Used
Main libraries used in this project:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`
- `shap`
- `opencv-python`
- `Pillow`
- `scikit-image`
- `mahotas`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Presentations
Project presentation materials are stored in:
- `4. Results : Presentations/`

## Author
Tiffany Delacruz
