# Skin Lesion Classification Capstone Project

## Project Overview
This repository contains my DATA 4382 capstone project on skin lesion classification. The project explores machine learning approaches for classifying skin lesion images into 9 classes using tabular features extracted from images.

## Project Goal
The goal of this project is to build and evaluate models that can classify skin lesion types from image-derived features and compare model performance using standard machine learning evaluation metrics.

## Dataset
- Dataset file: `1. Data/SC_Dataset_9_Classes.csv`
- Sample images are included in `1. Data/sample_images/`
- The dataset contains 9 skin lesion classes:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

## Methods Used
- Image-to-tabular feature engineering
- Random Forest classification
- XGBoost classification
- Class balancing with SMOTE
- Hyperparameter tuning with `RandomizedSearchCV`
- Model interpretation with SHAP

## Repository Structure
```text
1. Data/
  SC_Dataset_9_Classes.csv
  sample_images/

2. Notebooks/
  Image to Tabular Pipeline.ipynb
  xgBoost_9_classes.ipynb

3. Models/
  Machine Learning /
    RF_9_classes.ipynb

4. Results/

5. Images/
  ML Data Viz/
```

## Files Description
- `2. Notebooks/Image to Tabular Pipeline.ipynb`
  - Builds the pipeline that converts image data into tabular features.
- `2. Notebooks/xgBoost_9_classes.ipynb`
  - Trains and evaluates the XGBoost model for 9-class classification.
- `3. Models/Machine Learning /RF_9_classes.ipynb`
  - Trains and evaluates the Random Forest model for 9-class classification.
- `5. Images/ML Data Viz/`
  - Stores visual outputs such as confusion matrices, ROC curves, permutation importance plots, and SHAP importance plots.

## Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curves
- Feature importance and SHAP-based interpretation

## Libraries Used
Main Python libraries used in this project include:
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

See `requirements.txt` for the full list.

## How To Run
1. Install required libraries:
```bash
pip install -r requirements.txt
```
2. Open the notebooks in Jupyter Notebook or Google Colab.
3. Run the image-to-tabular pipeline notebook to generate features if needed.
4. Run the model notebooks to train and evaluate the models.

## Results
Model visualizations and machine learning output images are stored in:
- `5. Images/ML Data Viz/`

Additional result artifacts can be stored in:
- `4. Results/`

## Author
Tiffany Delacruz
