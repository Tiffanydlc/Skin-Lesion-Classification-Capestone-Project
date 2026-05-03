# Skin Lesion Classification Capstone Project

## Project Overview
This repository contains my DATA 4382 capstone project on skin lesion classification. The goal of this project is to classify skin lesion images into 9 diagnostic classes using image preprocessing, feature engineering, and machine learning models.

The project compares two tree-based machine learning approaches:
- Random Forest
- XGBoost

These models were trained on tabular features extracted from dermoscopic images and evaluated using visual and quantitative performance metrics. Earlier project phases also compared the feature-based pipeline against a MobileNetV2 deep learning baseline to study how classical machine learning behaves under limited data and class imbalance.

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

One of the main challenges in this project is severe class imbalance. The larger classes contain far more examples than the smaller classes, which makes minority lesion types harder to classify consistently and increases the risk that a model will favor the majority classes.

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

## Key Observations
- The classical machine learning pipeline performed strongly on structured image-derived features and consistently outperformed the early MobileNetV2 baseline explored during the project.
- In earlier three-class experiments, XGBoost achieved about 0.86 accuracy while Random Forest achieved about 0.79 to 0.81, showing that both models were competitive but XGBoost was slightly stronger overall.
- In later nine-class experiments, both Random Forest and XGBoost produced very similar macro ROC-AUC values of about 0.885 and 0.884, suggesting that both models captured useful class-separation patterns even in the harder multi-class setting.
- Nevus was one of the strongest-performing classes across experiments, while some minority classes remained more difficult to separate because of limited examples and overlapping feature distributions.
- Several misclassifications occurred between visually similar lesion types, which supports the idea that class imbalance and overlapping color patterns remain important modeling challenges.
- Feature importance analysis showed that the models did not depend on a single dominant predictor. Instead, they relied on groups of correlated features working together.
- Color histogram features were consistently influential, especially in earlier project phases, but later feature engineering showed that shape-based features such as asymmetry, border irregularity, radius, area, and rectangle-based measurements also became important.
- SHAP and permutation importance helped confirm that combinations of features were more informative than any one feature by itself, which improved interpretability of the final models.

## Preprocessing Improvements
Preprocessing was a major part of the project because dermoscopic images often contain hair artifacts that interfere with feature extraction.

- An earlier hair-removal method introduced noise, corrupted some images, and took about 1.5 hours to run on a small dataset.
- The pipeline was improved by replacing that method with DullRazor, which preserves lesion structure more effectively.
- The updated preprocessing step reduced runtime to about 3 minutes on the same smaller workload and produced cleaner images for downstream feature extraction.
- Cleaner preprocessing improved confidence that the extracted color, texture, and shape features represented the lesion itself rather than noise.

## Challenges
- Severe class imbalance made minority classes harder to detect and evaluate fairly.
- Some lesion classes had overlapping color and texture patterns, which increased confusion between clinically similar categories.
- Deep learning performance was less stable in the earlier stages of the project, especially under limited data and imbalance.
- Interpreting low individual feature-importance scores was difficult at first because many engineered features were correlated with one another.

## Conclusions And Future Work
- The project supports the idea that feature-based machine learning can be a practical and interpretable alternative to deep learning for medical image classification when data are limited or imbalanced.
- XGBoost and Random Forest both showed that handcrafted features can capture clinically meaningful lesion information.
- Future work includes improving feature engineering further, especially with medically informed ABCD-style features based on asymmetry, border, color, and diameter.
- Additional imbalance-handling strategies, deeper evaluation of minority classes, and further refinement of the deployment pipeline can help strengthen model reliability.

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
