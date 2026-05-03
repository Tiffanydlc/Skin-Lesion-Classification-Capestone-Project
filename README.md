# Skin Lesion Classification Capstone Project

## 1. Project Title
Skin Lesion Classification Using Classical Computer Vision, Random Forest, and XGBoost

## 2. Business Problem / Motivation
Early skin cancer detection matters because it can improve patient outcomes and support faster clinical decision-making. This project explores whether interpretable, feature-based machine learning can classify dermoscopic skin lesion images effectively, especially in a setting with limited data and strong class imbalance.

## 3. Project Overview
This project classifies 9 skin lesion classes using a classical computer vision pipeline. Images are preprocessed, converted into tabular features, and then modeled with Random Forest and XGBoost. The main result is that the tree-based models performed strongly on structured image-derived features and provided useful interpretability through feature importance, permutation importance, and SHAP.

## 4. Data
- Source: [Kaggle Skin Cancer ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data)
- Type: Dermoscopic image dataset converted into structured tabular features for modeling
- Size: 2,357 images across 9 classes
- Main file in this repository: `1. Data/SC_Dataset_9_Classes.csv`
- Key features:
  - RGB color histogram features
  - Haralick texture features
  - Edge-based features
  - Shape-inspired features such as asymmetry, border irregularity, radius, area, and rectangle-based measurements

The 9 classes are:
- Pigmented benign keratosis
- Melanoma
- Basal cell carcinoma
- Nevus
- Squamous cell carcinoma
- Vascular lesion
- Actinic keratosis
- Dermatofibroma
- Seborrheic keratosis

## 5. Data Preprocessing
- Cleaning steps:
  - Organized image and tabular data into a reproducible project structure
  - Applied image preprocessing before feature extraction
- Handling missing values:
  - Median imputation was used inside the machine learning pipeline through `SimpleImputer`
  - The tabular dataset was also checked for null values during analysis
- Feature engineering:
  - Extracted histogram-based color features
  - Extracted Haralick texture features
  - Added shape and lesion-geometry features
  - Added contrast and color-ratio style features for improved class separation
- Hair removal:
  - Replaced an earlier slower and less stable filter with DullRazor
  - Improved preprocessing speed and reduced image corruption

## 6. Exploratory Data Analysis (EDA)
### Class Distribution
![Class Distribution](5.%20Images/ML%20Data%20Viz/Class%20Distribution.png)

Insight:
- The dataset is strongly imbalanced, with some classes having many more examples than others. This makes minority-class prediction more difficult and increases the importance of using evaluation metrics beyond accuracy.

### Modeling Pipeline
![Modeling Pipeline](5.%20Images/ML%20Data%20Viz/Modeling%20Pipeline.png)

Insight:
- The project follows two main directions after preprocessing: a classical feature-engineering pipeline and a deep learning baseline path. The classical pipeline became the stronger and more stable direction for this dataset.

## 7. Modeling Approach
- Baseline model:
  - A MobileNetV2 deep learning baseline was explored in earlier project phases for comparison
- Advanced models:
  - Random Forest
  - XGBoost
- Why these models were chosen:
  - Both models perform well on structured tabular data
  - They work well with nonlinear interactions
  - They are more interpretable than CNN-based image models
  - They are practical under limited data and class imbalance

## 8. Model Training
- Tools used:
  - `scikit-learn`
  - `xgboost`
  - `imbalanced-learn`
  - `shap`
  - `opencv-python`
- Hyperparameters:
  - Random Forest was tuned with `RandomizedSearchCV`
  - XGBoost was tuned with a stratified split and regularized boosting settings
- Training process:
  - Used stratified splitting to preserve class balance
  - Applied feature engineering to convert image information into tabular inputs
  - Trained models on the engineered feature set
  - Evaluated predictions using held-out validation/test data
  - Saved trained deployment models as `.pkl` files in `3. Models/`

## 9. Results
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - ROC-AUC

Why these metrics were chosen:
- Accuracy alone can be misleading on imbalanced data
- Precision and recall help show class-specific performance
- F1-score balances precision and recall
- ROC-AUC helps evaluate class separation quality across thresholds

### Model Comparison Table
| Model | Setting | Reported Performance |
|---|---|---|
| Random Forest | Earlier 3-class experiments | About 0.79 to 0.81 accuracy |
| XGBoost | Earlier 3-class experiments | About 0.86 accuracy |
| Random Forest | 9-class experiment | Macro ROC-AUC about 0.885 |
| XGBoost | 9-class experiment | Macro ROC-AUC about 0.884 |

### Random Forest Visualizations
![RF Confusion Matrix](5.%20Images/ML%20Data%20Viz/RF%20Confusion%20Matrix.png)

![RF ROC Curves](5.%20Images/ML%20Data%20Viz/RF%20-ROC%20Curves.png)

### XGBoost Visualizations
![XGB Confusion Matrix](5.%20Images/ML%20Data%20Viz/XGB-%20Confusion%20Matrix.png)

![XGB ROC Curve](5.%20Images/ML%20Data%20Viz/XGB%20-%20ROC%20Curve.png)

## 10. Model Interpretation
This project includes multiple model interpretation techniques:
- Feature importance
- Permutation importance
- SHAP values

### Random Forest Interpretation
![RF Top 30 Feature Importance](5.%20Images/ML%20Data%20Viz/RF%20Top%2030%20Feature%20Importance.png)

![RF Permutation Importances](5.%20Images/ML%20Data%20Viz/RF%20-Permutation%20Importances.png)

![RF Global SHAP Importances](5.%20Images/ML%20Data%20Viz/RF%20Global%20SHAP%20Importances.png)

### XGBoost Interpretation
![XGB Top 30 Feature Importance](5.%20Images/ML%20Data%20Viz/XGB%20-%20Top%2030%20Feature%20Importance.png)

![XGB Permutation Importance](5.%20Images/ML%20Data%20Viz/XGB%20-%20Premutation%20Importance.png)

Interpretation summary:
- The models do not rely on one single feature
- Groups of correlated features drive predictions together
- Color histogram features were especially important in earlier project phases
- Later experiments showed that shape-based features also became highly influential

## 11. Key Insights
- The classical machine learning pipeline worked better than the earlier deep learning baseline for this project setting
- XGBoost was slightly stronger than Random Forest in earlier three-class experiments
- In the nine-class setting, both models performed very similarly in macro ROC-AUC
- Nevus was one of the strongest-performing classes across experiments
- Minority classes remained harder to classify because of imbalance and overlapping feature patterns
- The practical impact is that interpretable tree-based models can be a strong alternative when medical image data are limited, imbalanced, or noisy

## 12. Conclusion
This project shows that feature-based machine learning can classify skin lesion images effectively while remaining interpretable. Random Forest and XGBoost both performed well on tabular features extracted from dermoscopic images, and interpretation tools helped explain what drove the predictions.

## 13. Future Work
- Expand medically informed feature engineering using ABCD-style lesion features
- Improve minority-class performance further
- Continue refining preprocessing and feature selection
- Compare against stronger deep learning baselines
- Extend deployment and model evaluation workflows

## 14. How to Run
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run preprocessing:
  - Use `2. Notebooks/preprocess.py`
  - Or open the notebooks and run preprocessing cells

- Train model:
  - Run `2. Notebooks/RF_9_classes.ipynb`
  - Run `2. Notebooks/xgBoost_9_classes.ipynb`

- Evaluate results:
  - Review saved plots in `5. Images/ML Data Viz/`
  - Review saved model files in `3. Models/`
  - Review presentation/report material in `4. Results : Presentations/`

## 15. Repository Structure Explanation
```text
project-name/
├── README.md
├── requirements.txt
├── 1. Data/
├── 2. Notebooks/
├── 3. Models/
├── 4. Results : Presentations/
└── 5. Images/
```

Folder explanation:
- `1. Data/`
  - dataset CSV and sample lesion images
- `2. Notebooks/`
  - notebooks and scripts for preprocessing, feature extraction, and model training
- `3. Models/`
  - saved deployment `.pkl` model files
- `4. Results : Presentations/`
  - presentation PDFs and result communication materials
- `5. Images/`
  - visual outputs including EDA, confusion matrices, ROC curves, and interpretation plots

## 16. Requirements
Install packages with:

```bash
pip install -r requirements.txt
```
