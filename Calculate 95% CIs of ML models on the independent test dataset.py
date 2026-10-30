# Calculate 95% confidence intervals (CIs) for the independent test dataset

# ============================================================
# Importing the required libraries
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, matthews_corrcoef, precision_recall_curve, auc
from sklearn.svm import SVC # Support Vector Machine Classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.ensemble import ExtraTreesClassifier # Extra Trees Classifier
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Classifier
from xgboost import XGBClassifier # XGboost Classifier
from sklearn.neural_network import MLPClassifier # Artificial Neural Network Classifier
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Building ML models
# ============================================================

SVM = SVC(C=1.0, kernel='rbf', gamma=1, probability=True) # Support vector machine
DT = DecisionTreeClassifier(random_state=0) # Decision tree
RF = RandomForestClassifier(random_state=0) # Random forest
ERT = ExtraTreesClassifier(random_state=0) # Extra trees
GB = GradientBoostingClassifier(random_state=0) # Gradient boosting
XGBoost = XGBClassifier() # XGboost
ANN = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs', max_iter = 500, random_state = 0) # Artificial neural network

# ============================================================
# Load the dataset
# ============================================================
Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/ML Model Building (T1DM)/Training Dataset Preparation (T1DM)/5 Used T1DM Training Dataset/T1DM standardized training dataset 1.xlsx')
X = Data.drop(['Class', 'Official gene symbol'], axis = 1) # X is the feature set (independent variables)
Y = Data['Class'] # Y is the target variable (dependent variable): It could be binary 0 or 1 (Corresponding labels)
X1 = np.array(X)
Y1 = np.array(Y)

# Display the number of genes and features
print("\nT1DM standardized training dataset:\n","Number of genes = ",X1.shape[0],"\n","Number of features = ",X1.shape[1],"\n")
print("All the features in this dataset have continuous values")

# =======================================================================
# Split the dataset into 80% training dataset and 20% testing dataset
# =======================================================================

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.20, random_state = 0)

# ============================================================
# 2. Load the Independent Test Dataset
# ============================================================

# TODO: Replace this path with the actual path to your independent test dataset file
independent_test_path = 'E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/ML Model Building (T1DM)/Training Dataset Preparation (T1DM)/5 Used T1DM Training Dataset/Independent test dataset.xlsx'

test_data = pd.read_excel(independent_test_path)
X_test = test_data.drop(['Class', 'Official gene symbol'], axis = 1) 
Y_test = test_data['Class'] 

X_test_arr = np.array(X_test)
Y_test_arr = np.array(Y_test)

print("T1DM independent test dataset:\n", "Number of genes = ", X_test_arr.shape[0], "\n", "Number of features = ", X_test_arr.shape[1], "\n")
print("All the features in this dataset have continuous values")

# =======================================================================
# Train Model on Full Training Dataset
# =======================================================================

model = ANN
model.fit(X_train1, Y_train1)

# =======================================================================
# Generate Predictions on the Independent Test Dataset
# =======================================================================

Y_pred = model.predict(X_test_arr)
Y_prob = model.predict_proba(X_test_arr)[:, 1] 

# =======================================================================
# Bootstrapping Function for 95% Confidence Intervals
# =======================================================================

def compute_bootstrap_cis(y_true, y_pred, y_prob, n_resamples=1000, alpha=0.95):
    """
    Computes 95% confidence intervals for classification metrics on independent test data.
    """
    bootstrapped_scores = {'Accuracy': [], 'F1': [], 'AUROC': [], 'Precision': [], 'MCC': [], 'PR-AUC': []}
    n_samples = len(y_true)
    rng = np.random.default_rng(seed=0) # Seed fixed for reproducibility
    
    for _ in range(n_resamples):
        # Sample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        # Guard clause if sample accidentally contains only a single class
        if len(np.unique(y_true[indices])) < 2:
            continue
         
        # Append calculated metrics per bootstrap resample
        bootstrapped_scores['Accuracy'].append(accuracy_score(y_true[indices], y_pred[indices]))
        bootstrapped_scores['F1'].append(f1_score(y_true[indices], y_pred[indices]))
        bootstrapped_scores['AUROC'].append(roc_auc_score(y_true[indices], y_prob[indices]))
        bootstrapped_scores['Precision'].append(precision_score(y_true[indices], y_pred[indices]))
        bootstrapped_scores['MCC'].append(matthews_corrcoef(y_true[indices], y_pred[indices]))
        
        precision, recall, _ = precision_recall_curve(y_true[indices], y_prob[indices])
        bootstrapped_scores['PR-AUC'].append(auc(recall, precision))

    # Calculate percentiles for bounds
    lower_pct = ((1.0 - alpha) / 2.0) * 100
    upper_pct = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    print("\n--- Model Evaluation Metrics with 95% Confidence Intervals (Independent Test Dataset) ---")
    for metric, scores in bootstrapped_scores.items():
        median = np.median(scores)
        lower_bound = np.percentile(scores, lower_pct)
        upper_bound = np.percentile(scores, upper_pct)
        print(f"{metric:<10}: {median:.3f} (95% CI: {lower_bound:.3f} - {upper_bound:.3f})")

# Execute the evaluation
compute_bootstrap_cis(Y_test_arr, Y_pred, Y_prob)
