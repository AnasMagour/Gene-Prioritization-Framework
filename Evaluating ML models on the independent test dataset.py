# Building and evaluating ML models using python

# ============================================================
# Importing the required libraries
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# Importing the necessary modules
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, classification_report
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

# ===============================================================
# Prepare 10-fold cross-validation on the Training dataset only
# ===============================================================

kf = KFold(n_splits=10)
Accuracy_array = np.empty((10, 1))
F1_score_array = np.empty((10, 1))
AUROC_array = np.empty((10, 1))
Precision_array = np.empty(10)
MCC_array = np.empty(10)
PR_AUC_array = np.empty(10)

# Enumerate splits
for x, (train_index, test_index) in enumerate(kf.split(X_train1, Y_train1)):
    X_train, X_test = X_train1[train_index], X_train1[test_index]
    Y_train, Y_test = Y_train1[train_index], Y_train1[test_index]
    #Define the used classifier
    Classifier = SVM
    # Train the used classifier
    Trained_classifier = Classifier.fit(X_train, Y_train)
    # Predict the response for the testing dataset
    Y_predict = Trained_classifier.predict(X_test)
    Y_predict_proba = Trained_classifier.predict_proba(X_test)[:, 1]

    # Evaluate the used classifier by calculating the average evaluation metrics
    Accuracy_array[x] = accuracy_score(Y_test, Y_predict)
    F1_score_array[x] = f1_score(Y_test, Y_predict)
    AUROC_array[x] = roc_auc_score(Y_test, Y_predict_proba)
    Precision_array[x] = precision_score(Y_test, Y_predict)
    MCC_array[x] = matthews_corrcoef(Y_test, Y_predict)
    PR_AUC_array[x] = average_precision_score(Y_test, Y_predict_proba)

# ============================================================
# Display 10-fold cross-validation results
# ============================================================

print("\n**********10 K-Fold Cross Validation Results**********")
print('\nAverage accuracy = {:.3f}'.format(Accuracy_array.mean()))
print('\nAverage f1_score = {:.3f}'.format(F1_score_array.mean()))
print('\nAverage AUROC = {:.3f}'.format(AUROC_array.mean()))
print('\nAverage precision = {:.3f}'.format(Precision_array.mean()))
print('\nAverage MCC = {:.3f}'.format(MCC_array.mean()))
print('\nAverage PR-AUC = {:.3f}'.format(PR_AUC_array.mean()))

# ============================================================
# Train the FINAL classifier on the complete training set
# ============================================================

Final_classifier = SVM
Final_classifier.fit(X_train1, Y_train1)

# ============================================================
# Testing results on the independent test dataset
# ============================================================

print("\n**********Independent Testing Results**********")

# 1. Load the independent dataset (Update the path to your actual independent file)
Independent_Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/ML Model Building (T1DM)/Training Dataset Preparation (T1DM)/5 Used T1DM Training Dataset/Independent test dataset.xlsx')

# 2. Separate features and target variable
# Ensure the drop columns match your dataset structure
X_independent = Independent_Data.drop(['Class', 'Official gene symbol'], axis=1, errors='ignore') 
Y_independent = Independent_Data['Class']

X_ind_arr = np.array(X_independent)
Y_ind_arr = np.array(Y_independent)

# Display the dataset dimensions
print("Independent Test Dataset:")
print("Number of genes = ", X_ind_arr.shape[0])
print("Number of features = ", X_ind_arr.shape[1], "\n")

# 3. Predict responses and probabilities using the Final_classifier
Y_predict_ind = Final_classifier.predict(X_ind_arr)
Y_predict_proba_ind = Final_classifier.predict_proba(X_ind_arr)[:, 1]

# 4. Calculate and display evaluation metrics
ind_confusion_matrix = confusion_matrix(Y_ind_arr, Y_predict_ind)
print("Independent testing confusion matrix:\n", ind_confusion_matrix, "\n")

ind_classification_report = classification_report(Y_ind_arr, Y_predict_ind)
print("Independent testing classification report:\n", ind_classification_report)

ind_accuracy = accuracy_score(Y_ind_arr, Y_predict_ind)
print('Independent testing accuracy = {:.3f}'.format(ind_accuracy), "\n")

ind_f1_score = f1_score(Y_ind_arr, Y_predict_ind)
print('Independent testing f1-score = {:.3f}'.format(ind_f1_score), "\n")

ind_AUROC = roc_auc_score(Y_ind_arr, Y_predict_proba_ind)
print('Independent testing AUROC = {:.3f}'.format(ind_AUROC), "\n")

ind_precision = precision_score(Y_ind_arr, Y_predict_ind)
print('Independent testing precision = {:.3f}'.format(ind_precision), "\n")

ind_mcc = matthews_corrcoef(Y_ind_arr, Y_predict_ind)
print('Independent testing MCC = {:.3f}'.format(ind_mcc), "\n")

ind_pr_auc = average_precision_score(Y_ind_arr, Y_predict_proba_ind)
print('Independent testing PR-AUC = {:.3f}'.format(ind_pr_auc), "\n")
