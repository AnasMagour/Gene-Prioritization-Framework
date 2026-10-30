# Building and evaluating DNN model using Python

# ============================================================
# Importing the required libraries
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# Importing the necessary modules
# ============================================================

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Load the dataset
# ============================================================

Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/ML Model Building (T1DM)/Training Dataset Preparation (T1DM)/5 Used T1DM Training Dataset/T1DM standardized training dataset 1.xlsx')
X = Data.drop(['Class', 'Official gene symbol'], axis = 1) # X is the feature set (independent variables)
Y = Data['Class'] # Y is the target variable (dependent variable): It could be binary 0 or 1 (Corresponding labels)
X1 = np.asarray(X, dtype=np.float32)
Y1 = np.asarray(Y).ravel()

# Display the number of genes and features
print("\nT1DM standardized training dataset 1:\n","Number of genes = ",X1.shape[0],"\n","Number of features = ",X1.shape[1],"\n")
print("All the features in this dataset have continuous values")

# =======================================================================
# Split the dataset into 80% training dataset and 20% testing dataset
# =======================================================================

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.20, random_state = 0)

# ===============================================================================
# Function to build the DNN model: DNN model must be created for every CV fold.
# ===============================================================================

def build_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(6,input_dim=input_dim,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# ===============================================================
# Prepare 10-fold cross-validation on the Training dataset only
# ===============================================================

print("\n********** 10-Fold Cross Validation **********")

skf = KFold(n_splits=10)
Accuracy_array = np.empty(10)
F1_score_array = np.empty(10)
ROC_AUC_array = np.empty(10)
Precision_array = np.empty(10)
MCC_array = np.empty(10)
PR_AUC_array = np.empty(10)

for fold, (train_index, test_index) in enumerate(skf.split(X_train1, Y_train1), start=1):
    print(f"\nTraining Fold {fold}/10")
    X_train = X_train1[train_index]
    X_test = X_train1[test_index]
    Y_train = Y_train1[train_index]
    Y_test = Y_train1[test_index]
    # Build a fresh DNN model for this fold
    DNN_model_cv = build_dnn_model(X_train.shape[1])
    # Optional early stopping to reduce unnecessary overfitting
    early_stopping = EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
    # Train the DNN_model on the training dataset
    DNN_model_cv.fit(X_train,Y_train,epochs=100,batch_size=26,callbacks=[early_stopping])
    # Predicted probabilities
    Y_predict_prob = DNN_model_cv.predict(X_test).ravel()
    # Convert probabilities to class labels
    Y_predict = (Y_predict_prob >= 0.5).astype(int)
    # Evaluation metrics
    Accuracy_array[fold - 1] = accuracy_score(Y_test, Y_predict)
    F1_score_array[fold - 1] = f1_score(Y_test, Y_predict)
    ROC_AUC_array[fold - 1] = roc_auc_score(Y_test, Y_predict_prob)
    Precision_array[fold - 1] = precision_score(Y_test, Y_predict)
    MCC_array[fold - 1] = matthews_corrcoef(Y_test, Y_predict)
    PR_AUC_array[fold - 1] = average_precision_score(Y_test, Y_predict_prob)

    print(f"Fold {fold}: "f"Accuracy={Accuracy_array[fold - 1]:.3f}, "f"Precision={Precision_array[fold - 1]:.3f}, "f"F1={F1_score_array[fold - 1]:.3f}, "f"MCC={MCC_array[fold - 1]:.3f}, "f"ROC-AUC={ROC_AUC_array[fold - 1]:.3f}, "f"PR-AUC={PR_AUC_array[fold - 1]:.3f}")
    
# ============================================================
# Average cross-validation results
# ============================================================

print("\n********** 10-Fold Cross Validation Results **********")

print('\nAverage accuracy = {:.3f}'.format(Accuracy_array.mean()))
print('\nAverage f1-score = {:.3f}'.format(F1_score_array.mean()))
print('\nAverage ROC-AUC = {:.3f}'.format(ROC_AUC_array.mean()))
print('\nAverage precision = {:.3f}'.format(Precision_array.mean()))
print('\nAverage MCC = {:.3f}'.format(MCC_array.mean()))
print('\nAverage PR-AUC = {:.3f}'.format(PR_AUC_array.mean()))


# ============================================================
# Train the FINAL DNN model
# ============================================================

print("\n********** Training Final DNN Model **********")
DNN_model = build_dnn_model(X_train1.shape[1])
early_stopping_final = EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
DNN_model.fit(X_train1,Y_train1,epochs=100,batch_size=26,callbacks=[early_stopping_final])

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
Y_predict_proba_ind = DNN_model.predict(X_ind_arr).ravel()

# Convert these probabilities into hard 0 or 1 class labels
Y_predict_ind = (Y_predict_proba_ind >= 0.5).astype(int)

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
