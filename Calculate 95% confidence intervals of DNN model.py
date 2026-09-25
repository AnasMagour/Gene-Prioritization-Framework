# Calculate 95% confidence intervals (CIs) for the internal held-out test dataset of DNN model

# ============================================================
# Importing the required libraries
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# Importing the necessary modules
# ============================================================

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc 
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

# ==================================
# Function to build the DNN model
# ==================================

def build_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(6,input_dim=input_dim,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# =======================================================================
# Train Model & Generate Predictions on Held-Out Test Dataset
# =======================================================================

model = build_dnn_model(X_train1.shape[1])
early_stopping_final = EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
model.fit(X_train1,Y_train1,epochs=100,batch_size=26,callbacks=[early_stopping_final])
# Model outputs a 1-column array of probabilities for binary classification
Y_prob1 = model.predict(X_test1).ravel() 
# Convert probabilities to binary class predictions using a 0.5 threshold
Y_pred1 = (Y_prob1 >= 0.5).astype(int)

# =======================================================================
# Bootstrapping Function for 95% Confidence Intervals
# =======================================================================

def compute_bootstrap_cis(y_true, y_pred, y_prob, n_resamples=1000, alpha=0.95):
    """
    Computes 95% confidence intervals for classification metrics on held-out test data.
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
    
    print("\n--- Model Evaluation Metrics with 95% Confidence Intervals (Held-out Test Dataset) ---")
    for metric, scores in bootstrapped_scores.items():
        median = np.median(scores)
        lower_bound = np.percentile(scores, lower_pct)
        upper_bound = np.percentile(scores, upper_pct)
        print(f"{metric:<10}: {median:.3f} (95% CI: {lower_bound:.3f} - {upper_bound:.3f})")

# Execute the evaluation
compute_bootstrap_cis(Y_test1, Y_pred1, Y_prob1)
