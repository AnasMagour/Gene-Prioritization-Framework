# Mutual information for feature selection using python

# Import required libraries
import pandas as pd
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

# Displaying all rows
pd.set_option('display.max_rows', None)

# Loading the dataset (Standardized T1DM dataset)
Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/Data Preparation (T1DM)/T1DM Datasets/Standardized T1DM dataset.xlsx')

# Splitting the dataset into training data and testing data to avaoid overfitting
X = Data.drop(['Class', 'Official gene symbol'], axis = 1) # X is the feature set (independent variables)
Y = Data['Class'] # Y is the target variable (dependent variable): It could be binary 0 or 1 (Corresponding labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0) 

# Displaying the number of genes and features
print("\nStandardized T1DM dataset:\n","Number of genes = ",X.shape[0],"\n","Number of features = ",X.shape[1],"\n")
print("All the features in this dataset have continuous values\n")

# Determine the mutual information
mutual_info = mutual_info_classif(X_train, Y_train)

# Representing the mutual information values in a list form
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns

# Sorting the mutual information values in descending order
MI = mutual_info.sort_values(ascending = False)
print(MI)

# NOTE: The mutual information value will never be negative
# NOTE: The higher the value of the mutual information, the more important the feature



