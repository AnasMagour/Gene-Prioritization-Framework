# Pearson correlation for feature selection using python

# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split

# Loading the dataset (Standardized T1DM dataset)
Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/Data Preparation (T1DM)/Data Preprocess (T1DM)/3 Feature Selection (T1DM)/1 2 Pearson Correlation (T1DM)/PC Dataset (T1DM)/Standardized T1DM dataset after substage 1.xlsx')
dataset = Data.drop(['Official gene symbol'], axis = 1)

# Splitting the dataset into training data and testing data to avaoid overfitting
X = Data.drop(['Class', 'Official gene symbol'], axis = 1) # X is the feature set (independent variables)
Y = Data['Class'] # Y is the target variable (dependent variable): It could be binary 0 or 1 (Corresponding labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0) 

# Displaying the number of genes and features
print("\nStandardized T1DM dataset:\n","Number of genes = ",X.shape[0],"\n","Number of features = ",X.shape[1],"\n")
print("All the features in this dataset have continuous values\n")

# Displaying the Pearson correlation matrix
plt.figure(figsize=(10,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap = "viridis",annot_kws={"size":8}) # This heatmap shows the correlation of all the features of the dataset
plt.show()

# Part 1: (feature-feature correlation (FFC))
cor_FFC = abs(X_train.corr()) # Note: abs (absolute) function was used to convert a negative correlation into a positive correlation
# Note: We can't do the same for the test data also to avoid overfitting
# Selecting the highly correlated features with each other
corr_features = cor_FFC[cor_FFC>0.8] # Setting the threshold value for correlation
# Exporting the highly correlated features (corr_features) to an excel file
# NOTE: Before exporting the highly correlated features (corr_features), a DataFrame will be created to capture this information in python
# The following syntax is used to create the DataFrame
df = pd.DataFrame(corr_features)
# Define the path where you would like to store the exported excel file
df.to_excel(r'C:/Users/dell/Downloads/Highly correlated features output (0.8) (T1DM).xlsx', index=False)
# NOTE1: "r" character is placed before the path to avoid unicode error
# NOTE2: If you wish to include the index, then simply remove "index=False” from your code
# NOTE3: Finally, run the above code in python, and you will notice that a new excel file (called Output) would be created at the location that you specified (Downloads)

# Part 2: (feature-class correlation (FCC)) (Correlation with output (class or target) variable)
cor_target = abs(cor["Class"])  # Note: abs (absolute) function was used to convert a negative correlation into a positive correlation
# Selecting the highly correlated features with (class or target) variable
relevant_features = cor_target[cor_target>0]
print(relevant_features)
