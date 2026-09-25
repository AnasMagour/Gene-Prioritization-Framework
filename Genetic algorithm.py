# Genetic algorithm for feature selection using python

# Import the required libraries
import pandas as pd
import numpy as np 

# Import the necessary modules
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC # Support Vector Machine Classifier
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.metrics import accuracy_score

# Display all rows
pd.set_option('display.max_rows', None)

# Load the used dataset (Standardized T1DM dataset after stage 2)
Data = pd.read_excel('E:/Anas PhD Thesis/A Type 1 Diabetes Mellitus (T1DM)/Data Preparation (T1DM)/Data Preprocess (T1DM)/3 Feature Selection (T1DM)/2 Genetic Algorithm (T1DM)/GA Dataset (T1DM)/Standardized T1DM dataset after stage 1.xlsx')

# Split the used dataset into training dataset and testing dataset to avaoid overfitting
X = Data.drop(['Class', 'Official gene symbol'], axis = 1) # X is the feature set (independent variables)
Y = Data['Class'] # Y is the target variable (dependent variable): It could be binary 0 or 1 (Corresponding labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Display the number of genes (instances) and features
print("\nStandardized T1DM dataset:\n","Number of genes = ",X.shape[0],"\n","Number of features = ",X.shape[1],"\n")
print("All the features in this dataset have continuous values\n") 

# Build the used genetic algorithm

classifier = SVC() # Specify the classifier used as the estimator in GA_model (The estimator we're going to use is a support vector machine (SVM))
CV = StratifiedKFold(n_splits=3, shuffle=True) # Cross-validation strategy (int)

# Define the GAFeatureSelectionCV options
GA_model = GAFeatureSelectionCV(
    estimator = classifier, # Estimator object implementing ‘fit’ (The object to use to fit the data)
    cv = CV, # Cross-validation generator (If the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used)
    scoring ="accuracy",
    population_size = 30, # Size of the initial population to sample randomly generated individuals
    generations = 20, # Number of generations or iterations to run the evolutionary algorithm
    crossover_probability = 0.7, # Probability of crossover operation between two individuals
    mutation_probability = 0.1, # Probability of child mutation
    tournament_size = 2, # Number of individuals to perform tournament selection (only used if elitism = True)
    elitism = True, # If True takes the tournament_size best solution to the next generation. While if False, it uses a roulette selection mechanism
    n_jobs = -1, # Number of jobs to run in parallel (-1 means using all processors)
    verbose = True, # If True, it displays some metrics over the optimization while the algorithm runs
    keep_top_k = 2 # Number of best solutions to keep in the hof object (If a callback stops the algorithm before k iterations, it will return only one set of parameters per iteration)(This parameter determines the size of the hof)
    )

# Fit the GA_model to select the features
GA_model.fit(X_train, Y_train)

# Export the selected features using the genetic algorithm to an excel file
features = GA_model.support_
features_names = X.columns[features]
df = pd.DataFrame(features_names) # DataFrame will be created to capture this information in python
df.to_excel(r'C:/Users/dell/Downloads/Selected features by GA (T1DM).xlsx', index=False) # Define the path where you would like to store the exported excel file

# NOTE1: "r" character is placed before the path to avoid unicode error
# NOTE2: If you wish to include the index, then simply remove "index=False” from your code
# NOTE3: Finally, run the above code in python, and you will notice that a new excel file (called Output) would be created at the location that you specified (Downloads)

# Evaluate the selected features using the genetic algorithm by calculating the accuracy score in the testing dataset
X_test_ga = X_test.values
Y_predict_ga = GA_model.predict(X_test_ga) # Predict only with the selected features using the genetic algorithm
Accuracy = accuracy_score(Y_test, Y_predict_ga) # Calculate the accuracy score
print("\nAccuracy score = ", "{:.5f}". format(Accuracy)) # Print the accuracy score


