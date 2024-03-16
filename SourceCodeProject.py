 
# # Load the Data: Use a library like Pandas in Python to load the dataset from the CSV file.
 

import pandas as pd

# Load the CSV file into a DataFrame
file_path = "C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify it was loaded correctly
print(data.head())


# # Exploratory Data Analysis (EDA): Analyze the data to understand its structure, distributions, and relationships between variables.

  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Check dimensions of the dataset
print("Dimensions:", data.shape)

# Summary statistics
print(data.describe())

# Data visualization
# Example: Histogram of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Example: Bar chart of class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data['Class'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Example: Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# # Data Cleaning: Check for missing values, outliers, or inconsistencies in the data and handle them appropriately.

 

import numpy as np  # Add this line to import NumPy

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Remove rows with missing values (if needed)
# data.dropna(inplace=True)

# Handle missing values by imputation (if needed)
# data.fillna(data.mean(), inplace=True)

# Identify outliers using z-score
from scipy import stats
z_scores = stats.zscore(data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
abs_z_scores = np.abs(z_scores)  # Use np.abs() here
outliers = (abs_z_scores > 3).all(axis=1)
print("Number of Outliers:", outliers.sum())

# Handle outliers by removing them (if needed)
# data = data[~outliers]

# Handle outliers by transforming the feature (if needed)
# data['Amount'] = np.log(data['Amount'] + 1)  # Log transformation

# Visualize outliers using box plots (optional)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['Time', 'Amount']])
plt.title('Box Plot of Time and Amount')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.show()


# # Feature Engineering: Create new features or transform existing ones to improve model performance.
 


# Example of feature engineering
import pandas as pd

# Extract time-related features
data['Hour'] = pd.to_datetime(data['Time'], unit='s').dt.hour
data['DayOfWeek'] = pd.to_datetime(data['Time'], unit='s').dt.dayofweek

# Transform amount using logarithmic transformation
data['LogAmount'] = np.log(data['Amount'] + 1)  # Adding 1 to avoid log(0)

# Create interaction features
data['V1_V2'] = data['V1'] * data['V2']

# Calculate aggregate statistics
data['MeanV'] = data[['V1', 'V2', 'V3', 'V4', 'V5']].mean(axis=1)
data['StdV'] = data[['V1', 'V2', 'V3', 'V4', 'V5']].std(axis=1)

# Drop original time and amount columns if needed
# data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Perform feature selection if needed
# ...

# Display the updated dataset with engineered features
print(data.head())


 

# # Train/Test Split: Split the data into training and testing sets for model evaluation.

 


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv("C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv")

# Assuming your target variable is named 'Class'
X = df.drop(columns=['Class'])  # Features (all columns except 'Class')
y = df['Class']                 # Target variable ('Class' column)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the shapes of the training and testing sets
print("Training set - Features:", X_train.shape, "Labels:", y_train.shape)
print("Testing set - Features:", X_test.shape, "Labels:", y_test.shape)


# # Model Selection: Choose appropriate machine learning algorithms for classification (e.g., logistic regression, random forest, support vector machines).
#     ##Logistic Regression algorithm

 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load the CSV data into a pandas DataFrame
data = pd.read_csv("C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv")

# Step 2: Preprocess the data (if needed)
# For example, handle missing values, scale numerical features, encode categorical variables

# Step 3: Split the data into features (X) and target variable (y)
X = data.drop(columns=['Class'])  # Features (all columns except 'Class')
y = data['Class']                 # Target variable ('Class' column)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))


# # Model Training: Train the selected models on the training data.

 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Class'])  # Features (all columns except 'Class')
y = data['Class']                 # Target variable ('Class' column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model with increased max_iter
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Predict on the testing set
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))


# # Hyperparameter Tuning: Fine-tune model parameters to optimize performance.

# 


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the CSV data into a pandas DataFrame
data = pd.read_csv("C:/Users/Deep/Documents/Projects Shivam/Final Project/Project-2-Prediction of Credit Card fraud/creditcard.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Class'])  # Features (all columns except 'Class')
y = data['Class']                 # Target variable ('Class' column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# Initialize Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate the model with best hyperparameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))


 

