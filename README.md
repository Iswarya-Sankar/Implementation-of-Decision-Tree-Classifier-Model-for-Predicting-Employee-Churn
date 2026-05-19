# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the required libraries such as Pandas, Matplotlib, and Scikit-learn.
2. Create or load the employee dataset containing features like:
   Age
   Salary
   Years at Company
   Churn status
4.Separate the dataset into:
    Input features (X)
    Target variable (y)
5. Split the dataset into training data and testing data using train_test_split().
6. Create the Decision Tree Classifier model using DecisionTreeClassifier().
7. Train the classifier model using the training dataset with fit() method.
8. Predict employee churn for the testing dataset using predict() method.
9. Evaluate the performance of the model using:
    Accuracy Score
    Confusion Matrix
    Classification Report
10 Plot the Decision Tree using plot_tree() for visualization.
   Display the output and stop the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Iswarya S 
RegisterNumber: 212225040135
*/

#import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Employee.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
# 'left' is the target column (Employee Churn)
X = data.drop("left", axis=1)
y = data["left"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree Classifier model
model = DecisionTreeClassifier(
    criterion='entropy',   # You can also use 'gini'
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict using test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of Decision Tree Classifier:")
print(accuracy)

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(20,10))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Stayed", "Left"],
    filled=True
)

plt.title("Decision Tree for Employee Churn Prediction")
plt.show()

```

## Output:
<img width="554" height="657" alt="Screenshot 2026-05-19 143730" src="https://github.com/user-attachments/assets/1775b595-e570-4126-9346-13951a7b95e8" />
<img width="486" height="262" alt="Screenshot 2026-05-19 143740" src="https://github.com/user-attachments/assets/011efac5-8ca5-4cdb-8cc8-419aebf9d00e" />
<img width="1074" height="518" alt="Screenshot 2026-05-19 143807" src="https://github.com/user-attachments/assets/642f59ac-e1e4-49d4-a50a-51ff76d9bf4d" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
