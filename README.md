# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the dataset.
2. Preprocess the data and split it into training and testing sets.
3. Train the Decision Tree Classifier model using training data.
4. Predict the test data and evaluate using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Iswarya S 
RegisterNumber: 212225040135
*/

# Implementation of Decision Tree Classifier Model
# for Predicting Employee Churn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree

# Dataset
data = {
    'Age': [25, 30, 45, 35, 40, 23, 50, 28, 32, 48],
    'Salary': [30000, 40000, 80000, 50000, 65000, 28000, 90000, 35000, 45000, 85000],
    'YearsAtCompany': [1, 3, 10, 5, 7, 1, 15, 2, 4, 12],
    'Churn': [1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Age', 'Salary', 'YearsAtCompany']]
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Model
model = DecisionTreeClassifier(max_depth=3)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Predicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(18, 10))

tree.plot_tree(
    model,
    feature_names=['Age', 'Salary', 'YearsAtCompany'],
    class_names=['Stayed', 'Left'],
    filled=True,
    rounded=True,
    fontsize=14
)

plt.title("Decision Tree for Employee Churn Prediction", fontsize=20)
plt.show()
```

## Output:
<img width="505" height="317" alt="Screenshot 2026-05-18 091127" src="https://github.com/user-attachments/assets/c1afef4a-df73-4363-9288-83f8ca1ca5d8" />
<img width="740" height="550" alt="image" src="https://github.com/user-attachments/assets/b814b7f3-5d69-4b54-a46e-e48940ce6aca" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
