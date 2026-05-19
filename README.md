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

# Program: Decision Tree Classifier for Employee Churn Prediction

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 2: Create Sample Dataset
data = {
    'satisfaction_level': [0.38,0.80,0.11,0.72,0.37,0.41,0.10,0.92,0.89,0.42],
    'average_monthly_hours': [157,262,272,223,159,153,247,259,224,142],
    'promotion_last_5years': [0,0,0,1,0,0,0,1,0,0],
    'salary': ['low','medium','low','high','low','medium','low','high','medium','low'],
    'left': [1,0,1,0,1,0,1,0,0,1]
}

df = pd.DataFrame(data)

# Step 3: Convert Categorical Data into Numerical Data
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])

# Step 4: Split Features and Target
X = df.drop('left', axis=1)
y = df['left']

# Step 5: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 6: Create Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Step 7: Train the Model
model.fit(X_train, y_train)

# Step 8: Predict the Output
y_pred = model.predict(X_test)

# Step 9: Find Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 10: Display Results
print("Predicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

```

## Output:
<img width="505" height="317" alt="Screenshot 2026-05-18 091127" src="https://github.com/user-attachments/assets/c1afef4a-df73-4363-9288-83f8ca1ca5d8" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
