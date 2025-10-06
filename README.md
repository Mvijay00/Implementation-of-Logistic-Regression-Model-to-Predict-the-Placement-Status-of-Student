# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
The code loads a placement dataset, preprocesses it by dropping irrelevant columns, and label-encodes categorical variables to convert them into numeric format suitable for model training.

It splits the data into training and testing sets, then trains a logistic regression model using the training data to predict the binary "status" (placed or not).

Predictions are made on the test set, and the model's accuracy score is printed, representing the overall proportion of correct predictions among the test samples.

A detailed classification report is generated showing per-class precision (correct positive predictions), recall (coverage of actual positives), and F1-score (balance of precision and recall), giving fuller insight into the model’s performance on each class.

The code includes a sample prediction on a new input example, emphasizing the need to match feature order and values as per the training data, demonstrating the model's practical use.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
*/
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")

# Drop unwanted columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

# Optional: Check for missing values and duplicates
#print(data1.isnull().sum())
#print(data1.duplicated().sum())

# Remove whitespaces in column names to avoid key errors
data1.columns = data1.columns.str.strip()

# Encode categorical columns
le = LabelEncoder()
categorical_columns = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in categorical_columns:
    data1[col] = le.fit_transform(data1[col])

# Split features and target variable
X = data1.iloc[:, :-1]  # all columns except last one
y = data1["status"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train Logistic Regression model
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# Evaluate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report1 = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_report1)

# Predict status for a new input - ensure the order and values match features
new_sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]  # adjust values as necessary
prediction = lr.predict(new_sample)
print("Prediction for new sample:", prediction)

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
