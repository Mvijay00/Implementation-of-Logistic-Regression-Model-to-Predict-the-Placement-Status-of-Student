# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
-The code loads a dataset from "placement.csv" with features cgpa and iq and target placement, then splits it into training and testing sets using an 80-20 ratio.

-A machine learning pipeline is created with data scaling (StandardScaler) followed by logistic regression (LogisticRegression with 'liblinear' solver) to model the binary placement outcome.

-The model is trained on training data, then evaluated on the test data by calculating accuracy, generating a classification report, and plotting a confusion matrix and ROC curve to assess performance.

-It extracts and prints the logistic regression coefficients as feature importance, showing the influence of cgpa and iq on placement prediction.

-Finally, the pipeline predicts placement and associated probability for a new student with given cgpa and iq values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:25017872
*/
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)

# Assuming data1 already has your dataset loaded with cgpa, iq, and placement fields
data1=pd.read_csv('placement.csv')
# Split features and target
X = data1[['cgpa', 'iq']]  # Using only the features you mentioned
y = data1["placement"]  # Assuming "placement" is your target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a pipeline with scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature importance
coefs = pd.DataFrame(
    pipeline.named_steps['model'].coef_.T,
    index=X.columns,
    columns=['Coefficient']
).sort_values(by='Coefficient', ascending=False)
print("\nFeature Importance:")
print(coefs)

# Make a prediction for a new student
new_student = [[8.5, 120]]  # Example values for cgpa and iq
prediction = pipeline.predict(new_student)
prediction_prob = pipeline.predict_proba(new_student)
print(f"\nPrediction for new student: {'Placed' if prediction[0] == 1 else 'Not Placed'}")
print(f"Probability of placement: {prediction_prob[0][1]:.4f}")

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="1033" height="301" alt="Screenshot 2025-10-07 103250" src="https://github.com/user-attachments/assets/933b27e1-24ee-457a-8990-17864e62f733" />
<img width="1312" height="716" alt="Screenshot 2025-10-07 103307" src="https://github.com/user-attachments/assets/37fdd1a2-7c66-4b62-ae9b-ccbc706a12b7" />
<img width="1437" height="700" alt="Screenshot 2025-10-07 103324" src="https://github.com/user-attachments/assets/0d5f319c-a529-4336-b214-6ebc8d57b4c9" />
<img width="826" height="205" alt="Screenshot 2025-10-07 103345" src="https://github.com/user-attachments/assets/38066b67-1ba4-4646-b042-6c380d26df2e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
