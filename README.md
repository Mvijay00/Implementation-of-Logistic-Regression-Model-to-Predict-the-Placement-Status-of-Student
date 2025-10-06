# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
The code trains a logistic regression model using cgpa and iq as features to predict whether a student gets placed or not.

It uses a pipeline that combines feature scaling with StandardScaler and model fitting using LogisticRegression.

The model is evaluated using metrics like accuracy, classification report, confusion matrix, and ROC curve with AUC.

Visualizations are generated for both the confusion matrix and the ROC curve to analyze model performance.

Finally, the model is used to predict placement outcomes and probabilities for a new student's cgpa and iq values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
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
![screenshot(50)](https://github.com/user-attachments/assets/a1a3107b-97c4-49a6-8ae8-cc8adad50a36)
![screenshot(51)](https://github.com/user-attachments/assets/54ba91c5-3133-4dc5-a23a-3650568e09bd)
![screenshot(52)](https://github.com/user-attachments/assets/671ec858-9b9f-4f6c-8f9e-ae3edb434d2c)
![screenshot(53)](https://github.com/user-attachments/assets/2e0ede25-78f7-435e-949c-1130a361c4b7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
