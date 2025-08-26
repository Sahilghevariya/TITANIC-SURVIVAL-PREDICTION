# Titanic Survival Prediction

# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
print("Dataset Loaded:")
print(data.head())

# Step 2: Data Preprocessing
# Drop irrelevant columns
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical to numerical
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Accuracy:", log_acc)
print(classification_report(y_test, y_pred_log))

# Step 5: Decision Tree Model
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_acc)
print(classification_report(y_test, y_pred_dt))

# Step 6: Confusion Matrices
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
# Step 7: Conclusion
print("\nConclusion:")
print("Both models have been evaluated. Logistic Regression performed with an accuracy of {:.2f}% and Decision Tree with {:.2f}%.".format(log_acc * 100, dt_acc * 100))
print("Further tuning and feature engineering could improve these results.")    
# Used {:.2f}% to format the accuracy as a percentage.
# End of Titanic Survival Prediction Script