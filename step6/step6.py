import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load and normalize data
X_train = pd.read_csv('step5/one_hot_encoding.csv', sep=",")
y = pd.read_csv('step5/labels.csv', sep=",")  # Assuming labels are in a separate file

X_train = normalize(X_train.values)

# Split data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_train, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# Train and evaluate models
best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train_rf, y_train_rf)
    predictions_test = model.predict(X_test_rf)
    accuracy = accuracy_score(y_test_rf, predictions_test)
    precision = precision_score(y_test_rf, predictions_test, average='weighted')
    recall = recall_score(y_test_rf, predictions_test, average='weighted')
    f1 = f1_score(y_test_rf, predictions_test, average='weighted')
    
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Export the best model
joblib.dump(best_model, 'best_model.pkl')