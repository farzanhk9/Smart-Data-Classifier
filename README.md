# A professional data classification script using the Scikit-Learn library.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 2. Initialize and Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. Predict and Evaluate
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
