from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def train_model():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    
    # Save model if performance is satisfactory
    if accuracy > 0.9:  # Threshold can be adjusted based on requirements
        joblib.dump(clf, 'model.joblib')
        return accuracy, True
    else:
        return accuracy, False
