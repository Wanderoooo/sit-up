from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(y_train)

    model = SVC(kernel='rbf', C=0.01, gamma=0.01)  # You can try different kernels like 'rbf', 'poly', etc.
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print('Cross-validation scores: ', scores)
    print('Average cross-validation score: ', scores.mean())

    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return model

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Print the accuracy
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return clf

def predict(model, metrics):
    # Use the model to predict the pose
    metrics = np.array(metrics).reshape(1, -1)
    return model.predict(metrics)[0]