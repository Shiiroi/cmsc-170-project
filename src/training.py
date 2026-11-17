from naive_bayes import train_nb
from knn import train_knn
from c45_decision_tree import train_dt

def model_training(train_set):
    X, y, features = train_set["X"], train_set["y"], train_set["features"]
    return {
        "Gaussian Naive Bayes": train_nb(X, y, var_smoothing=1e-9),
        "KNN": train_knn(X, y, k=5, p=2),
        "C4.5 Decision Tree": train_dt(X, y, features, min_samples=10, max_depth=10)
    }