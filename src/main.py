from data_utils import load_data, split_by_area
from metrics import (accuracy_metric, get_precision, get_recall, get_weighted_mean, get_f1_score, get_income_counts)
from naive_bayes import train_nb, predict_nb
from knn import train_knn, predict_knn
from c45_decision_tree import train_dt, predict_dt

# ================= Load, Split ===================
X, y, areas, deciles, features = load_data()
train_idx, test_idx = split_by_area(areas, test_ratio=0.3, seed=42)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Total number of records: {len(X)}")
print(f"Training instances: {len(X_train)}")
print(f"Testing instances : {len(X_test)}")


# ================= Train, Predict ===================
print("Training and classifying...")
predictions = {}

# NB
model = train_nb(X_train, y_train, var_smoothing=1e-9)
predictions["Gaussian Naive Bayes"] = predict_nb(model, X_test)

# KNN
model = train_knn(X_train, y_train, k=5, p=2)
predictions["KNN (k=5, p=2)"] = predict_knn(model, X_test)

# DT
model = train_dt(X_train, y_train, features, min_samples=10, max_depth=10)
predictions["C4.5 Decision Tree"] = predict_dt(model, X_test)

# ================= Evaluate, Compare ===================
counts = get_income_counts(y_test)

print("\nCOMPARISON OF METHODS")
print(f"{'Method':<24}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
for name, pred in predictions.items():
    precision = get_precision(y_test, pred)
    recall = get_recall(y_test, pred)
    f1 = get_f1_score(precision, recall)

    accuracy = accuracy_metric(y_test, pred)
    w_precision = get_weighted_mean(precision, counts)
    w_recall = get_weighted_mean(recall, counts)
    w_f1 = get_weighted_mean(f1, counts)

    print(f"{name:<24}{accuracy:<12f}{w_precision:<12f}{w_recall:<12f}{w_f1:<12f}")