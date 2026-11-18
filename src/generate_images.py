import os
from data_utils import load_data, split_by_area
from naive_bayes import train_nb
from knn import train_knn
from c45_decision_tree import train_dt
from evaluation import model_evaluation
from visualization import plot_metric_bar, plot_overall_metrics

def generate():
    X, y, areas, deciles, features = load_data()
    train_idx, test_idx = split_by_area(areas)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    models = {
        "Gaussian Naive Bayes": train_nb(X_train, y_train),
        "KNN": train_knn(X_train, y_train, k=5, p=2),
        "C4.5 Decision Tree": train_dt(X_train, y_train, features, 10, 10)
    }

    test_set = {"X": X_test, "y": y_test}
    results = model_evaluation(models, test_set)

    plot_overall_metrics(results, "docs/assets/figures/overall_metrics.png")
    for metric in ("precision", "recall", "f1"):
        plot_metric_bar(results, metric, f"docs/assets/figures/{metric}_by_tier.png")

generate()