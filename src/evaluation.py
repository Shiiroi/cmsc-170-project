from naive_bayes import predict_nb
from knn import predict_knn
from c45_decision_tree import predict_dt
from metrics import (accuracy_metric, get_precision, get_recall, get_f1_score,
                     get_income_counts, get_weighted_mean)

PREDICT = {
    "Gaussian Naive Bayes": predict_nb,
    "KNN": predict_knn,
    "C4.5 Decision Tree": predict_dt,
}

def model_evaluation(models, test_set):
    y_test = test_set["y"]
    counts = get_income_counts(y_test)
    results = {}

    for name, model in models.items():
        pred = PREDICT[name](model, test_set["X"])

        precision = get_precision(y_test, pred)
        recall = get_recall(y_test, pred)
        f1 = get_f1_score(precision, recall)

        results[name] = {
            "predictions": pred,
            "accuracy": accuracy_metric(y_test, pred),
            "precision": get_weighted_mean(precision, counts),
            "recall": get_weighted_mean(recall, counts),
            "f1": get_weighted_mean(f1, counts),
        }
    return results

def print_comparison(results):
    print("\nCOMPARISON OF METHODS (test set)")
    print(f"{'Method':<24}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    for name, r in results.items():
        print(f"{name:<24}{r['accuracy']:<12f}{r['precision']:<12f}{r['recall']:<12f}{r['f1']:<12f}")

