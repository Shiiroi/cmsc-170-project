# accuracy_metric() is adapted from:
#   J. Brownlee, "How To Implement Machine Learning Algorithm Performance
#   Metrics From Scratch With Python," Machine Learning Mastery.
#   https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:   # prediction matches the true label
            correct += 1
    return correct / float(len(actual)) * 100.0

TIER_NAMES = {0: "Low-Income", 1: "Middle-Income", 2: "High-Income"}

def print_results(actual, predicted, deciles):
    # Same output format for every method (based on knn)
    incorrect = []
    for i in range(len(actual)):
        if actual[i] != predicted[i]:
            incorrect.append((i + 1, deciles[i], actual[i], predicted[i]))

    print(f"Correct predictions: {len(actual) - len(incorrect)}")
    print(f"Incorrect predictions: {len(incorrect)}")
    print(f"Classification Accuracy: {accuracy_metric(actual, predicted):.2f}%")

    print(f"{'Data Record #':<15}{'Decile':<8}{'Actual':<16}{'Predicted'}")
    for rec_num, decile, a, p in incorrect:
        print(f"{rec_num:<15}{decile:<8}{TIER_NAMES[a]:<16}{TIER_NAMES[p]}")