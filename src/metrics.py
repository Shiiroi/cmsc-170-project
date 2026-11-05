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