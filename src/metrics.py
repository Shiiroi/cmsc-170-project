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

    # calculate precision, recall, f1 metrics
    precision = get_precision(actual, predicted)
    recall = get_recall(actual, predicted)
    f1 = get_f1_score(precision, recall)
    counts = get_income_counts(actual)

    # calculate weighted means of each metric
    weighted_precision = get_weighted_mean(precision, counts)
    weighted_recall = get_weighted_mean(recall, counts)
    weighted_f1 = get_weighted_mean(f1, counts)

    # display precision and recall table
    print(f"{'Income Tier':<16}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    for class_idx in TIER_NAMES:
        tier = TIER_NAMES[class_idx]
        print(f"{tier:<16}{precision[tier]:<12.4f}{recall[tier]:<12.4f}{f1[tier]:<12.4f}")

    print(f"{'Weighted Mean':<16}{weighted_precision:<12.4f}{weighted_recall:<12.4f}{weighted_f1:<12.4f}")


# formulas (precision recall f1) used are based from:
#   A. Abhang, "Implementing Acc.,Precision,Recall,F1 from scratch," Kaggle,
#   https://www.kaggle.com/code/ajinkyaabhang/implementing-acc-precision-recall-f1-from-scratch

# count true positives for a specific class
def true_positive(y_true, y_pred, target_class):
    tp = 0
    for gt, pred in zip(y_true, y_pred):
        if gt == target_class and pred == target_class:
            tp += 1
    return tp

# count false positives for a specific class
def false_positive(y_true, y_pred, target_class):
    fp = 0
    for gt, pred in zip(y_true, y_pred):
        if gt != target_class and pred == target_class:
            fp += 1
    return fp

# count false negatives for a specific class
def false_negative(y_true, y_pred, target_class):
    fn = 0
    for gt, pred in zip(y_true, y_pred):
        if gt == target_class and pred != target_class:
            fn += 1
    return fn

# precision formula: TP / (TP + FP)
def get_precision(y_true, y_pred):
    # (0: Low-Income, 1: Middle-Income, 2: High-Income)
    precision_dict = {}
    # calculate precision for all inome tiers
    for class_idx, class_name in TIER_NAMES.items():
        tp = true_positive(y_true, y_pred, class_idx)
        fp = false_positive(y_true, y_pred, class_idx)

        if (tp + fp) > 0:
            precision_dict[class_name] = float(tp) / float(tp + fp)
        else:
            precision_dict[class_name] = 0.0

    return precision_dict

# recall formula: TP / (TP + FN)
def get_recall(y_true, y_pred):
    # (0: Low-Income, 1: Middle-Income, 2: High-Income)
    recall_dict = {}
    # calculate recall for all inome tiers
    for class_idx, class_name in TIER_NAMES.items():
        tp = true_positive(y_true, y_pred, class_idx)
        fn = false_negative(y_true, y_pred, class_idx)

        if (tp + fn) > 0:
            recall_dict[class_name] = float(tp) / float(tp + fn)
        else:
            recall_dict[class_name] = 0.0

    return recall_dict

# count instances per tier
def get_income_counts(y_true):
    counts = {name: 0 for name in TIER_NAMES.values()}
    for y in y_true:
        counts[TIER_NAMES[y]] += 1
    return counts

# weighted mean: sum(score_i * instances_i) / total_instances
def get_weighted_mean(scores, counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0

    weighted_sum = sum(scores[tier] * counts[tier] for tier in scores)
    return weighted_sum / float(total)

# f1 raw score: 2*((precision*recall)/(precision+recall))
def get_f1_score(precision_dict, recall_dict):
    f1_dict = {}
    for class_name in TIER_NAMES.values():
        p = precision_dict[class_name]
        r = recall_dict[class_name]

        if (p + r) > 0:
            f1_dict[class_name] = 2.0 * ((p * r ) / (p + r))
        else:
            f1_dict[class_name] = 0.0
    return f1_dict