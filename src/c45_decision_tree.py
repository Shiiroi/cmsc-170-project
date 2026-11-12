# from C45 import C45Classifier
# import pandas as pd
import math
import collections
from data_utils import load_data, split_by_area, DATA_PATH
from metrics import print_results

# W/ pandas already 
#################################################
# def C45ImplementationWithPandas():
#     tablefile = pd.read_csv(DATA_PATH)

#     X = tablefile.drop(columns=["income_tier", "area_name"])
#     y = tablefile["income_tier"]

#     model = C45Classifier()
#     model.fit(X, y)

#     model.evaluate(X, y)
#     model.summary()
#################################################
# No Pandas
ATTRIBUTES = []

def row_to_dict(values, features):
    # [42.4, 17.9, ...] -> {"food_home_pct": 42.4, "food_outside_pct": 17.9, ...}
    row = {}
    for j in range(len(features)):
        row[features[j]] = values[j]
    return row


def train_dt(X_train, y_train, features, min_samples=10, max_depth=10):
    global ATTRIBUTES
    ATTRIBUTES = features

    train_rows = []
    for i in range(len(X_train)):
        row = row_to_dict(X_train[i], features)
        row["income_tier"] = y_train[i]
        train_rows.append(row)

    tree = C45DecisionTree(min_samples=min_samples, max_depth=max_depth)
    tree.fit(train_rows)
    return {"tree": tree, "features": features,
            "min_samples": min_samples, "max_depth": max_depth}


def predict_dt(model, X_test):
    test_rows = []
    for values in X_test:
        test_rows.append(row_to_dict(values, model["features"]))
    return model["tree"].predict(test_rows)


class C45DecisionTree:
    def __init__(self, min_samples, max_depth):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.tree = None

    def fit(self, rows, target = "income_tier"):
        self.tree = self.build(rows, list(ATTRIBUTES), 0, target)
        return self

    def build(self, rows, attributes, depth, target):
        labels = [r[target] for r in rows]
        if len(set(labels)) == 1:
            return ("leaf", labels[0])
        if not attributes or depth >= self.max_depth or len(rows) < self.min_samples:
            return ("leaf", collections.Counter(labels).most_common(1)[0][0])
        best = (None, None, -1.0)
        for f in attributes:
            t, ratio = best_threshold(rows, f)
            if ratio > best[2]:
                best = (f, t, ratio)
        if best[0] is None:
            return ("leaf", collections.Counter(labels).most_common(1)[0][0])

        f, t, _ = best
        left = [r for r in rows if r[f] <= t]
        right = [r for r in rows if r[f] > t]
        return ("node", f, t, self.build(left, attributes, depth + 1, target), self.build(right, attributes, depth + 1, target))

    def predict_one(self, row):
        node = self.tree
        while node[0] == "node":
            _, f, t, left, right = node
            node = left if row[f] <= t else right
        return node[1]

    def predict(self, rows):
        return [self.predict_one(r) for r in rows]
    
def entropy(label):
    n = len(label)
    if n == 0:
        return 0.0
    counts = collections.Counter(label)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def split_rows(rows, attribute, threshold):
    left = [r for r in rows if r[attribute] <= threshold]
    right = [r for r in rows if r[attribute] > threshold]
    return left, right

def best_threshold(rows, attribute):
    vals = sorted({r[attribute] for r in rows})
    best = (None, -1.0)
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2
        left, right = split_rows(rows, attribute, t)
        if not left or not right:
            continue
        gain = entropy([r["income_tier"] for r in rows]) - (len(left)/len(rows))*entropy([r["income_tier"] for r in left]) - (len(right)/len(rows))*entropy([r["income_tier"] for r in right])
        split_info = -sum((len(s)/len(rows))*math.log2(len(s)/len(rows)) for s in (left, right))
        ratio = gain / split_info if split_info else 0
        if ratio > best[1]:
            best = (t, ratio)
    return best

# Just for testing C4.5 Tree w/o pandas Behavior
def test():
    # print("C4.5 Decision Tree Implementation using pandas (as ideal result)")
    # C45ImplementationWithPandas()

    # print("\n")
    X, y, areas, deciles, features = load_data()
    train_idx, test_idx = split_by_area(areas)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    deciles_test = [deciles[i] for i in test_idx]    # only for displaying results

    print(f"Total number of records: {len(X)}")
    print(f"Training instances: {len(X_train)}")
    print(f"Testing instances : {len(X_test)}")

    min_samples = 10
    # max_depth = math.log2(len(ATTRIBUTES))
    max_depth = 10

    print("Classifying test instances...")
    print("C4.5 DECISION TREE CLASSIFICATION RESULTS")
    print(f"min_samples = {min_samples}")
    print(f"max_depth = {max_depth}")

    model = train_dt(X_train, y_train, features, min_samples, max_depth)
    predictions = predict_dt(model, X_test)

    print_results(y_test, predictions, deciles_test)

    # print(model.tree)

if __name__ == "__main__":
    test()

