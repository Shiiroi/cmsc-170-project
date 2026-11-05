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
X, y, areas, deciles, features = load_data()
train_idx, test_idx = split_by_area(areas)

ATTRIBUTES = features

def row_to_dict(values):
    row = {}
    for j in range(len(features)):
        row[features[j]] = values[j]
    return row

train_rows = []
for i in train_idx:
    row = row_to_dict(X[i])
    row["income_tier"] = y[i]
    train_rows.append(row)

test_rows = []
y_test = []          
deciles_test = []    # only for displaying results
for i in test_idx:
    test_rows.append(row_to_dict(X[i]))
    y_test.append(y[i])
    deciles_test.append(deciles[i])

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

    print(f"Total number of records: {len(X)}")
    print(f"Training instances: {len(train_rows)}")
    print(f"Testing instances : {len(test_rows)}")

    min_samples = 10
    max_depth = math.log2(len(ATTRIBUTES))

    print("Classifying test instances...")
    print("C4.5 DECISION TREE CLASSIFICATION RESULTS")
    print(f"min_samples = {min_samples}")
    print(f"max_depth = {max_depth}")

    model = C45DecisionTree(min_samples=min_samples, max_depth=max_depth)
    model.fit(train_rows)

    predictions = model.predict(test_rows)
    print_results(y_test, predictions, deciles_test)

    # print(model.tree)

test()
