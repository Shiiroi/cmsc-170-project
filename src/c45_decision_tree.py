from C45 import C45Classifier
import pandas as pd
import math
import collections
import csv

# W/ pandas already 
#################################################
def C45ImplementationWithPandas():
    tablefile = pd.read_csv("../cmsc-170-project/table7_cleaned.csv")

    X = tablefile.drop(columns=["income_tier", "area_name"])
    y = tablefile["income_tier"]

    model = C45Classifier()
    model.fit(X, y)

    model.evaluate(X, y)
    model.summary()
#################################################
# No Pandas
with open("../cmsc-170-project/table7_cleaned.csv", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = []
    for r in reader:
        row = {}
        for k, v in r.items():
            if k in ("area_name",):
                row[k] = v
            elif k == "income_tier":
                row[k] = int(v)
            else:
                row[k] = float(v)
        rows.append(row)

ATTRIBUTES = ["food_home_pct", "food_outside_pct", "housing_utilities_pct",
            "transport_pct", "infocomm_pct", "education_pct", "health_pct",
            "clothing_pct", "recreation_pct", "occasions_pct"]

X = [{f: r[a] for a in ATTRIBUTES} for r in rows]
y = [r["income_tier"] for r in rows]

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
        for a in attributes:
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
    print("C4.5 Decision Tree Implementation using pandas (as ideal result)")
    C45ImplementationWithPandas()

    print("\n")

    print("C4.5 Decision Tree Implementation")
    model = C45DecisionTree(min_samples=10, max_depth=math.log2(len(ATTRIBUTES)))
    model.fit(rows)

    predictions = model.predict(X)

    accuracy = sum( p == t for p, t in zip(predictions, y)) / len(y)
    print(f"Accuracy: {accuracy:.3f}")

    # print(model.tree)

test()
