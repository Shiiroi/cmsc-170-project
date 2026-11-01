import csv
import numpy as np

X = []   # features
y = []   # labels: 0 = Low, 1 = Middle, 2 = High

with open("table7_cleaned.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # features = every column ending in "_pct" (spending shares, %)
    # area_name and decile are left out: decile is what the label(income_tiers) is made from
    features = []
    for c in reader.fieldnames:
        if c.endswith("_pct"):
            features.append(c)

    for row in reader:
        values = []
        for c in features:
            values.append(float(row[c]))
        X.append(values)
        y.append(int(row["income_tier"]))   #labels = income_tier

X = np.array(X)
y = np.array(y)
print("Data:", X.shape, "features:", len(features))

# Test split
split = 83 * 10
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print("Training rows:", len(X_train), "(83 areas)")
print("Testing rows :", len(X_test), "(36 areas)")