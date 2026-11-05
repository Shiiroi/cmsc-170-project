import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "processed", "table7_cleaned.csv")

def load_data(path=DATA_PATH):
    # Returns:
    #   X           feature values  
    #   y           labels  0 = Low, 1 = Middle, 2 = High 
    #   areas       area name of each row (used for splitting)
    #   deciles     decile of each row (used for displaying results, income_tier is made from it)
    #   features    names of the feature columns (for displaying results)
    X, y, areas, deciles = [], [], [], []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # features = every column ending in "_pct", found by name incase we adjust the number of features in the future
        features = [c for c in reader.fieldnames if c.endswith("_pct")]

        for row in reader:
            X.append([float(row[c]) for c in features])
            y.append(int(row["income_tier"]))
            areas.append(row["area_name"])
            deciles.append(int(row["decile"]))
    
    return np.array(X), np.array(y), areas, deciles, features

def split_by_area(areas, n_train_areas=83):
    # Returns:
    #   train_idx   row indices of train rows 
    #   test_idx    row indices of test rows

    # **note:   test split for now is first 83(train data), remaining 36(test) [todo: randomize later]

    unique_areas = list(dict.fromkeys(areas))
    train_areas = set(unique_areas[:n_train_areas])

    train_idx = [i for i, a in enumerate(areas) if a in train_areas]
    test_idx = [i for i, a in enumerate(areas) if a not in train_areas]
    return np.array(train_idx), np.array(test_idx)    