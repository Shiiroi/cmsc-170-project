import numpy as np
from data_utils import load_data, split_by_area
from metrics import print_results

# minkowski formula
def minkowski(training, testing, p):
  summation = 0
  for i in range(len(testing)):
    summation += abs(training[i] - testing[i]) ** p

  return summation ** (1.0 / p)

# count top vote
def voting(labels):
  # counter
  counts = {0: 0, 1: 0, 2: 0}
  for l in labels:
    counts[l] += 1
  # get max vote and filter income tier equla to it
  max_votes = max(counts.values())
  winners = [cls for cls, count in counts.items() if count == max_votes]

  if len(winners) == 1:
    return winners[0]
  # tie breaker find the closer neighbor 
  for l in labels:
    if l in winners:
      return l

# Load data
X, y, areas, deciles, features = load_data()

# Test split
train_idx, test_idx = split_by_area(areas)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
deciles_test = [deciles[i] for i in test_idx]   # only for displaying results

print(f"Total number of records: {len(X)}")
print(f"Training instances: {len(X_train)}")
print(f"Testing instances :{len(X_test)}")

# get input k and p
isOdd = False
isPos = False
k=0
p=0
while (not isOdd):
  k = int(input("Enter an odd value of k:"))
  if k > 0 and k % 2 != 0:
    isOdd = True

while (not isPos):
  p = float(input("Enter the value of p for Minkowski distance: "))
  if p > 0:
    isPos = True

print("Classifying test instances...")
print("KNN CLASSIFICATION RESULTS")
print(f"k = {k}")
print(f"p = {p}")

predictions = []
for d in X_test:
  distances = []
  for dt,wt in zip(X_train, y_train):
    dist = minkowski(dt, d, p)
    distances.append((dist, wt))

  distances = sorted(
      distances,
      key=lambda x: x[0],
      # reverse=True
  )

  topk = distances[:k]

  pred = []
  for (dp, wp) in topk:
    pred.append(wp)

  prediction = voting(pred)
  predictions.append(prediction)

print_results(y_test, predictions, deciles_test)
