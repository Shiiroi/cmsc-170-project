import csv
import numpy as np
import random
import os

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "data", "processed", "table7_cleaned.csv")

raw_data = []

with open(path, "r", encoding="utf-8") as csvFile:
  csvReader = csv.reader(csvFile)
  col_names = next(csvReader)
  for row in csvReader:
    if not row:
      continue
    area = row[0]
    decile = int(row[1])
    # 10 expenditure percentage features (cols 2 to 11)
    data = [float(row[i]) for i in range(2, 12)]
    convertedArray = np.array(data)
    # income_tier (0: Low-Income, 1: Middle-Income, 2: High-Income)
    income_tier = int(row[12])
    raw_data.append((convertedArray, income_tier, area, decile))


# 83 places traning 36 places testing (TODO: UPDATE TO RANDOMIZED?)
train_set =  830 
test_set =  360

training_data = raw_data[:train_set]
testing_data = raw_data[train_set:train_set + test_set]

print(f"Total number of records: {len(raw_data)}")
print(f"Training instances: {len(training_data)}")
print(f"Testing instances :{len(testing_data)}")

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
for (d, w, a, dec) in testing_data:
  distances = []
  for (dt, wt, at, dect) in training_data:
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

# count correct
correct = 0
incorrect = []
incorrect_info = []

for i in range(len(testing_data)):
  if predictions[i] == testing_data[i][1]:
    correct += 1
  else:
    incorrect.append((testing_data[i][1], predictions[i]))
    incorrect_info.append((i + 1, testing_data[i][3]))

# accuracy
accuracy = correct / len(testing_data)
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {len(incorrect)}")
print(f"Classification Accuracy: {accuracy*100:.2f}%")

print("Data Record #\t\t Decile \t Actual \t Predicted")

# map class numbers to the income tier 
tier_names = {0: "Low-Income", 1: "Middle-Income", 2: "High-Income"}

curr_idx = 0
for (a, p_val) in incorrect:
  rec_num, dec_val = incorrect_info[curr_idx]
  print(f"{rec_num}\t{dec_val}\t{tier_names[a]}\t{tier_names[p_val]}")
  curr_idx += 1
