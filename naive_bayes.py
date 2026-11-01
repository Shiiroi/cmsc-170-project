# naive_nayes.py - Gaussian Naive Bayes variant

# Reference (structure adapted from, MIT License):
#     D. Oniani, "ai" repository,
#     https://github.com/oniani/ai/blob/main/model/ml/gaussian_naive_bayes.py

# Our changes to the reference:
#     1. Priors are computed once in fit() (the reference recomputes them for every row inside predict()).
#     2. Variance smoothin is added once to the variances in fit() (the reference adds them inside the likelihood()).
#     3. The log of the Gaussian is computed directly (the reference computes the Gaussian first, then takes its log, which can give log(0) = -inf).

# Gaussian NB is the variant of Naive Bayes for continuous features.
#     https://medium.com/data-science/gaussian-naive-bayes-explained-a-visual-guide-with-code-examples-for-beginners-04949cef383c

# NB in the lab counts words, but since our features are percentages, we cannot
# "count" it, so each feature in each class is modelled as a bell curve,
# and its height at the value is used as P(x | class).

import csv
import numpy as np

def fit(X, y, var_smoothing=1e-9):
    classes = np.unique(y)                              # [0,1,2] income_tiers

    #amount added to every variance so it is never 0 (laplace counterpart for Gaussian NB)
    #https://medium.com/data-science/gaussian-naive-bayes-explained-a-visual-guide-with-code-examples-for-beginners-04949cef383c
    smoothing = var_smoothing

    priors = []
    means = []
    variances = []

    for c in classes:
        Xc = X[y==c]                                    # only the rows of class c(income_tier)
        priors.append(len(Xc)/len(X))                   # P(class)
        means.append(Xc.mean(axis=0))                   # mean of each feature
        variances.append(Xc.var(axis=0) + smoothing)    # variance of each feature
    
    return classes, np.array(priors), np.array(means), np.array(variances)


def log_likelihood(x, mean, var):
    # Gaussian formula:  P(x|class) = coeff * exponent
    #   coeff    = 1 / sqrt(2*pi*var)
    #   exponent = exp(-(x - mean)^2 / (2*var))
    # https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

    # note: here we return the LOG of P(x|class). Since log(a * b) = log(a) + log(b)
    # the coeff * exponent in the formula becomes log_coeff + log_exponent.

    # compared to the reference code, which computes the gaussian first and then
    # takes its log (log(gaussian)), here we compute the log directly, so a tiny
    # probability can never round down to 0 and give log(0) = -inf. results are same, just added extra precautions

    log_coeff = -0.5 * np.log(2 * np.pi * var)
    log_exponent = -((x-mean)**2) / (2 * var)
    return log_coeff + log_exponent

def predict(X, classes, priors, means, variances):
    predictions = []

    for row in X:
        scores = []
        for i in range(len(classes)):
            # score = log P(class) + sum of log P(feature | class)
            log_prior = np.log(priors[i])

            # naive assumption: each feature judged separately, the added up (independence)
            log_likelihoods = log_likelihood(row, means[i], variances[i])
            
            # posterior: prior * likelihood
            scores.append(log_prior + np.sum(log_likelihoods))

        # Store the label with the largest posterior probability
        predictions.append(classes[np.argmax(scores)])
    
    return np.array(predictions) 


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

# Train
classes, priors, means, variances = fit(X_train, y_train, var_smoothing=1e-9)

print("Classes:", classes)
print("Priors :", priors.round(4))
print("food_home_pct mean per class:", means[:, 0].round(2))

# Likelihood
for c in classes:
    ll = log_likelihood(42.4, means[c, 0], variances[c, 0])
    print(f"class {c}: log-likelihood = {ll:.4f}   likelihood = {np.exp(ll):.5f}")

# Test
pred = predict(X_test, classes, priors, means, variances)

correct = np.sum(pred == y_test)          
accuracy = correct / len(y_test)
print("Correct:", correct, "/", len(y_test))
print("Accuracy:", round(float(accuracy), 4))