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

import numpy as np
from data_utils import load_data, split_by_area
from metrics import print_results

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

def train_nb(X_train, y_train, var_smoothing=1e-9):
    classes, priors, means, variances = fit(X_train, y_train, var_smoothing)
    return {"classes": classes, "priors": priors, "means": means, "variances": variances, "var_smoothing": var_smoothing}

def predict_nb(model, X_test):
    return predict(X_test, model["classes"], model["priors"], model["means"], model["variances"])


if __name__=="__main__":
    # Load data
    X, y, areas, deciles, features = load_data()

    # Test split
    train_idx, test_idx = split_by_area(areas)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    deciles_test = [deciles[i] for i in test_idx]     # only for displaying results

    print(f"Total number of records: {len(X)}")
    print(f"Training instances: {len(X_train)}")
    print(f"Testing instances : {len(X_test)}")

    # Train
    var_smoothing = 1e-9
    classes, priors, means, variances = fit(X_train, y_train, var_smoothing)

    # Test
    print("Classifying test instances...")
    print("GAUSSIAN NAIVE BAYES CLASSIFICATION RESULTS")
    print(f"var_smoothing = {var_smoothing}")

    pred = predict(X_test, classes, priors, means, variances)
    print_results(y_test, pred, deciles_test)