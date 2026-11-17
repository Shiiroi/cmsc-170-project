from prepare_dataset import data_collection, data_cleaning, data_preprocessing
from data_utils import split_data_set
from training import model_training
from evaluation import model_evaluation, print_comparison

# Data Collection
raw_rows = data_collection()

# Data Cleaning
cleaned_rows = data_cleaning(raw_rows)

# Data Preprocessing
dataset = data_preprocessing(cleaned_rows)

# Split Data Set
train_set, test_set = split_data_set(dataset)
print(f"Total number of records: {len(dataset["y"])}")
print(f"Training instances: {len(train_set["y"])}")
print(f"Testing instances : {len(test_set["y"])}")

# Model Training (using train set only)
models = model_training(train_set)

# Model Performance Evaluation (test set)
results = model_evaluation(models, test_set)
print_comparison(results)