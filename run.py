import os
import sys
import pickle

import numpy as np

from data import (
    load_data,
    manual_preprocess,
    split_numerical_categorical,
    remove_nans,
    remove_correlated_features_by_nan,
    impute,
    standardize,
    select_cat_features_predictive,
    one_hot_encode_train_test_with_ids,
    save_features,
    transport_AGEG5YR,
    create_csv_submission,
)
from logistic_regression import (
    build_model_data,
    augmentation,
    cross_validate,
    reg_weighted_logistic_regression,
    stable_sigmoid,
)
from random_forest import (
    RandomForest,
    tune_threshold,
    split_train_val,
)
from metrics import f1_score

print("""
This script\n
1. Pre-processes the data (might take time), \n
2. Trains logistic regression (and optionally random forest - uncomment/comment lines starting with RANDOM FOREST in the file), and \n
3. Generates predictions \n
""")

# RANDOM FOREST: SET to TRUE if you want to train random forest
# train_random_forest = True if inp == "y" else False

# --------------------------------------------------------------
# Preprocess the data
# --------------------------------------------------------------

np.set_printoptions(suppress=True)

print("\nLoading data:")
features, row_ids, ids_map = load_data("dataset/x_train.csv")
test_features, test_row_ids, test_ids_map = load_data("dataset/x_test.csv")

labels = np.genfromtxt("dataset/y_train.csv", delimiter=",")

print("Original train features shape:", features.shape)
print("Original test features shape:", test_features.shape)
print("Original train labels shape", labels.shape)

print("\nManually preprocessing data:")

features = manual_preprocess(features)
test_features = manual_preprocess(test_features)

print("Shape of manually preprocessed training features:", features.shape)
print("Shape of manually preprocessed testing features:", test_features.shape)

print("\nSplitting numerical and categorical features and adding indicators:")

(
    numerical_features,
    categorical_features,
    test_numerical_features,
    test_categorical_features,
) = split_numerical_categorical(features, test_features)

print("Shape of numerical training features:", numerical_features.shape)
print("Shape of categorical training features:", categorical_features.shape)
print("Shape of numerical testing features:", test_numerical_features.shape)
print("Shape of categorical testing features:", test_categorical_features.shape)

print("\nRemoving features with >60% NaN values:")

numerical_features_clean, test_numerical_features_clean = remove_nans(
    numerical_features, test_numerical_features, 60
)
categorical_features_clean, test_categorical_features_clean = remove_nans(
    categorical_features, test_categorical_features, 60
)

print("Shape of cleaned numerical training features:", numerical_features_clean.shape)
print(
    "Shape of cleaned categorical training features:",
    categorical_features_clean.shape,
)
print(
    "Shape of cleaned numerical testing features:",
    test_numerical_features_clean.shape,
)
print(
    "Shape of cleaned categorical testing features:",
    test_categorical_features_clean.shape,
)

print("\nRemoving correlated numerical features:")

print(f"Original number of features: {len(numerical_features_clean[0, :])}")
_, kept_num_cols = remove_correlated_features_by_nan(
    numerical_features_clean[1:, :], threshold=0.8
)
numerical_features_clean = numerical_features_clean[:, kept_num_cols]
test_numerical_features_clean = test_numerical_features_clean[:, kept_num_cols]
print(f"Number of features kept: {len(kept_num_cols)}")

print("\nImputing NaN values...")

numerical_features_clean[1:, :], test_numerical_features_clean[1:, :] = impute(
    numerical_features_clean[1:, :], test_numerical_features_clean[1:, :]
)

# Ensure there are no NaN values in features, labels and test_features
nan_rows = np.isnan(numerical_features_clean).any(axis=1)
nan_rows_test = np.isnan(test_numerical_features_clean).any(axis=1)

assert np.sum(nan_rows) == 0
assert np.sum(nan_rows_test) == 0

# Ensure there are no inf values in features, labels and test_features
inf_rows = np.isinf(numerical_features_clean).any(axis=1)
inf_rows_test = np.isinf(test_numerical_features_clean).any(axis=1)

assert np.sum(inf_rows) == 0
assert np.sum(inf_rows_test) == 0

print("\nNormalising numerical features:")

# features statistics
print("Numerical features statistics (before normalization):")
print("min mean: ", np.min(np.mean(numerical_features_clean[1:, :], axis=0)))
print("min std: ", np.min(np.std(numerical_features_clean[1:, :], axis=0)))
print("max std: ", np.max(np.std(numerical_features_clean[1:, :], axis=0)))
print("max: ", np.max(numerical_features_clean[1:, :]))
print("min: ", np.min(numerical_features_clean[1:, :]))

# test_features statistics
print("Test numerical features statistics (before normalization):")
print("min mean: ", np.min(np.mean(test_numerical_features_clean[1:, :], axis=0)))
print("min std: ", np.min(np.std(test_numerical_features_clean[1:, :], axis=0)))
print("max std: ", np.max(np.std(test_numerical_features_clean[1:, :], axis=0)))
print("max:", np.max(test_numerical_features_clean[1:, :]))
print("min:", np.min(test_numerical_features_clean[1:, :]))

numerical_features_clean[1:, :], test_numerical_features_clean[1:, :] = standardize(
    numerical_features_clean[1:, :], test_numerical_features_clean[1:, :]
)

# features statistics
print("Numerical features statistics (after normalization):")
print("min mean: ", np.min(np.mean(numerical_features_clean[1:, :], axis=0)))
print("min std: ", np.min(np.std(numerical_features_clean[1:, :], axis=0)))
print("max std: ", np.max(np.std(numerical_features_clean[1:, :], axis=0)))
print("max: ", np.max(numerical_features_clean[1:, :]))
print("min: ", np.min(numerical_features_clean[1:, :]))

# test_features statistics
print("Test numerical features statistics (after normalization):")
print("min mean: ", np.min(np.mean(test_numerical_features_clean[1:, :], axis=0)))
print("min std: ", np.min(np.std(test_numerical_features_clean[1:, :], axis=0)))
print("max std: ", np.max(np.std(test_numerical_features_clean[1:, :], axis=0)))
print("max:", np.max(test_numerical_features_clean[1:, :]))
print("min:", np.min(test_numerical_features_clean[1:, :]))

print("\nSelecting categorical features:")

print(f"Original number of features: {len(categorical_features_clean[0, :])}")

_, kept_cat_cols = select_cat_features_predictive(
    features=categorical_features_clean[1:, :],
    y_target=labels[1:, 1],
    predictiveness_threshold=0.1,
    association_threshold=0.8,
)

print(f"Number of features kept: {len(kept_cat_cols)}")

categorical_features_clean = categorical_features_clean[:, kept_cat_cols]
test_categorical_features_clean = test_categorical_features_clean[:, kept_cat_cols]

print("\nRemoving NaNs in categorical features by replacing with -1...")

categorical_features_clean[1:, :] = np.nan_to_num(
    categorical_features_clean[1:, :], nan=-1
)
test_categorical_features_clean[1:, :] = np.nan_to_num(
    test_categorical_features_clean[1:, :], nan=-1
)

# Ensure there are no NaN values in features, y and test_features
nan_rows = np.isnan(categorical_features_clean).any(axis=1)
nan_rows_test = np.isnan(test_categorical_features_clean).any(axis=1)

assert np.sum(nan_rows) == 0
assert np.sum(nan_rows_test) == 0

# Ensure there are no inf values in features, y and test_features
inf_rows = np.isinf(categorical_features_clean).any(axis=1)
inf_rows_test = np.isinf(test_categorical_features_clean).any(axis=1)

assert np.sum(inf_rows) == 0
assert np.sum(inf_rows_test) == 0

# RANDOM FOREST: Treat the categorical variable _AGEG5YR as a numerical one in order to not one-hot-encode it
# USE THIS LINE ONLY WITH THE RANDOM FOREST MODEL
# if train_random_forest:
#     (
#         categorical_features_clean,
#         test_categorical_features_clean,
#         numerical_features_clean,
#         test_numerical_features_clean,
#     ) = transport_AGEG5YR(
#         categorical_features_clean,
#         test_categorical_features_clean,
#         numerical_features_clean,
#         test_numerical_features_clean,
#     )

print("\nOne-hot encoding categorical features:")

(
    categorical_features_clean,
    test_categorical_features_clean,
    categories_per_feature,
) = one_hot_encode_train_test_with_ids(
    categorical_features_clean, test_categorical_features_clean, handle_nan=False
)

print(
    "Final shape of categorical training features after one-hot encoding:",
    categorical_features_clean.shape,
)
print(
    "Final shape of categorical testing features after one-hot encoding:",
    test_categorical_features_clean.shape,
)

print("\nCombining numerical and categorical features:")

features_clean = np.concatenate(
    (numerical_features_clean, categorical_features_clean), axis=1
)
test_features_clean = np.concatenate(
    (test_numerical_features_clean, test_categorical_features_clean), axis=1
)

print("Final shape of training features:", features_clean.shape)
print("Final shape of testing features:", test_features_clean.shape)

print("\nSaving processed features to disk:")

if not os.path.exists("dataset_processed"):
    os.mkdir("dataset_processed")

save_features(features_clean, ids_map, row_ids, "dataset_processed/x_train_clean.csv")
save_features(
    test_features_clean,
    test_ids_map,
    test_row_ids,
    "dataset_processed/x_test_clean.csv",
)

# --------------------------------------------------------------
# Train the best model
# --------------------------------------------------------------

print("\nLoading data...")
y = np.genfromtxt(
    "dataset/y_train.csv",
    delimiter=",",
    skip_header=1,
    dtype=int,
    usecols=1,
)
X = np.genfromtxt("dataset_processed/x_train_clean.csv", delimiter=",", skip_header=1)
X_test = np.genfromtxt(
    "dataset_processed/x_test_clean.csv", delimiter=",", skip_header=1
)

train_ids = X[:, 0].astype(dtype=int)
test_ids = X_test[:, 0].astype(dtype=int)
X = X[:, 1:]
X_test = X_test[:, 1:]

### # RANDOM FOREST: comment when training random forest
print("\nAugmenting features:")
X_num = X[:, 0:60]
X_test_num = X_test[:, 0:60]

X_other = X[:, 60:]
X_test_other = X_test[:, 60:]

print("Shape of training features before augmenting:", X.shape)
print("Shape of testing features before augmenting:", X_test.shape)

X_num_aug, X_test_num_aug = augmentation(X_num, X_test_num, M=2, num_interactions=500)

X = np.concatenate((X_num_aug, X_other), axis=1)
X_test = np.concatenate((X_test_num_aug, X_test_other), axis=1)

print("Shape of training features after augmenting:", X.shape)
print("Shape of testing features after augmenting:", X_test.shape)
###

# Ensure there are no NaN values in X, y and X_test
nan_rows_X = np.isnan(X).any(axis=1)
nan_rows_y = np.isnan(y).any()
nan_rows_X_test = np.isnan(X_test).any(axis=1)

assert np.sum(nan_rows_X) == 0
assert np.sum(nan_rows_y) == 0
assert np.sum(nan_rows_X_test) == 0

# Ensure there are no inf values in X, y and X_test
inf_rows_X = np.isinf(X).any(axis=1)
inf_rows_y = np.isinf(y).any()
inf_rows_X_test = np.isinf(X_test).any(axis=1)

assert np.sum(inf_rows_X) == 0
assert np.sum(inf_rows_y) == 0
assert np.sum(inf_rows_X_test) == 0

# Check data types
print("\nData information:")
print("Data type:")
print("X: ", X.dtype)
print("X_train: ", X.dtype)
print("y: ", y.dtype)

# X statistics
print("Training feature statistics (numerical features only):")
print("min mean: ", np.min(np.mean(X[:, 0:60], axis=0)))
print("min std: ", np.min(np.std(X[:, 0:60], axis=0)))
print("max std: ", np.max(np.std(X[:, 0:60], axis=0)))
print("max: ", np.max(X[:, 0:60]))
print("min: ", np.min(X[:, 0:60]))

# X_test statistics
print("Testing feature statistics (numerical features only):")
print("min mean: ", np.min(np.mean(X_test[:, 0:60], axis=0)))
print("min std: ", np.min(np.std(X_test[:, 0:60], axis=0)))
print("max std: ", np.max(np.std(X_test[:, 0:60], axis=0)))
print("max:", np.max(X_test[:, 0:60]))
print("min:", np.min(X_test[:, 0:60]))

# y statistics
print("Label statistics")
print("min mean: ", np.min(np.mean(y, axis=0)))
print("min std: ", np.min(np.std(y, axis=0)))
print("max std: ", np.max(np.std(y, axis=0)))
print("max: ", np.max(y))
print("min: ", np.min(y))

X = build_model_data(X)
X_test = build_model_data(X_test)
N, D = X.shape

### # RANDOM FOREST: comment when training random forest
print(
    "Training regularised logistic regression with class weighting and threshold tuning..."
)

# Convert labels from {-1, 1} to {0, 1}
y_01 = (y > 0).astype(float)

# Compute class weights (currently testing square root of relative frequency)
relative_frequency = np.sum(y_01 == 0) / np.sum(y_01 == 1)
class_weights = {0: 1.0, 1: relative_frequency}

print(f"Using weights: (Class 0: 1.0, Class 1: {class_weights[1]:.2f})")

original_max_epochs = 200
splits = 5
train_split_size = (splits - 1) / splits
batch_size = 4096
iterations_per_epoch = (len(y_01) * train_split_size + batch_size - 1) // batch_size
initial_gamma = 0.001

best_lambda, best_epoch, best_threshold = cross_validate(
    X=X,
    y=y_01,
    initial_w=np.zeros(D),
    lambda_grid=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    max_epochs=original_max_epochs,
    gamma=initial_gamma,
    class_weights=class_weights,
    patience=50,
    batch_size=batch_size,
    decay_epochs_ratio=1.0,
    n_splits=splits,
    threshold_strategy="quantile",
    verbose_lambda=1,
    verbose_fold=1,
)

optimal_iterations = best_epoch * iterations_per_epoch
new_iterations_per_epoch = (len(y_01) + batch_size - 1) // batch_size
final_epochs = int(
    (optimal_iterations + new_iterations_per_epoch - 1) // new_iterations_per_epoch
)
new_decay_ratio = original_max_epochs / final_epochs

best_w, _, _ = reg_weighted_logistic_regression(
    X_train=X,
    y_train=y_01,
    initial_w=np.zeros(D),
    max_epochs=final_epochs,
    gamma=initial_gamma,
    decay_epochs_ratio=new_decay_ratio,
    lambda_=best_lambda,
    class_weights=class_weights,
    patience=final_epochs + 1,
    batch_size=batch_size,
    verbose=1,
)
###

### # RANDOM FOREST: uncomment when training random forest
# MODEL_PATH = "forest_d25_t200.pkl"
# THRESH_PATH = "forest_d25_t200_thresh.pkl"
# sys.setrecursionlimit(5000)

# if os.path.exists(MODEL_PATH) and os.path.exists(THRESH_PATH):
#     # Check if the model is already trained and saved, if yes load it
#     print(f"Loading model from {MODEL_PATH}...")
#     with open(MODEL_PATH, "rb") as f:
#         rf = pickle.load(f)

#     # Check if the threshold is already trained and saved, if yes load it
#     print(f"Loading threshold from {THRESH_PATH}...")
#     with open(THRESH_PATH, "rb") as f:
#         best_threshold = pickle.load(f)

# elif os.path.exists(MODEL_PATH) and not os.path.exists(THRESH_PATH):
#     # Check if the model is already trained and saved, if yes load it
#     print(f"Loading model from {MODEL_PATH}...")
#     with open(MODEL_PATH, "rb") as f:
#         rf = pickle.load(f)

#     # Training and validation splitting
#     train_idx, val_idx = split_train_val(y, val_size=0.2, random_seed=40)
#     X_train = X[train_idx]
#     y_train = y[train_idx]
#     X_val = X[val_idx]
#     y_val = y[val_idx]

#     best_threshold, _ = tune_threshold(X_val, y_val, rf)
#     print("Threshold tuned!")

#     print(f"Saving best threshold ({best_threshold}) to {THRESH_PATH}...")
#     with open(THRESH_PATH, "wb") as f:
#         pickle.dump(best_threshold, f)

# else:
#     # There isn't a pre-trained model, train a new one
#     print(f"Model file {MODEL_PATH} not found. Train a new one ...")

#     # The n_feature above is splitted into 10 numerical and 8 categorical in order to stratify the split, sqrt(X.shape[1])
#     n_features_tuple = (7, 13)

#     # Create the model
#     rf = RandomForest(
#         n_trees=200, max_depth=15, min_samples_split=20, n_features=n_features_tuple
#     )
#     # Set the indices of the numerical and the categorical variables in the dataset
#     rf.num_indices = np.arange(0, 60)
#     rf.cat_indices = np.arange(60, X.shape[1])

#     # Training and validation splitting
#     train_idx, val_idx = split_train_val(y, val_size=0.2, random_seed=40)
#     X_train = X[train_idx]
#     y_train = y[train_idx]
#     X_val = X[val_idx]
#     y_val = y[val_idx]

#     print("Starting the training...")
#     rf.fit(X_train, y_train)
#     print("Training done!")

#     print("Saving the model...")
#     print(f"Saving model to {MODEL_PATH}...")
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(rf, f)

#     best_threshold, _ = tune_threshold(X_val, y_val)
#     print("Threshold tuned!")

#     print(f"Saving best threshold ({best_threshold}) to {THRESH_PATH}...")
#     with open(THRESH_PATH, "wb") as f:
#         pickle.dump(best_threshold, f)
###

# --------------------------------------------------------------
# Generate predictions on the test set
# --------------------------------------------------------------

# Predict probabilities
### # RANDOM FOREST: comment when training random forest
y_pred = stable_sigmoid(X_test @ best_w)
###

### # RANDOM FOREST: uncomment when training random forest
# y_pred = rf.predict(X_test, threshold=best_threshold)
###

# Convert probabilities to classes in {-1, 1}
y_pred[y_pred >= best_threshold] = 1
y_pred[y_pred < best_threshold] = -1

print("Number of predicted positives: ", np.sum(y_pred == 1))
print("Number of predicted negatives: ", np.sum(y_pred == -1))
print("F1 score: ", f1_score(y_pred, y))

create_csv_submission(test_ids, y_pred, "submission.csv")
