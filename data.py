"""
Module for loading, cleaning, and saving dataset features and labels.

Preprocssing pipeline:
1. Load data from CSV files.
2. Manual preprocessing to fix known data issues.
3. Split features into numerical and categorical based on predefined indices.
4. Remove features with >threshold% NaN values.
5. Remove highly correlated features, keeping the one with fewest NaNs.
6. Impute and fill missing values.
7. Standardize numerical features to have zero mean and unit variance.
8. Selects categorical features based on association, predictiveness, and NaN count
9. One-hot encode categorical features.
10. Save processed features and labels to CSV files.
"""

import csv
import warnings

import numpy as np

from numpy.typing import NDArray


def load_data(
    features_file: str = "dataset/features_train.csv",
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[int, str]]:
    """Load features, row IDs, and feature names from a CSV file.

    Args:
        features_file: path to the features CSV file.

    Returns:
        features (without row IDs), row IDs, mapping of feature IDs to original names.

    Note:
        1. The first row of returned features is a unique identifier
           for identifying each feature later in the processing pipeline.
    """
    features = np.genfromtxt(features_file, delimiter=",").astype(np.float32)

    # create feature IDs such that we can identify them later
    # (when we remove some features for example)
    num_features = len(features[0, :])
    feature_ids = np.arange(1_000_000, 1_000_000 + num_features, step=1)
    features[0, :] = feature_ids

    # open with the csv module to read feature/column names
    with open(features_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in reader:
            feature_names = row[0].split(",")
            break

    # map the feature names to unique ID declared above
    feature_names_dict = {1_000_000 + i: feature_names[i] for i in range(num_features)}

    return features[:, 1:], features[:, 0].reshape(-1, 1), feature_names_dict


def manual_preprocess(features: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Manually preprocess the features based on known data issues.

    Args:
        features: features as a numpy array (from `load_data`).

    Returns:
        Preprocessed features as a numpy array.
    """
    features[features[:, 6] == 1100, 6] = 1
    features[features[:, 6] == 1200, 6] = 0

    features[features[:, 11] == 1, 10] = 2
    features[features[:, 21] == 1, 10] = 1
    features[features[:, 22] == 1, 10] = 2

    features[features[:, 15] > 6, 15] = 6
    features[features[:, 16] > 6, 16] = 6
    features[features[:, 17] > 6, 17] = 6

    features[features[:, 25] == 77, 25] = np.nan
    features[features[:, 25] == 99, 25] = np.nan

    features[features[:, 27] == 88, 27] = 0
    features[features[:, 27] == 77, 27] = np.nan
    features[features[:, 27] == 99, 27] = np.nan

    features[features[:, 28] == 88, 28] = 0
    features[features[:, 28] == 77, 28] = np.nan
    features[features[:, 28] == 99, 28] = np.nan

    features[features[:, 29] == 88, 29] = 0
    features[features[:, 29] == 77, 29] = np.nan
    features[features[:, 29] == 99, 29] = np.nan

    features[features[:, 49] == 98, 49] = np.nan
    features[features[:, 49] == 99, 49] = np.nan

    features[features[:, 59] == 88, 59] = 0
    features[features[:, 59] == 99, 59] = np.nan

    features[features[:, 62] == 7777, 62] = np.nan
    features[features[:, 62] == 9999, 62] = np.nan
    special_rows = (features[:, 62] >= 9000) & (features[:, 62] <= 9998)
    features[special_rows, 62] = (features[special_rows, 62] - 9000) * 2.2

    features[features[:, 63] == 7777, 63] = np.nan
    features[features[:, 63] == 9999, 63] = np.nan
    special_rows = (features[:, 63] >= 9000) & (features[:, 63] <= 9998)
    features[special_rows, 63] = features[special_rows, 63] - 9000
    special_rows = (features[:, 63] >= 200) & (features[:, 63] <= 711)
    feet = features[special_rows, 63] // 100
    inches = features[special_rows, 63] % 100
    meters = (feet * 0.3048) + (inches * 0.0254)
    features[special_rows, 63] = meters

    features[features[:, 77] == 777, 77] = np.nan
    features[features[:, 77] == 999, 77] = np.nan
    features[features[:, 77] == 888, 77] = 0
    special_rows = (features[:, 77] >= 101) & (features[:, 77] <= 199)
    features[special_rows, 77] = features[special_rows, 77] - 100
    special_rows = (features[:, 77] >= 201) & (features[:, 77] <= 299)
    features[special_rows, 77] = (features[special_rows, 77] - 200) / 7.0

    features[features[:, 78] == 77, 78] = np.nan
    features[features[:, 78] == 99, 78] = np.nan

    features[features[:, 79] == 77, 79] = np.nan
    features[features[:, 79] == 88, 79] = 0
    features[features[:, 79] == 99, 79] = np.nan

    features[features[:, 80] == 77, 80] = np.nan
    features[features[:, 80] == 99, 80] = np.nan

    special_rows = (features[:, 81] >= 101) & (features[:, 81] <= 199)
    features[special_rows, 81] = features[special_rows, 81] - 100
    special_rows = (features[:, 81] >= 201) & (features[:, 81] <= 299)
    features[special_rows, 81] = (features[special_rows, 81] - 200) / 7.0
    features[features[:, 81] == 300, 81] = 0
    special_rows = (features[:, 81] >= 301) & (features[:, 81] <= 399)
    features[special_rows, 81] = (features[special_rows, 81] - 300) / 30.0
    features[features[:, 81] == 555, 81] = 0
    features[features[:, 81] == 777, 81] = np.nan
    features[features[:, 81] == 999, 81] = np.nan

    special_rows = (features[:, 82] >= 101) & (features[:, 82] <= 199)
    features[special_rows, 82] = features[special_rows, 82] - 100
    special_rows = (features[:, 82] >= 201) & (features[:, 82] <= 299)
    features[special_rows, 82] = (features[special_rows, 82] - 200) / 7.0
    features[features[:, 82] == 300, 82] = 0
    special_rows = (features[:, 82] >= 301) & (features[:, 82] <= 399)
    features[special_rows, 82] = (features[special_rows, 82] - 300) / 30.0
    features[features[:, 82] == 555, 82] = 0
    features[features[:, 82] == 777, 82] = np.nan
    features[features[:, 82] == 999, 82] = np.nan

    special_rows = (features[:, 83] >= 101) & (features[:, 83] <= 199)
    features[special_rows, 83] = features[special_rows, 83] - 100
    special_rows = (features[:, 83] >= 201) & (features[:, 83] <= 299)
    features[special_rows, 83] = (features[special_rows, 83] - 200) / 7.0
    features[features[:, 83] == 300, 83] = 0
    special_rows = (features[:, 83] >= 301) & (features[:, 83] <= 399)
    features[special_rows, 83] = (features[special_rows, 83] - 300) / 30.0
    features[features[:, 83] == 555, 83] = 0
    features[features[:, 83] == 777, 83] = np.nan
    features[features[:, 83] == 999, 83] = np.nan

    special_rows = (features[:, 84] >= 101) & (features[:, 84] <= 199)
    features[special_rows, 84] = features[special_rows, 84] - 100
    special_rows = (features[:, 84] >= 201) & (features[:, 84] <= 299)
    features[special_rows, 84] = (features[special_rows, 84] - 200) / 7.0
    features[features[:, 84] == 300, 84] = 0
    special_rows = (features[:, 84] >= 301) & (features[:, 84] <= 399)
    features[special_rows, 84] = (features[special_rows, 84] - 300) / 30.0
    features[features[:, 84] == 555, 84] = 0
    features[features[:, 84] == 777, 84] = np.nan
    features[features[:, 84] == 999, 84] = np.nan

    special_rows = (features[:, 85] >= 101) & (features[:, 85] <= 199)
    features[special_rows, 85] = features[special_rows, 85] - 100
    special_rows = (features[:, 85] >= 201) & (features[:, 85] <= 299)
    features[special_rows, 85] = (features[special_rows, 85] - 200) / 7.0
    features[features[:, 85] == 300, 85] = 0
    special_rows = (features[:, 85] >= 301) & (features[:, 85] <= 399)
    features[special_rows, 85] = (features[special_rows, 85] - 300) / 30.0
    features[features[:, 85] == 555, 85] = 0
    features[features[:, 85] == 777, 85] = np.nan
    features[features[:, 85] == 999, 85] = np.nan

    special_rows = (features[:, 86] >= 101) & (features[:, 86] <= 199)
    features[special_rows, 86] = features[special_rows, 86] - 100
    special_rows = (features[:, 86] >= 201) & (features[:, 86] <= 299)
    features[special_rows, 86] = (features[special_rows, 86] - 200) / 7.0
    features[features[:, 86] == 300, 86] = 0
    special_rows = (features[:, 86] >= 301) & (features[:, 86] <= 399)
    features[special_rows, 86] = (features[special_rows, 86] - 300) / 30.0
    features[features[:, 86] == 555, 86] = 0
    features[features[:, 86] == 777, 86] = np.nan
    features[features[:, 86] == 999, 86] = np.nan

    special_rows = (features[:, 89] >= 101) & (features[:, 89] <= 199)
    features[special_rows, 89] = features[special_rows, 89] - 100
    special_rows = (features[:, 89] >= 201) & (features[:, 89] <= 299)
    features[special_rows, 89] = (features[special_rows, 89] - 200) / 4.285
    features[features[:, 89] == 777, 89] = np.nan
    features[features[:, 89] == 999, 89] = np.nan

    features[features[:, 90] == 777, 90] = np.nan
    features[features[:, 90] == 999, 90] = np.nan
    special_rows = ((features[:, 90] >= 1) & (features[:, 90] <= 759)) | (
        (features[:, 90] >= 800) & (features[:, 90] <= 959)
    )
    features[special_rows, 90] = (features[special_rows, 90] // 100) + (
        features[special_rows, 90] % 100
    ) / 60.0

    special_rows = (features[:, 92] >= 101) & (features[:, 92] <= 199)
    features[special_rows, 92] = features[special_rows, 92] - 100
    special_rows = (features[:, 92] >= 201) & (features[:, 92] <= 299)
    features[special_rows, 92] = (features[special_rows, 92] - 200) / 4.285
    features[features[:, 92] == 777, 92] = np.nan
    features[features[:, 92] == 999, 92] = np.nan

    features[features[:, 93] == 777, 93] = np.nan
    features[features[:, 93] == 999, 93] = np.nan
    special_rows = ((features[:, 93] >= 1) & (features[:, 93] <= 759)) | (
        (features[:, 93] >= 800) & (features[:, 93] <= 959)
    )
    features[special_rows, 93] = (features[special_rows, 93] // 100) + (
        features[special_rows, 93] % 100
    ) / 60.0

    special_rows = (features[:, 94] >= 101) & (features[:, 94] <= 199)
    features[special_rows, 94] = features[special_rows, 94] - 100
    special_rows = (features[:, 94] >= 201) & (features[:, 94] <= 299)
    features[special_rows, 94] = (features[special_rows, 94] - 200) / 4.285
    features[features[:, 94] == 777, 94] = np.nan
    features[features[:, 94] == 999, 94] = np.nan
    features[features[:, 94] == 888, 94] = 0

    features[features[:, 98] == 77, 98] = np.nan
    features[features[:, 98] == 99, 98] = np.nan

    special_rows = (features[:, 110] >= 101) & (features[:, 110] <= 199)
    features[special_rows, 110] = features[special_rows, 110] - 100
    special_rows = (features[:, 110] >= 201) & (features[:, 110] <= 299)
    features[special_rows, 110] = (features[special_rows, 110] - 200) / 7.0
    special_rows = (features[:, 110] >= 301) & (features[:, 110] <= 399)
    features[special_rows, 110] = (features[special_rows, 110] - 300) / 30.0
    special_rows = (features[:, 110] >= 401) & (features[:, 110] <= 499)
    features[special_rows, 110] = (features[special_rows, 110] - 400) / 365.0
    features[features[:, 94] == 777, 94] = np.nan
    features[features[:, 94] == 999, 94] = np.nan
    features[features[:, 94] == 888, 94] = 0

    special_rows = (features[:, 111] >= 101) & (features[:, 111] <= 199)
    features[special_rows, 111] = features[special_rows, 111] - 100
    special_rows = (features[:, 111] >= 201) & (features[:, 111] <= 299)
    features[special_rows, 111] = (features[special_rows, 111] - 200) / 7.0
    special_rows = (features[:, 111] >= 301) & (features[:, 111] <= 399)
    features[special_rows, 111] = (features[special_rows, 111] - 300) / 30.0
    special_rows = (features[:, 111] >= 401) & (features[:, 111] <= 499)
    features[special_rows, 111] = (features[special_rows, 111] - 400) / 365.0
    features[features[:, 111] == 555, 111] = np.nan
    features[features[:, 111] == 777, 111] = np.nan
    features[features[:, 111] == 999, 111] = np.nan
    features[features[:, 111] == 888, 111] = 0

    features[features[:, 112] == 99, 112] = np.nan
    features[features[:, 112] == 77, 112] = np.nan
    features[features[:, 112] == 88, 112] = 0

    features[features[:, 113] == 99, 113] = np.nan
    features[features[:, 113] == 77, 113] = np.nan
    features[features[:, 113] == 98, 113] = 0
    features[features[:, 113] == 88, 113] = 0

    features[features[:, 114] == 99, 114] = np.nan
    features[features[:, 114] == 77, 114] = np.nan
    features[features[:, 114] == 88, 114] = 0

    special_rows = (features[:, 143] >= 101) & (features[:, 143] <= 199)
    features[special_rows, 143] = features[special_rows, 143] - 100
    special_rows = (features[:, 143] >= 201) & (features[:, 143] <= 299)
    features[special_rows, 143] = (features[special_rows, 143] - 200) * 7.0
    special_rows = (features[:, 143] >= 301) & (features[:, 143] <= 399)
    features[special_rows, 143] = (features[special_rows, 143] - 300) * 30.0
    special_rows = (features[:, 143] >= 401) & (features[:, 143] <= 499)
    features[special_rows, 143] = (features[special_rows, 143] - 400) * 365.0
    features[features[:, 143] == 555, 143] = np.nanmax(features[:, 143])
    features[features[:, 143] == 777, 143] = np.nan
    features[features[:, 143] == 999, 143] = np.nan

    features[features[:, 145] == 97, 145] = 10
    features[features[:, 145] == 777, 145] = np.nan
    features[features[:, 145] == 999, 145] = np.nan

    features[features[:, 147] == 88, 147] = 0
    features[features[:, 147] == 98, 147] = np.nan

    features[features[:, 148] == 88, 148] = 0
    features[features[:, 148] == 98, 148] = np.nan

    features[features[:, 149] == 88, 149] = 0
    features[features[:, 149] == 98, 149] = np.nan
    features[features[:, 149] == 99, 149] = np.nan

    features[features[:, 150] == 888, 150] = 0
    features[features[:, 150] == 777, 150] = np.nan
    features[features[:, 150] == 999, 150] = np.nan

    features[features[:, 168] == 77, 168] = np.nan
    features[features[:, 168] == 99, 168] = np.nan

    features[features[:, 195] == 97, 195] = np.nan
    features[features[:, 195] == 98, 195] = 0
    features[features[:, 195] == 99, 195] = np.nan

    features[features[:, 197] == 97, 197] = np.nan
    features[features[:, 197] == 98, 197] = 0
    features[features[:, 197] == 99, 197] = np.nan

    features[features[:, 206] == 77, 206] = np.nan
    features[features[:, 206] == 88, 206] = 0
    features[features[:, 206] == 99, 206] = np.nan

    features[features[:, 207] == 77, 207] = np.nan
    features[features[:, 207] == 88, 207] = 0
    features[features[:, 207] == 99, 207] = np.nan

    features[features[:, 208] == 77, 208] = np.nan
    features[features[:, 208] == 88, 208] = 0
    features[features[:, 208] == 99, 208] = np.nan

    features[features[:, 209] == 77, 209] = np.nan
    features[features[:, 209] == 88, 209] = 0
    features[features[:, 209] == 99, 209] = np.nan

    features[features[:, 210] == 77, 210] = np.nan
    features[features[:, 210] == 88, 210] = 0
    features[features[:, 210] == 99, 210] = np.nan

    features[features[:, 211] == 77, 211] = np.nan
    features[features[:, 211] == 88, 211] = 0
    features[features[:, 211] == 99, 211] = np.nan

    features[features[:, 212] == 77, 212] = np.nan
    features[features[:, 212] == 88, 212] = 0
    features[features[:, 212] == 99, 212] = np.nan

    features[features[:, 213] == 77, 213] = np.nan
    features[features[:, 213] == 88, 213] = 0
    features[features[:, 213] == 99, 213] = np.nan

    features[features[:, 256] == 9, 256] = np.nan

    features[features[:, 262] == 900, 262] = np.nan

    features[features[:, 264] == 99900, 264] = np.nan

    return features


def split_numerical_categorical(
    features: NDArray[np.float32],
    test_features: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Split features into numerical and categorical based on provided indices.

    Args:
        features: features as a numpy array (from `load_data`).
        numerical_indices: indices of numerical features.
        categorical_indices: indices of categorical features.
    Returns:
        tuple of numerical features and categorical features.
    """
    to_delete = np.array(
        [0, 1, 2, 4, 7, 8, 9, 11, 12, 13, 18, 19, 21, 22, 23, 24, 55, 101, 105, 219]
    )
    numerical = np.array(
        [
            3,
            15,
            16,
            17,
            25,
            27,
            28,
            29,
            49,
            59,
            62,
            63,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            89,
            90,
            92,
            93,
            94,
            98,
            110,
            111,
            112,
            113,
            114,
            143,
            145,
            147,
            148,
            149,
            150,
            168,
            195,
            197,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            220,
            221,
            222,
            226,
            228,
            229,
            248,
            250,
            251,
            252,
            253,
            256,
            262,
            264,
            266,
            267,
            268,
            269,
            270,
            271,
            276,
            277,
            287,
            288,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            299,
            300,
            301,
            302,
            303,
            304,
        ]
    )
    categorical = np.arange(len(features[0, :]))
    not_categorical = np.concatenate((numerical, to_delete), axis=0)
    categorical = np.setdiff1d(categorical, not_categorical)

    numerical += 1000001
    categorical += 1000001

    mask_to_numerical = np.isin(features[0], numerical)
    numerical_features = features[:, mask_to_numerical]
    test_numerical_features = test_features[:, mask_to_numerical]

    mask_to_categorical = np.isin(features[0], categorical)
    categorical_features = features[:, mask_to_categorical]
    test_categorical_features = test_features[:, mask_to_categorical]

    cols_to_indicate = np.arange(len(numerical_features[0, :]))

    indicator_features = np.isnan(numerical_features[:, cols_to_indicate]).astype(int)
    indicator_features[0, :] = numerical_features[0, cols_to_indicate]
    categorical_features = np.hstack((categorical_features, indicator_features))

    test_indicator_features = np.isnan(
        test_numerical_features[:, cols_to_indicate]
    ).astype(int)
    test_indicator_features[0, :] = test_numerical_features[0, cols_to_indicate]
    test_categorical_features = np.hstack(
        (test_categorical_features, test_indicator_features)
    )

    return (
        numerical_features,
        categorical_features,
        test_numerical_features,
        test_categorical_features,
    )


def remove_nans(
    features: NDArray[np.float32],
    test_features: NDArray[np.float32],
    threshold: int | float = 50.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Remove features with >threshold% NaN values and change all other NaNs to `nan`.

    Args:
        features: features as a numpy array (from `load_data`).
        test_features: test features as a numpy array.
        threshold: percentage threshold for removing features.

    Returns:
        features and test_features with columns with >threshold% NaNs removed.
    """
    # create a mask to remove columns with >threshold% NaNs
    nan_mask = (
        (np.sum(np.isnan(features), axis=0) + np.sum(np.isnan(test_features), axis=0))
        / (2 * features.shape[0])
    ) * 100
    new_features = features[:, nan_mask < threshold]
    new_test_features = test_features[:, nan_mask < threshold]
    return new_features, new_test_features


def remove_correlated_features_by_nan(
    features: NDArray[np.float32],
    threshold: float = 0.9,
) -> tuple[NDArray[np.float32], list[int]]:
    """Removes highly correlated features, keeping the one with fewest NaNs.

    Calculates Pearson correlation coefficients. If the absolute correlation
    between two features exceeds the threshold, they are considered part of
    a correlated group. From each group, only the feature with the minimum
    number of NaN values is retained.

    Args:
        features: The feature matrix (samples x features) as a numpy array.
                  Should contain NaNs for the selection logic to work.
        threshold: Absolute correlation coefficient threshold (e.g., 0.9).

    Returns:
        tuple containing:
            - features_reduced: Numpy array with correlated features removed.
            - kept_indices: List of the indices of the columns that were kept.
    """
    _, num_features = features.shape

    # 1. Calculate NaN counts for each feature
    nan_counts = np.isnan(features).sum(axis=0)

    # 2. Compute the correlation matrix
    try:
        # Suppress RuntimeWarnings about invalid values if NaNs exist
        with np.errstate(invalid="ignore"):
            corref = np.corrcoef(features, rowvar=False)
        # Replace NaNs in the correlation matrix with 0
        corref = np.nan_to_num(corref)
    except Exception as e:
        print(f"Error computing correlation matrix: {e}")
        return features, list(range(num_features))

    # 3. Identify groups of correlated features (using connected components)
    processed = [False] * num_features
    groups = []
    for i in range(num_features):
        if not processed[i]:
            current_group = set()
            queue = [i]
            processed[i] = True

            while queue:
                u = queue.pop(0)
                current_group.add(u)
                # Find neighbors (correlated features)
                # Check correlations with all other features (v != u)
                for v in range(num_features):
                    if u == v:  # Skip self-correlation
                        continue

                    # Use abs() for threshold check
                    if not processed[v] and abs(corref[u, v]) > threshold:
                        processed[v] = True
                        queue.append(v)
            groups.append(list(current_group))

    # 4. Select the feature with the fewest NaNs from each group
    indices_to_keep = []
    for group in groups:
        if len(group) == 1:
            # If a feature isn't correlated with any others, keep it
            indices_to_keep.append(group[0])
        else:
            # For correlated groups, find the one with the minimum NaNs
            group_nan_counts = nan_counts[group]
            min_nan_local_idx = np.argmin(
                group_nan_counts
            )  # Index within the group list
            feature_to_keep = group[min_nan_local_idx]  # Original index
            indices_to_keep.append(feature_to_keep)

    # Ensure indices are sorted for consistent column order
    indices_to_keep.sort()

    # 5. Create the reduced feature set
    features_reduced = features[:, indices_to_keep]

    return features_reduced, indices_to_keep


def impute(
    features: NDArray[np.float32], test_features: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Impute NaN values with the median of each feature calculated from both datasets.

    Args:
        features: numpy array of training features.
        test_features: numpy array of test features.

    Returns:
        Tuple of numpy arrays with imputed training and test features.
    """
    num_features = len(features[0, :])
    X = np.concatenate((features, test_features), axis=0)

    # Calculate medians, ignoring NaNs
    # Suppress RuntimeWarnings for columns that might be all NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(X, axis=0)

    # Find columns that were all NaN (median will be NaN) and replace their median with 0
    all_nan_cols = np.isnan(medians)
    medians[all_nan_cols] = 0

    # Impute using calculated medians
    for i in range(num_features):
        nan_mask = np.isnan(features[:, i])
        features[nan_mask, i] = medians[i]
        nan_mask = np.isnan(test_features[:, i])
        test_features[nan_mask, i] = medians[i]

    return features, test_features


def standardize(
    features: NDArray[np.float32], test_features: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Standardize features to have zero mean and unit variance.

    Args:
        features: numpy array of training features.
        test_features: numpy array of test features.

    Returns:
        Tuple of numpy arrays with standardized training and test features.
    """
    concat = np.concatenate((features, test_features), axis=0)
    mean = np.mean(concat, axis=0)
    std = np.std(concat, axis=0)
    std[std == 0] = 1e-8

    features_std = (features - mean) / std
    test_features_std = (test_features - mean) / std
    return features_std, test_features_std


def calculate_cramers_v_numpy(
    col1: NDArray[np.float32], col2: NDArray[np.float32]
) -> float:
    """
    Calculates Cramer's V using only NumPy.

    Args:
        col1: 1D numpy array representing the first categorical variable.
        col2: 1D numpy array representing the second categorical variable.

    Returns:
        Cramer's V statistic as a float between 0 and 1.
    """
    # Combine columns and drop rows where EITHER column has NaN
    valid_mask = ~np.isnan(col1) & ~np.isnan(col2)
    n_valid = np.sum(valid_mask)

    if n_valid < 2:  # Need at least 2 valid pairs
        return 0.0

    col1_valid = col1[valid_mask]
    col2_valid = col2[valid_mask]

    # Create contingency table (observed frequencies O_ij)
    unique1, counts1 = np.unique(col1_valid, return_inverse=True)
    unique2, counts2 = np.unique(col2_valid, return_inverse=True)

    # Handle cases where a column has only one unique value after NaN removal
    if len(unique1) < 2 or len(unique2) < 2:
        return 0.0

    contingency_table = np.zeros((len(unique1), len(unique2)), dtype=int)
    np.add.at(contingency_table, (counts1, counts2), 1)

    # Calculate Chi-squared statistic
    row_totals = contingency_table.sum(axis=1)
    col_totals = contingency_table.sum(axis=0)
    n = contingency_table.sum()

    if n == 0:
        return 0.0

    # Calculate expected frequencies E_ij
    expected = np.outer(row_totals, col_totals) / n

    # Chi-squared formula: sum(((O - E)^2) / E)
    # Add a small epsilon to avoid division by zero if expected is 0
    epsilon = 1e-10
    chi2 = np.sum((contingency_table - expected) ** 2 / (expected + epsilon))

    # Calculate Cramer's V
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0:  # Avoid division by zero if table has only 1 row or 1 col
        return 0.0

    v = np.sqrt(chi2 / (n * min_dim))

    # Clamp value to be between 0 and 1 (due to potential float inaccuracies)
    v = max(0.0, min(v, 1.0))

    return v


def select_cat_features_predictive(
    features: NDArray[np.float32],
    y_target: NDArray[np.int16],
    predictiveness_threshold: float = 0.1,
    association_threshold: float = 0.8,
    nan_placeholder: float | int | None = None,
) -> tuple[NDArray[np.float32], list[int]]:
    """
    Selects categorical features based on association, predictiveness, and NaN count.

    1. Calculates predictiveness (Cramer's V) of each feature vs. y_target.
    2. Identifies groups of highly associated features (Cramer's V > association_threshold).
    3. Keeps features if they are predictive enough (Cramer's V vs. y > predictiveness_threshold).
    4. From associated groups, keeps the *most predictive* feature (if multiple are predictive,
       breaks ties using fewest NaNs). If no feature in a group is predictive, discards all.

    Args:
        features: Numpy array (samples x features) containing categorical data.
                  Assumes features are numerically encoded (int/float).
        y_target: 1D Numpy array (samples,) containing the binary target variable (e.g., 0s and 1s).
        predictiveness_threshold: Cramer's V threshold for association with the target (e.g., 0.1).
                                  Features below this are considered not predictive enough.
        association_threshold: Cramer's V threshold for association between features (e.g., 0.8).
        nan_placeholder: If NaNs are represented by a specific value, provide it here.

    Returns:
        tuple containing:
            - features_reduced: Numpy array with less predictive and associated features removed.
                                Contains original data (including NaNs) for kept columns.
            - kept_indices: List of the original indices of the columns kept.
    """
    if features.ndim != 2:
        raise ValueError("Input 'features' must be a 2D NumPy array.")
    if y_target.ndim != 1 or len(y_target) != features.shape[0]:
        raise ValueError(
            "y_target must be a 1D array with the same number of samples as features."
        )

    num_samples, num_features = features.shape

    # --- Make a working copy and standardize NaNs ---
    features_working = features.astype(float).copy()
    y_target_working = y_target.astype(
        float
    ).copy()  # Ensure target is float for NaN handling
    if nan_placeholder is not None:
        features_working[features_working == nan_placeholder] = np.nan
        # Apply placeholder logic to target if needed, though usually not applicable for binary target
        if np.any(y_target_working == nan_placeholder):
            y_target_working[y_target_working == nan_placeholder] = np.nan

    # 1. Calculate original NaN counts
    nan_counts = np.isnan(features_working).sum(axis=0)

    # 2. Calculate Predictiveness Scores (Cramer's V vs. y_target)
    predictiveness_scores = np.zeros(num_features)
    for i in range(num_features):
        predictiveness_scores[i] = calculate_cramers_v_numpy(
            features_working[:, i], y_target_working
        )
        # Debug print (optional)
        # print(f"Feature {i} predictiveness: {predictiveness_scores[i]:.4f}")

    # 3. Compute the feature association matrix (Cramer's V) if not provided
    if num_features > 1:
        assoc_matrix = np.identity(num_features)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                v = calculate_cramers_v_numpy(
                    features_working[:, i], features_working[:, j]
                )
                assoc_matrix[i, j] = assoc_matrix[j, i] = v
    elif num_features <= 1:
        assoc_matrix = np.identity(num_features)  # Handle single feature case

    # 4. Identify groups of associated features
    processed = [False] * num_features
    groups = []
    is_grouped = [
        False
    ] * num_features  # Track if a feature belongs to a multi-member group
    for i in range(num_features):
        if not processed[i]:
            current_group_indices = set()
            queue = [i]
            processed[i] = True

            while queue:
                u = queue.pop(0)
                current_group_indices.add(u)
                for v in range(num_features):
                    if u == v:
                        continue
                    # Check association matrix
                    if not processed[v] and assoc_matrix[u, v] > association_threshold:
                        processed[v] = True
                        queue.append(v)

            group_list = list(current_group_indices)
            groups.append(group_list)
            if len(group_list) > 1:
                for idx in group_list:
                    is_grouped[idx] = True  # Mark features in multi-member groups

    # 5. Select features based on predictiveness and NaNs
    indices_to_keep_set = set()  # Use a set to avoid duplicates

    # Process multi-feature groups first
    for group in groups:
        if len(group) == 1:
            continue  # Handle single features later

        # Filter group members by predictiveness threshold
        predictive_members = [
            idx
            for idx in group
            if predictiveness_scores[idx] >= predictiveness_threshold
        ]

        if not predictive_members:
            # If NO feature in the group is predictive enough, discard all
            # print(f"Discarding group {group} - none are predictive enough.") # Optional debug
            continue
        elif len(predictive_members) == 1:
            # If only one is predictive, keep that one
            indices_to_keep_set.add(predictive_members[0])
            # print(f"Keeping {predictive_members[0]} from group {group} - only predictive one.") # Optional debug
        else:
            # If multiple are predictive, find the one with the highest predictiveness score.
            # Use fewest NaNs as a tie-breaker.
            predictive_scores_subset = predictiveness_scores[predictive_members]
            max_score = np.max(predictive_scores_subset)

            # Get all local indices that tie for the max score
            max_score_local_indices = np.where(predictive_scores_subset == max_score)[0]

            if len(max_score_local_indices) == 1:
                # No tie, just keep the most predictive one
                feature_to_keep = predictive_members[max_score_local_indices[0]]
            else:
                # Tie in predictiveness, use fewest NaNs as a tie-breaker

                # Get the *original* indices of the tied features
                tied_original_indices = [
                    predictive_members[i] for i in max_score_local_indices
                ]

                # Get the NaN counts for *only* the tied features
                nan_counts_of_tied = nan_counts[tied_original_indices]

                # Find the local index (within the *tied* group) of the one with min NaNs
                min_nan_tied_local_idx = np.argmin(nan_counts_of_tied)

                # Select the corresponding original index
                feature_to_keep = tied_original_indices[min_nan_tied_local_idx]

            indices_to_keep_set.add(feature_to_keep)
            # print(f"Keeping {feature_to_keep} from group {group} - most predictive (NaN tie-break).") # Optional debug

    # Process single features (those not in any multi-feature group)
    for i in range(num_features):
        if not is_grouped[i]:
            # If it's a single feature, keep it ONLY if it meets the predictiveness threshold
            if predictiveness_scores[i] >= predictiveness_threshold:
                indices_to_keep_set.add(i)
                # print(f"Keeping single feature {i} - predictive enough.") # Optional debug
            # else:
            # print(f"Discarding single feature {i} - not predictive enough.") # Optional debug

    # Convert set to sorted list
    indices_to_keep = sorted(list(indices_to_keep_set))

    # 6. Create the reduced feature set using original indices from original features array
    if not indices_to_keep:  # Handle case where no features are kept
        features_reduced = np.zeros((num_samples, 0))
    else:
        # Use original 'features' array slice, preserving original data types/NaNs
        features_reduced = features[:, indices_to_keep]  # type: ignore[assignment]

    return features_reduced, indices_to_keep  # type: ignore[return-value]


def transport_AGEG5YR(
    categorical_features_clean: NDArray[np.float32],
    test_categorical_features_clean: NDArray[np.float32],
    numerical_features_clean: NDArray[np.float32],
    test_numerical_features_clean: NDArray[np.float32],
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """
    Remove the columns _AGEGYR5 from the categorical features
    and add it to the numerical, in order to not one-hot-encode it
    and to not standardize it

    Args:
        categorical_features_clean: 2D NumPy array for categorical training data (samples x features).
        test_categorical_features_clean: 2D NumPy array for categorical test data (samples x features).
        numerical_features_clean: 2D NumPy array for numerical training data (samples x features).
        test_numerical_features_clean: 2D NumPy array for numerical test data (samples x features).

    Returns:
        tuple containing:
            categorical_features_clean: 2D NumPy array for categorical training data without _AGEGYR5 (samples x features).
            test_categorical_features_clean: 2D NumPy array for categorical test data without _AGEGYR5(samples x features).
            numerical_features_clean: 2D NumPy array for numerical training data with _AGEGYR5(samples x features).
            test_numerical_features_clean: 2D NumPy array for numerical test data with _AGEGYR5(samples x features).

    """
    beautiful_col = np.where(categorical_features_clean[0, :] == 1000247)
    beautiful_data = np.ravel(categorical_features_clean[:, beautiful_col]).reshape(
        -1, 1
    )
    categorical_features_clean = np.delete(
        categorical_features_clean, beautiful_col, axis=1
    )

    test_beautiful_col = np.where(test_categorical_features_clean[0, :] == 1000247)
    test_beautiful_data = np.ravel(
        test_categorical_features_clean[:, test_beautiful_col]
    ).reshape(-1, 1)
    test_categorical_features_clean = np.delete(
        test_categorical_features_clean, test_beautiful_col, axis=1
    )

    numerical_features_clean = np.concatenate(
        (numerical_features_clean, beautiful_data), axis=1
    )

    test_numerical_features_clean = np.concatenate(
        (test_numerical_features_clean, test_beautiful_data), axis=1
    )
    return (
        categorical_features_clean,
        test_categorical_features_clean,
        numerical_features_clean,
        test_numerical_features_clean,
    )


def one_hot_encode_train_test_with_ids(
    features: NDArray[np.float32],
    test_features: NDArray[np.float32],
    handle_nan: bool = True,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], list[list[str | int | float | None]]
]:
    """
    Performs consistent one-hot encoding on train/test matrices with ID rows.

    The first row of each matrix is treated as column IDs.
    Determines categories based on data rows (excluding IDs) from both sets.
    Expands columns and replicates the original ID across the new columns.

    Args:
        features: 2D NumPy array for training data (samples x features).
                      The first row must contain column IDs.
                      Assumes float type if NaNs exist in data rows.
        test_features: 2D NumPy array for test data (samples x features).
                     The first row must contain column IDs.
                     Must have the same number of columns as features.
                     Assumes float type if NaNs exist in data rows.
        handle_nan: If True, treats NaN (or None for object arrays) in data rows
                    as a separate category. If False, rows with NaN/None will
                    have all zeros for that feature's encoded columns.

    Returns:
        tuple containing:
            - encoded_train_with_ids: Encoded train matrix including the replicated ID row.
            - encoded_test_with_ids: Encoded test matrix including the replicated ID row.
            - all_feature_categories: A list of lists, where each inner list contains
                                      the unique categories found for the corresponding
                                      original feature column (in the order they appear
                                      in the encoded output).
    """
    if features.shape[1] != test_features.shape[1]:
        raise ValueError(
            "Train and test matrices must have the same number of columns."
        )
    if features.shape[0] < 2 or test_features.shape[0] < 2:
        raise ValueError("Matrices must have at least 2 rows (ID row + data row).")

    # --- Extract IDs and Data ---
    ids_train = features[0, :]
    ids_test = test_features[
        0, :
    ]  # Assuming test IDs might be different, but structure is same
    data_train = features[1:, :]
    data_test = test_features[1:, :]

    num_train_samples = data_train.shape[0]
    num_test_samples = data_test.shape[0]
    num_features = data_train.shape[1]

    encoded_train_cols = []
    encoded_test_cols = []
    new_ids_train_list = []
    new_ids_test_list = []
    all_feature_categories = []

    for col_idx in range(num_features):
        col_train = data_train[:, col_idx]
        col_test = data_test[:, col_idx]

        # Combine columns to find all unique categories across both data sets
        combined_col = np.concatenate((col_train, col_test))

        # Determine if NaNs exist and identify unique categories from combined data
        is_nan_mask_combined = (combined_col == np.nan) | (combined_col is None)
        if combined_col.dtype == object:
            is_nan_mask_combined = np.array(
                [x is np.nan or x is None for x in combined_col]
            )

        has_nan_combined = np.any(is_nan_mask_combined)

        # Get unique non-NaN/None values from the combined column
        unique_categories = np.unique(combined_col[~is_nan_mask_combined])

        # Store the categories for this feature
        current_feature_cats = list(unique_categories)

        # Determine number of new columns for this feature
        num_new_cols = len(unique_categories)
        if handle_nan and has_nan_combined:
            num_new_cols += 1
            current_feature_cats.append(np.nan)  # Add NaN representation

        all_feature_categories.append(current_feature_cats)

        # --- Replicate original IDs for the new columns ---
        original_id_train = ids_train[col_idx]
        original_id_test = ids_test[col_idx]
        new_ids_train_list.extend([original_id_train] * num_new_cols)
        new_ids_test_list.extend([original_id_test] * num_new_cols)

        # Initialize encoded columns for this feature for both train and test data rows
        col_encoded_train = np.zeros((num_train_samples, num_new_cols), dtype=int)
        col_encoded_test = np.zeros((num_test_samples, num_new_cols), dtype=int)

        # --- Fill based on categories ---
        current_new_col_idx = 0
        is_nan_mask_train = (col_train == np.nan) | (col_train is None)
        if col_train.dtype == object:
            is_nan_mask_train = np.array([x is np.nan or x is None for x in col_train])
        is_nan_mask_test = (col_test == np.nan) | (col_test is None)
        if col_test.dtype == object:
            is_nan_mask_test = np.array([x is np.nan or x is None for x in col_test])

        for category in unique_categories:
            # Handle train data
            match_mask_train = col_train == category
            if category is not np.nan and category is not None:
                match_mask_train &= ~is_nan_mask_train
            col_encoded_train[match_mask_train, current_new_col_idx] = 1

            # Handle test data
            match_mask_test = col_test == category
            if category is not np.nan and category is not None:
                match_mask_test &= ~is_nan_mask_test
            col_encoded_test[match_mask_test, current_new_col_idx] = 1

            current_new_col_idx += 1

        # --- Fill NaN column if needed ---
        if handle_nan and has_nan_combined:
            col_encoded_train[is_nan_mask_train, current_new_col_idx] = 1
            col_encoded_test[is_nan_mask_test, current_new_col_idx] = 1

        encoded_train_cols.append(col_encoded_train)
        encoded_test_cols.append(col_encoded_test)

    # --- Combine all encoded data columns horizontally ---
    if not encoded_train_cols:  # Handle empty input
        encoded_train_data = np.zeros((num_train_samples, 0), dtype=int)
        encoded_test_data = np.zeros((num_test_samples, 0), dtype=int)
        new_ids_train = np.array([])
        new_ids_test = np.array([])
    else:
        encoded_train_data = np.hstack(encoded_train_cols)
        encoded_test_data = np.hstack(encoded_test_cols)
        # Convert the replicated ID lists to numpy arrays
        new_ids_train = np.array(new_ids_train_list).reshape(1, -1)
        new_ids_test = np.array(new_ids_test_list).reshape(1, -1)

    # --- Add the new ID row back to the top ---
    encoded_train_with_ids = np.vstack((new_ids_train, encoded_train_data))
    encoded_test_with_ids = np.vstack((new_ids_test, encoded_test_data))

    return encoded_train_with_ids, encoded_test_with_ids, all_feature_categories


def save_features(
    features: NDArray[np.float32],
    feature_names_dict: dict[int, str],
    row_ids: NDArray[np.float32],
    file_path: str,
) -> None:
    """Save features to a CSV file.

    Join the row IDs back to the features and replace feature IDs
    with original names before saving.

    Args:
        features: features as a numpy array.
        feature_names_dict: mapping of feature IDs to original names (from `load_data`).
        row_ids: row IDs as a numpy array (from `load_data`).
        file_path: path to save the CSV file.
    """
    # replace feature ID with names and save the features
    final_features = np.append(row_ids, features, axis=1)
    np.savetxt(
        file_path,
        final_features[1:, :],
        delimiter=",",
        header=",".join([feature_names_dict[i] for i in final_features[0, :]]),
        comments="",
        fmt="%f",
    )


# helper functions provided in the assignment


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
