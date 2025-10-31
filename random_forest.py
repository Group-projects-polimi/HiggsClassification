from __future__ import annotations


import numpy as np

from numpy.typing import NDArray

from metrics import f1_score


class Node:
    """Represents a node in the decision tree.

    A node can either be an internal node with a split condition (feature and threshold)
    or a leaf node holding a prediction value.

    Attributes:
        feature: Index of the feature used for splitting at this node. None for leaf nodes.
        threshold: Threshold value for the split condition. None for leaf nodes.
        left: Left child node (samples where feature <= threshold). None for leaf nodes.
        right: Right child node (samples where feature > threshold). None for leaf nodes.
        value: Predicted class label (-1 or 1) for this leaf node. None for internal nodes.
    """

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: Node | None = None,
        right: Node | None = None,
        *,
        value: int | None = None,
    ) -> None:
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Leaf node's prediction value (if it's a leaf)

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree:
    """A single Decision Tree classifier.

    Builds a tree recursively by finding the best split at each node based on Gini impurity.
    Handles numerical features and uses stratified feature sampling if feature indices are provided.

    Attributes:
        min_samples_split: Minimum number of samples required to split a node.
        max_depth: Maximum depth the tree is allowed to grow.
        n_features: Tuple (n_num_features, n_cat_features) specifying how many
                    numerical and categorical features to consider at each split.
        num_indices (np.ndarray): Array of indices corresponding to numerical features.
        cat_indices (np.ndarray): Array of indices corresponding to categorical features.
        root (Node): The root node of the trained decision tree.
    """

    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_features: tuple[int, int] | None = None,
        num_indices: NDArray[np.int_] | None = None,
        cat_indices: NDArray[np.int_] | None = None,
    ) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root: Node | None = None
        self.num_indices = num_indices
        self.cat_indices = cat_indices

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        """Builds the decision tree from the training data (X, y).

        Args:
            X: Training input samples. Shape (n_samples, n_total_features).
            y: Target class labels {-1, 1}. Shape (n_samples,).
        """
        print(f" Starting DecisionTree.fit on {X.shape[0]} samples...")
        self.n_features = self.n_features if self.n_features else X.shape[1]
        self.root = self._build_tree(X, y)

    def _build_tree(
        self, X: NDArray[np.float32], y: NDArray[np.float32], depth: int = 0
    ) -> Node:
        """Recursively builds the decision tree.
        Finds the best split at the current node and creates child nodes,
        until a stopping criterion (max depth, min samples, pure node) is met.
        Args:
            X (np.ndarray): Data samples at the current node.
            y (np.ndarray): Labels for samples at the current node.
            depth (int): Current depth in the tree.
        Returns:
            Node: The root node of the subtree built from this point.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        n_num_to_sample, n_cat_to_sample = self.n_features
        # Sample from numerical features
        num_feat_idxs = np.random.choice(
            self.num_indices, n_num_to_sample, replace=False
        )
        # Sample from categorical features
        cat_feat_idxs = np.random.choice(
            self.cat_indices, n_cat_to_sample, replace=False
        )
        # Combine them into the final list
        feat_idxs = np.concatenate([num_feat_idxs, cat_feat_idxs])

        best_feat, best_thresh = self._find_best_split(
            X, y, feat_idxs, threshold_num=1000
        )

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create child nodes recursively
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def _find_best_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        feat_idxs: NDArray[np.int_],
        threshold_num: int = 1000,
    ) -> tuple[NDArray[np.int_], int] | tuple[None, None]:
        """Finds the best feature and threshold to split the data at the current node.

        Iterates through the randomly selected features (`feat_idxs`) and potential
        thresholds for each, calculating the weighted Gini impurity for each split.
        Selects the split that minimizes Gini impurity. Uses percentile-based
        threshold sampling for features with many unique values.

        Args:
            X: Data samples at the current node.
            y: Labels for samples at the current node.
            feat_idxs: Indices of the features to consider for splitting.
            threshold_num: Max number of thresholds to check per feature using percentiles.

        Returns:
            (best_feature_index, best_threshold_value) or (None, None) if no split improves Gini.
        """
        best_gini = 1.0
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            feature_data = X[:, feat_idx]
            # Get all unique values for this feature
            all_thresholds = np.unique(feature_data)

            if len(all_thresholds) > threshold_num:
                # If we have too many, pick threshold_num percentiles as thresholds
                percentile_points = np.linspace(0, 100, threshold_num + 1)
                # Get the values at those percentiles
                thresholds = np.percentile(feature_data, percentile_points[1:-1])
                thresholds = np.unique(thresholds)
            else:
                # If we have a reasonable number, just use them all
                thresholds = all_thresholds

            for thresh in thresholds:
                # Split data
                left_idxs = feature_data <= thresh
                y_left, y_right = y[left_idxs], y[~left_idxs]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # Skip splits that don't produce two children

                # Calculate weighted average Gini
                gini = self._weighted_gini(y, y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _gini(self, y: NDArray[np.int16]) -> float:
        """Calculates the Gini impurity for a set of labels.

        Gini = 1 - sum(proportion_of_class_k ** 2)
        Lower Gini means higher node purity.

        Args:
            y: Dataset labels. Shape (n_samples,).

        Returns:
            float: Gini impurity value between 0 and 1.
        """
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions**2)

    def _weighted_gini(
        self, y_parent: DecisionTree, y_left: DecisionTree, y_right: DecisionTree
    ) -> float:
        """Calculates the weighted average Gini impurity of a split.

        Args:
            y_parent: Parent DecisionTree.
            y_left: Left DecisionTree.
            y_right: Right DecisionTree.

        Returns:
            float: Weighted Gini impurity of the split.
        """
        n_parent = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)
        return (n_left / n_parent) * gini_left + (n_right / n_parent) * gini_right

    def _most_common_label(self, y: NDArray[np.float32]) -> int:
        """Finds the most frequent label in an array. Used for leaf node prediction.

        Args:
            y: Array of labels.

        Returns:
            int: The most frequent label. Returns -1 if input is empty.
        """
        if len(y) == 0:
            return -1
        classes, counts = np.unique(y, return_counts=True)
        # Find the index of the highest count
        max_count_index = np.argmax(counts)
        # Return the class label associated with that count
        return classes[max_count_index]

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predicts class labels for input samples by traversing the tree.

        Args:
            X: Input samples. Shape (n_samples, n_features).

        Returns:
            NDArray: Predicted class labels {-1, 1} for each sample. Shape (n_samples,).
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: NDArray[np.float32], node: Node) -> int:
        """Recursively traverses the tree to find the prediction for a single sample 'x'.

        Args:
            x: A single input sample. Shape (n_features,).
            node: The current Node being visited.

        Returns:
            int: The predicted class label {-1, 1} from the leaf node reached.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForest:
    """A Random Forest classifier implemented from scratch using NumPy.
    This ensemble method builds multiple decision trees on different subsets
    of the data and features, using bagging and feature randomness to
    reduce variance and improve generalization. Predictions are made
    by aggregating the votes from individual trees.

    Attributes:
        n_trees: The number of decision trees in the forest.
        max_depth: The maximum depth allowed for each decision tree.
        min_samples_split: The minimum number of samples required to split an internal node.
        n_features: (n_num_features, n_cat_features) specifying how many
                    numerical and categorical features to consider at each split.
        num_indices: Array of indices corresponding to numerical features.
        cat_indices: Array of indices corresponding to categorical features.
        trees: A list containing the trained DecisionTree objects.
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        n_features: tuple[int, int] | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees: list[DecisionTree] = []
        self.num_indices: NDArray[np.int_] | None = None
        self.cat_indices: NDArray[np.int_] | None = None

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        """Trains the Random Forest model on the provided dataset.

        Builds `n_trees` decision trees, each trained on a balanced bootstrap
        sample of the data (if applicable for imbalance).

        Args:
            X: The training input samples. Shape (n_samples, n_features).
            y: The target values (class labels). Shape (n_samples,).
        """
        print(f"\nStarting RandomForest.fit with {self.n_trees} trees...")
        print(f" Parameters: max_depth={self.max_depth}, n_features={self.n_features}")
        self.trees = []

        for i in range(self.n_trees):
            print(f"\nBuilding and training tree {i + 1}/{self.n_trees}...")

            # Create a bootstrap sample
            X_sample, y_sample = self.balanced_bootstrap_sample(X, y)

            # Create and train a new tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                num_indices=self.num_indices,
                cat_indices=self.cat_indices,
            )
            tree.fit(X_sample, y_sample)
            print(f" Training of tree {i + 1} done!")
            # Store the trained tree
            self.trees.append(tree)

    def bootstrap_sample(
        self, X: NDArray[np.float32], y: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Creates a standard (imbalanced) bootstrap sample of the data.

        Args:
            X: Input features. Shape (n_samples, n_features).
            y: Target labels. Shape (n_samples,).

        Returns:
            Tuple of (X_sample, y_sample) bootstrapped data.
        """
        n_samples = X.shape[0]
        # Just sample n_samples, with replacement, from the original data
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def balanced_bootstrap_sample(
        self, X: NDArray[np.float32], y: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Creates a balanced bootstrap sample of the data for training a single tree.

        Ensuring an equal number of positive (1) and negative (-1) class samples, based on the count of the minority class.
        If either class is missing, falls back to a standard bootstrap sample.

        Args:
            X: The full training input samples. Shape (n_samples, n_features).
            y: The full training labels {-1, 1}. Shape (n_samples,).

        Returns:
            tuple: A tuple containing (X_sample, y_sample), the bootstrapped data.
        """
        n_samples = X.shape[0]

        # Get indices for positive (1) and negative (-1) classes
        pos_idxs = np.where(y == 1)[0]
        neg_idxs = np.where(y == -1)[0]

        # Find the size of the minority class (positives)
        n_pos = len(pos_idxs)

        # Safety check: If either class is missing, balanced sampling is impossible.
        #    Fall back to a standard (imbalanced) bootstrap for this tree.
        if n_pos == 0 or len(neg_idxs) == 0:
            idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            return X[idxs], y[idxs]

        # Create a balanced sample
        # Sample 'n_pos' indices (with replacement) from the positive class
        pos_sample_idxs = np.random.choice(pos_idxs, size=n_pos, replace=True)
        # Sample 'n_pos' indices (with replacement) from the negative class
        neg_sample_idxs = np.random.choice(neg_idxs, size=n_pos * 2, replace=True)
        # Combine them
        final_idxs = np.concatenate([pos_sample_idxs, neg_sample_idxs])

        return X[final_idxs], y[final_idxs]

    def predict(
        self, X: NDArray[np.float32], threshold: float = 0.4
    ) -> NDArray[np.float32]:
        """Predicts class labels for input samples using the trained Random Forest.

        Aggregates predictions from all trees and applies a threshold to determine
        the final class labels.

        Args:
            X: Input samples to predict. Shape (n_samples, n_features).
            threshold: Decision threshold for positive class. A sample is predicted
                     as positive if the average vote across trees exceeds this threshold.
                     Defaults to 0.4.

        Returns:
            Predicted class labels {-1, 1}. Shape (n_samples,).
        """
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # `tree_preds` shape is (n_trees, n_samples)
        # It contains -1s and 1s.

        # Calculate the average vote for each sample.
        # axis=0 means we average *down the columns* (across the trees).
        # A score of 1.0 means 100% "positive" votes.
        # A score of -1.0 means 100% "negative" votes.
        # A score of 0.0 means a 50/50 split.

        vote_scores = np.mean(tree_preds, axis=0)

        # Instead of a 50% threshold (score > 0), we use a stricter one.
        # A threshold of 0.4 means we need at least 70% of trees
        # to vote positive. ( (70 * 1) + (30 * -1) ) / 100 = 0.4
        final_preds = np.where(vote_scores > threshold, 1, -1)

        return final_preds

    def _most_common_label(self, y: NDArray[np.float32]) -> int:
        """Returns the most common label in the input array.

        Args:
            y: Array of labels.

        Returns:
            The most frequent label. Returns -1 if input is empty.
        """
        if len(y) == 0:
            # Return a sensible default. Since your submission needs {-1, 1},
            # let's pick one. -1 is a safe bet.
            return -1

        classes, counts = np.unique(y, return_counts=True)

        # Find the index of the highest count
        max_count_index = np.argmax(counts)

        # Return the class label associated with that count
        return classes[max_count_index]


def tune_threshold(
    X_val: NDArray[np.float32], y_true: NDArray[np.float32], rf: RandomForest
) -> tuple[float, np.float32]:
    """Finds the optimal prediction threshold to maximize the F1 score on a validation set.
    Iterates through a range of potential thresholds, predicts labels using the RandomForest model,
    calculates the F1 score for each threshold, and returns the threshold that
    yielded the highest F1 score.

    Args:
        X_val: Validation set features. Shape (n_val_samples, n_features).
        y_true: True labels for the validation set. Shape (n_val_samples,).
        rf: Trained RandomForest model instance.

    Returns:
        tuple: (best_threshold, best_f1_score) where:
            best_threshold: The threshold value that resulted in the highest F1 score.
            best_f1_score: The highest F1 score achieved on the validation set.
    """
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true contains values other than -1 and 1.")

    y_true_mapped = np.where(y_true == -1, 0, 1)
    print("\nStarting threshold tuning...")

    # Test at regular intervals
    thresholds = np.linspace(-0.2, 0.2, 20)

    scores = []
    for i, t in enumerate(thresholds):
        print(f"   Testing threshold {i + 1}/{len(thresholds)}: {t:.3f}")
        y_pred = rf.predict(X_val, threshold=t)
        y_pred = np.where(y_pred == -1, 0, 1)
        f1 = f1_score(y_true_mapped, y_pred)
        scores.append(f1)
        print(f"   F1 Score: {f1:.4f}")

    best_idx = np.argmax(scores)
    best_t = thresholds[best_idx]
    best_f1 = scores[best_idx]

    print(len(thresholds), "unique thresholds tested")
    print("Best threshold: ", best_t)
    print("Best F1 score: ", best_f1)

    return best_t, best_f1


def split_train_val(
    y: NDArray[np.float32], val_size: float = 0.2, random_seed: int = 40
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Performs a stratified split of the data indices.

    Args:
        y: The labels of the training set. Shape (n_samples,).
        val_size: The proportion of data to use for validation (0.0 to 1.0).
                 Defaults to 0.2.
        random_seed: Random seed for reproducibility. Defaults to 40.

    Returns:
        tuple: (train_idx, val_idx) where:
            train_idx: The indices of the training set.
            val_idx: The indices of the validation set
    """
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y contains values other than -1 and 1.")

    # Create a local random number generator
    rng = np.random.default_rng(random_seed)

    # The indices of the positive and negative lables, in order to create a balanced stratification
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    # Shuffle the indices
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Calculate the number of validation samples for each class
    n_pos_val = int(len(pos_idx) * val_size)
    n_neg_val = int(len(neg_idx) * val_size)

    # Get validation indices
    val_pos_idx = pos_idx[:n_pos_val]
    val_neg_idx = neg_idx[:n_neg_val]
    val_idx = np.concatenate([val_pos_idx, val_neg_idx])

    # Get training indices (the rest of the data)
    train_pos_idx = pos_idx[n_pos_val:]
    train_neg_idx = neg_idx[n_neg_val:]
    train_idx = np.concatenate([train_pos_idx, train_neg_idx])

    # Shuffle the final arrays one more time so they aren't [all_pos, all_neg]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx
