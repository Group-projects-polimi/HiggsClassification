"""
Utilities and training functions for a better logistic regression model.
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from metrics import f1_score


def augmentation(
    features: NDArray[np.float32],
    test_features: NDArray[np.float32],
    M: int = 2,
    num_interactions: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Augments and standardizes feature matrices.

    This function calculates the mean and standard deviation from the
    combined train and test sets to scale the data, which is not
    standard practice and might lead to overly optimistic model scores.

    Polynomials: For each original feature (column), it adds new features representing
    that feature raised to the powers 2, 3, ..., M.

    Interactions: Adds 'num_interactions' new features, each being the product
    of two randomly selected *original* features.

    Standardization: The final augmented features are standardized (zero mean,
    unit variance) using only numpy.

    Args:
        features: Training features (shape [n_samples, n_features]).
        test_features: Test features (shape [n_test_samples, n_features]).
        M: The maximum power to include for polynomial terms.
        num_interactions: The number of random 2-way interactions to create.

    Returns:
        A tuple (X_aug_scaled, X_test_aug_scaled) of the augmented and scaled feature arrays.
    """

    original_X_rows = features.shape[0]
    original_features_count = features.shape[1]

    # --- 1. Combine X and X_test to apply transformations identically ---
    concat = np.concatenate((features, test_features), axis=0)

    # Start the list of all feature columns with the originals
    augmented_parts = [concat]

    # --- 2. Polynomial Features ---
    for i in range(original_features_count):
        original_col = concat[:, i : i + 1]  # Keep it as a 2D (N, 1) array
        for power in range(2, M + 1):
            powered_col = np.power(original_col, power)
            augmented_parts.append(powered_col)

    # --- 3. Interaction Features ---
    if num_interactions > 0 and original_features_count >= 2:
        rng = np.random.default_rng()

        for _ in range(num_interactions):
            # Select two *different* indices from the *original* feature set
            idx1, idx2 = rng.choice(original_features_count, 2, replace=False)

            # Extract the full columns
            col1 = concat[:, idx1]
            col2 = concat[:, idx2]

            # Create the interaction term and reshape to (N, 1)
            interaction_col = (col1 * col2).reshape(-1, 1)
            augmented_parts.append(interaction_col)

    # --- 4. Concatenate all parts ---
    concat_aug = np.concatenate(augmented_parts, axis=1)

    # --- 5. Standardize using only numpy (with data leakage) ---

    # Calculate mean and std deviation from the *COMBINED* data
    mean = np.mean(concat_aug, axis=0)
    std = np.std(concat_aug, axis=0)

    # Handle divide-by-zero for constant features
    std_safe = np.where(std == 0, 1.0, std)

    # Apply the (leaked) statistics to the combined set
    concat_scaled = (concat_aug - mean) / std_safe

    # --- 6. Split back into train and test *after* scaling ---
    features_aug_scaled = concat_scaled[:original_X_rows, :]
    test_features_aug_scaled = concat_scaled[original_X_rows:, :]

    return features_aug_scaled, test_features_aug_scaled


def batch_iter_v2(y, X, batch_size, shuffle=True):
    """
    Generate a minibatch iterator for a dataset, covering all samples once per epoch

    Args:
        y: numpy array of shape (N,), data labels
        X: numpy array of shape (N,D), data points
        batch_size: scalar, size of each batch
        shuffle: boolean, whether to shuffle the data

    Yields:
        Tuples of (y, X) with batch_size data points
    """
    N = len(y)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_idx = indices[start_idx:end_idx]
        yield y[batch_idx], X[batch_idx]


def binary_crossentropy_weighted(y, X, w, class_weight):
    """
    Compute the class-weighted binary cross-entropy loss

    Args:
        y: shape=(N,), data labels
        X: shape=(N, D), data points
        w: shape=(D,), model weights
        class_weight: dict or tuple, class weights

    Returns:
        scalar, weighted loss
    """
    w0, w1 = class_weight[0], class_weight[1]
    p = np.clip(stable_sigmoid(X @ w), 1e-12, 1 - 1e-12)
    return -np.mean(w1 * y * np.log(p) + w0 * (1 - y) * np.log(1 - p))


def stable_sigmoid(t):
    """
    Apply sigmoid function on t, stable version

    Args:
        t: scalar or numpy array, values to apply sigmoid function on

    Returns:
        scalar or numpy array, sigmoid function applied on t
    """
    t = np.asarray(t)
    out = np.empty_like(t, dtype=float)

    pos_mask = t >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1 / (1 + np.exp(-t[pos_mask]))
    exp_t = np.exp(t[neg_mask])
    out[neg_mask] = exp_t / (1 + exp_t)

    return out


def compute_bc_gradient_weighted(y, X, w, class_weight):
    """
    Compute the gradient of class-weighted binary cross-entropy loss

    Args:
        y: shape=(N,), data labels
        X: shape=(N, D), data points
        w: shape=(D,), model weights
        class_weight: dict or tuple, class weights

    Returns:
        Numpy array of shape (D,), weighted gradient
    """
    w0, w1 = class_weight[0], class_weight[1]
    p = stable_sigmoid(X @ w)
    weights = np.where(y == 1, w1, w0)
    return X.T.dot(weights * (p - y)) / y.shape[0]


def reg_weighted_logistic_regression(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    lambda_=1e-3,
    initial_w=None,
    max_epochs=100,
    gamma=0.001,
    decay_epochs_ratio=1.0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    class_weights=(1, 1),
    patience=5,
    batch_size=512,
    verbose=1,
):
    """
    Regularized and weighted logistic regression using the Adam optimizer and
    a exponential decay learning rate scheduler

    Args:
        X_train: numpy array of shape (N,D), data points
        y_train: numpy array of shape (N,), data labels
        X_val: numpy array of shape (N,D), data points
        y_val: numpy array of shape (N,), data labels
        lambda_: scalar, regularization parameter
        initial_w: numpy array of shape (D, ), model weights initialization
        max_epochs: scalar, maximum number of epochs
        gamma: scalar, stepsize
        decay_epochs_ratio: scalar, fraction of epoch over which gamma decays
        beta1: scalar, Adam exponential decay rate for 1st moment
        beta2: scalar, Adam exponential decay rate for 2nd moment
        epsilon: scalar, Adam term for numerical stability
        class_weights: dict or tuple, class weights
        patience: scalar, patience for early stopping
        batch_size: scalar, batch size
        verbose: bool, verbosity level

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
        last_best_epoch: scalar, epoch of last best model
    """
    if X_val is None or y_val is None:
        X_val, y_val = X_train, y_train
        if verbose:
            print("Validation sets not provided, using training sets for validation.")

    if not np.all(np.isin(y_train, [0, 1])):
        raise ValueError("y_train contains values other than 0 and 1.")
    elif not np.all(np.isin(y_val, [0, 1])):
        raise ValueError("y_val contains values other than 0 and 1.")

    if initial_w is None:
        initial_w = np.zeros(X_train.shape[1])

    best_w = initial_w.copy()
    current_w = initial_w.copy()
    best_loss = binary_crossentropy_weighted(y_val, X_val, best_w, class_weights)

    last_best_epoch = 0
    gamma0 = gamma
    min_gamma = gamma0 * 0.001

    # iter refers to batch updates, different from epochs
    iterations_per_epoch = (len(y_train) + batch_size - 1) // batch_size
    n_iter = 0

    decay_iters = max(1, max_epochs * decay_epochs_ratio * iterations_per_epoch)

    # Adam 1st and 2nd moment vectors
    m = np.zeros_like(initial_w)
    v = np.zeros_like(initial_w)

    print_interval = (max_epochs + 49) // 50

    for epoch in range(max_epochs):
        for y_batch, tx_batch in batch_iter_v2(
            y_train, X_train, batch_size, shuffle=True
        ):
            # Exponential-decay learning-rate scheduler
            if n_iter <= decay_iters:
                gamma = gamma0 * (min_gamma / gamma0) ** (n_iter / decay_iters)
            else:
                gamma = min_gamma

            # Gradient with regularization (avoids regularizing the first term)
            gradient = compute_bc_gradient_weighted(
                y_batch, tx_batch, current_w, class_weights
            )
            reg_grad = 2 * lambda_ * current_w
            reg_grad[0] = 0
            gradient += reg_grad

            n_iter += 1

            # Adam update
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            m_hat = m / (1 - beta1**n_iter)
            v_hat = v / (1 - beta2**n_iter)

            # Weight update
            current_w -= gamma * m_hat / (np.sqrt(v_hat) + epsilon)

        eval_loss = binary_crossentropy_weighted(y_val, X_val, current_w, class_weights)
        reg_loss = eval_loss + lambda_ * np.sum(current_w[1:] ** 2)

        if eval_loss < best_loss:
            best_w = current_w.copy()
            best_loss = eval_loss
            last_best_epoch = epoch
            if verbose:
                print(
                    "New best loss at epoch {bi}/{ti}: val_loss = {l:.5f} (reg_loss = {rl:.5f}), gamma = {g:.5f}".format(
                        bi=epoch, ti=max_epochs - 1, l=eval_loss, rl=reg_loss, g=gamma
                    )
                )

        # log info
        if verbose and epoch % print_interval == 0:
            print(
                "Adam epoch {bi}/{ti}: val_loss = {l:.5f} (reg_loss = {rl:.5f}), gamma = {g:.5f}, last improvement at epoch {e}".format(
                    bi=epoch,
                    ti=max_epochs - 1,
                    l=eval_loss,
                    rl=reg_loss,
                    g=gamma,
                    e=last_best_epoch,
                )
            )

        if verbose and epoch - last_best_epoch > patience:
            print("Early stopping at epoch {i}".format(i=epoch))
            break

    if verbose:
        print("Best loss: {l}".format(l=best_loss))

    return best_w, best_loss, last_best_epoch


def tune_threshold(p, y_true, threshold_strategy, make_plots):
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true contains values other than 0 and 1.")

    if threshold_strategy == "unique":
        # Test every possible decision boundary, slower
        thresholds = np.unique(p)
    elif threshold_strategy == "rounded":
        # Round possible decision boundaries, faster
        thresholds = np.unique(np.round(p, 4))
    elif threshold_strategy == "linspace":
        # Test at regular intervals, fastest
        thresholds = np.linspace(0, 1, 1000)
    elif threshold_strategy == "quantile":
        # Like linespace, but useful when predictions are crowded towards 0 and 1
        thresholds = np.quantile(p, np.linspace(0, 1, 1000))
    else:
        raise ValueError("Invalid threshold strategy")

    scores = [f1_score(y_true, (p >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(scores)
    best_t = thresholds[best_idx]
    best_f1 = scores[best_idx]

    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(p, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].set_title("Distribution of predicted probabilities (p)")
        axes[0].set_xlabel("Predicted probability")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(thresholds, scores)
        axes[1].set_title("F1 vs Threshold")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("F1 score")

        plt.tight_layout()
        plt.show()

    print(len(thresholds), "unique thresholds tested")
    print("Best threshold: ", best_t)
    print("Best F1 score: ", best_f1)

    return best_t, best_f1


def stratified_kfold_indices(y, n_splits=5, random_seed=40):
    """
    Generate stratified K-fold indices

    Args:
        y: numpy array of shape (N,), data labels
        n_splits: scalar, number of folds
        random_seed: scalar, random seed

    Returns:
        List, (train_idx, val_idx) tuples
    """
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y contains values other than 0 and 1.")

    # Create a local random number generator
    # this is better than setting the seed directly
    # since that would have side effects on other functions
    rng = np.random.default_rng(random_seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds = []
    for i in range(n_splits):
        # Compute validation indices for this fold
        pos_val = pos_idx[i::n_splits]
        neg_val = neg_idx[i::n_splits]
        val_idx = np.concatenate([pos_val, neg_val])

        # Training indices are the complement
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))

    return folds


def cross_validate(
    X,
    y,
    initial_w,
    lambda_grid,
    max_epochs=500,
    gamma=0.001,
    decay_epochs_ratio=1.0,
    class_weights=(1, 1),
    patience=15,
    batch_size=512,
    n_splits=5,
    threshold_strategy="linspace",
    verbose_lambda=1,
    verbose_fold=0,
):
    """
    Perform K-fold cross-validation to tune the lambda_ hyperparameter  based
    on the highest average validation F1-score.

    Args:

    Returns:

    """

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y contains values other than 0 and 1.")

    best_overall_lambda = None
    best_overall_f1 = -1.0
    best_overall_epoch = -1
    best_overall_threshold = -1

    if verbose_lambda:
        print("Starting CV")
        print(f"Testing {len(lambda_grid)} lambda values: {lambda_grid}\n")

    for lambda_ in lambda_grid:
        if verbose_lambda:
            print(f"Testing Lambda = {lambda_}")

        folds = stratified_kfold_indices(y, n_splits)
        fold_best_epochs = []
        fold_best_thresholds = []
        fold_best_f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train the model
            best_w, _, last_best_epoch = reg_weighted_logistic_regression(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                initial_w=initial_w.copy(),
                max_epochs=max_epochs,
                gamma=gamma,
                decay_epochs_ratio=decay_epochs_ratio,
                lambda_=lambda_,
                class_weights=class_weights,
                patience=patience,
                batch_size=batch_size,
                verbose=verbose_fold,
            )

            fold_best_epochs.append(last_best_epoch + 1)  # convert 0-based to count

            y_pred_probs = stable_sigmoid(X_val @ best_w)
            best_threshold, best_f1 = tune_threshold(
                y_pred_probs, y_val, threshold_strategy, make_plots=False
            )

            fold_best_thresholds.append(best_threshold)
            fold_best_f1_scores.append(best_f1)

            if verbose_lambda:
                print(
                    f"    Fold {fold_idx + 1}/{n_splits}: epoch={last_best_epoch + 1}, threshold={best_threshold:.4f}, F1={best_f1:.4f}"
                )

        avg_best_epoch = int(np.mean(fold_best_epochs))
        avg_best_threshold = float(np.mean(fold_best_thresholds))
        avg_best_f1 = float(np.mean(fold_best_f1_scores))

        if verbose_lambda:
            print(
                f"Results for lambda {lambda_}: avg. F1 = {avg_best_f1:.4f}, avg. epochs = {avg_best_epoch}, avg. threshold = {avg_best_threshold:.4f}\n"
            )

        if avg_best_f1 > best_overall_f1:
            best_overall_f1 = avg_best_f1
            best_overall_lambda = lambda_
            best_overall_epoch = avg_best_epoch
            best_overall_threshold = avg_best_threshold

    if verbose_lambda:
        print("CV complete")
        print(f"Best Lambda: {best_overall_lambda}")
        print(f"Best avg. F1 score: {best_overall_f1:.4f}")
        print(f"Associated avg. epochs: {best_overall_epoch}")
        print(f"Associated avg. threshold: {best_overall_threshold:.4f}")

    return best_overall_lambda, best_overall_epoch, best_overall_threshold


def build_model_data(x):
    x = np.c_[np.ones(x.shape[0]), x]
    return x
