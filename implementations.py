import numpy as np


def mean_squared_error(y, X, w):
    """
    Compute the cost by mean squared error

    Args:
        y:  shape=(N, ), data labels
        X: shape=(N, D), data points
        w:  shape=(D, ), model weights

    Returns:
        scalar, loss
    """
    e = y - np.dot(X, w)
    return np.dot(e.T, e) / (2 * y.shape[0])


def compute_mse_gradient(y, X, w):
    """
    Compute the gradient of the mse loss at w

    Args:
        y: numpy array of shape (N, ), data labels
        X: numpy array of shape (N,D), data points
        w: numpy array of shape (D, ), model weights

    Returns:
        Numpy array of shape (D, ), gradient of the loss at w.
    """
    return -np.dot(X.T, y - np.dot(X, w)) / y.shape[0]


def sigmoid(t):
    """
    Apply sigmoid function on t

    Args:
        t: scalar or numpy array, values to apply sigmoid function on

    Returns:
        scalar or numpy array, sigmoid function applied on t
    """
    return 1 / (1 + np.exp(-t))


def binary_crossentropy(y, X, w):
    """
    Compute the cost by negative log likelihood

    Args:
        y:  shape=(N, ), data labels
        X: shape=(N, D), data points
        w:  shape=(D, ), model weights

    Returns:
        scalar, loss
    """
    return (
        -np.sum(y * np.log(sigmoid(X @ w)) + (1 - y) * np.log(1 - sigmoid(X @ w)))
        / y.shape[0]
    )


def compute_bc_gradient(y, X, w):
    """
    Compute the gradient of the negative log likelihood loss at w

    Args:
        y:  shape=(N, ), data labels
        X: shape=(N, D), data points
        w:  shape=(D, ), model weights

    Returns:
        Numpy array of shape (D, ), gradient of the loss at w
    """
    return X.T.dot(sigmoid(X @ w) - y) / y.shape[0]


def mean_squared_error_gd(y, X, initial_w, max_iters, gamma):
    """
    Gradient Descent algorithm

    Args:
        y: numpy array of shape (N, ), data labels
        X: numpy array of shape (N,D), data points
        initial_w: numpy array of shape=(D, ), model weights initialization
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape=(D, ), final model weights
        best_loss: scalar, final loss
    """
    best_w = initial_w.copy()
    current_w = initial_w.copy()
    best_loss = mean_squared_error(y, X, best_w)

    print_interval = (max_iters + 9) // 10

    for n_iter in range(max_iters):
        g = compute_mse_gradient(y, X, current_w)
        current_w -= gamma * g

        loss = mean_squared_error(y, X, current_w)

        if loss < best_loss:
            best_w = current_w.copy()
            best_loss = loss

        if n_iter % print_interval == 0:
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss


def batch_iter(y, X, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Args:
        y: numpy array of shape (N, ), data labels
        X: numpy array of shape (N,D), data points
        batch_size: scalar, minibatch
        num_batches: scalar, number of batches
        shuffle: boolean, whether to shuffle the data

    Yields:
        Tuples of (y, X) with batch_size data points
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)

    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start_index in idxs:
        end_index = start_index + batch_size
        yield y[start_index:end_index], X[start_index:end_index]


def mean_squared_error_sgd(y, X, initial_w, max_iters, gamma):
    """
    Stochastic Gradient Descent algorithm

    Args:
        y: numpy array of shape (N, ), data labels
        X: numpy array of shape (N,D), data points
        initial_w: numpy array of shape (D, ), model weights initialization
        batch_size: scalar, number of data points in a mini-batch
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
    """
    best_w = initial_w.copy()
    current_w = initial_w.copy()
    best_loss = mean_squared_error(y, X, best_w)

    print_interval = (max_iters + 9) // 10

    n_iter = 0
    for minibatch_y, minibatch_x in batch_iter(
        y, X, 1, num_batches=max_iters, shuffle=True
    ):
        g = compute_mse_gradient(minibatch_y, minibatch_x, current_w)
        current_w -= gamma * g

        loss = mean_squared_error(y, X, current_w)

        if loss < best_loss:
            best_w = current_w.copy()
            best_loss = loss

        if n_iter % print_interval == 0:
            print(
                "SGD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

        n_iter += 1

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss


def least_squares(y, X):
    """
    Compute the least squares solution.

    Args:
        y: numpy array of shape (N,), data labels
        X: numpy array of shape (N,D), data points

    Returns:
        w: numpy array of shape (D,), final model weights
        loss: scalar, final loss
    """
    w = np.linalg.solve(X.T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ y)
    return w, mean_squared_error(y, X, w)


def ridge_regression(y, X, lambda_):
    """
    Ridge regression.

    Args:
        y: numpy array of shape (N,), data labels
        X: numpy array of shape (N,D), data points
        lambda_: scalar, regularization parameter

    Returns:
        w: numpy array of shape (D,), final model weights
        loss: scalar, final loss
    """
    aI = 2 * X.shape[0] * lambda_ * np.identity(X.shape[1])
    w = np.linalg.solve(X.T.dot(X) + aI, X.T.dot(y))
    return w, mean_squared_error(y, X, w)


def logistic_regression(y, X, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    Args:
        y: numpy array of shape (N,), data labels
        X: numpy array of shape (N,D), data points
        initial_w: numpy array of shape (D, ), model weights initialization
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
    """
    best_w = initial_w.copy()
    current_w = initial_w.copy()
    best_loss = binary_crossentropy(y, X, best_w)

    print_interval = (max_iters + 9) // 10

    for n_iter in range(max_iters):
        gradient = compute_bc_gradient(y, X, current_w)
        current_w -= gamma * gradient

        loss = binary_crossentropy(y, X, current_w)

        if loss < best_loss:
            best_w = current_w.copy()
            best_loss = loss

        if n_iter % print_interval == 0:
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss


def reg_logistic_regression(y, X, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    Args:
        y: numpy array of shape (N,), data labels
        X: numpy array of shape (N,D), data points
        lambda_: scalar, regularization parameter
        initial_w: numpy array of shape (D, ), model weights initialization
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
    """
    best_w = initial_w.copy()
    current_w = initial_w.copy()
    best_loss = binary_crossentropy(y, X, best_w) + lambda_ * np.squeeze(
        best_w.T @ best_w
    )

    print_interval = (max_iters + 9) // 10

    for n_iter in range(max_iters):
        gradient = compute_bc_gradient(y, X, current_w) + 2 * lambda_ * current_w
        current_w -= gamma * gradient

        loss = binary_crossentropy(y, X, current_w) + lambda_ * np.squeeze(
            current_w.T @ current_w
        )

        if loss < best_loss:
            best_w = current_w.copy()
            best_loss = loss

        # log info
        if n_iter % print_interval == 0:
            print(
                "GD iter. {bi}/{ti}: loss (with regularization) = {l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    print("Best loss: {l}".format(l=best_loss))
    return best_w, binary_crossentropy(y, X, best_w)
