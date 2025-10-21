"""
    Compute the cost by mean squared error

    Args:
        y:  shape=(N, ), data labels
        tx: shape=(N, D), data points
        w:  shape=(D, ), model weights

    Returns:
        scalar, loss
    """
def mean_squared_error(y, tx, w):
    e = y - np.dot(tx, w)
    return np.dot(e.T, e) / (2 * y.shape[0])

"""
    Compute the gradient of the mse loss at w

    Args:
        y: numpy array of shape (N, ), data labels
        tx: numpy array of shape (N,D), data points
        w: numpy array of shape (D, ), model weights

    Returns:
        Numpy array of shape (D, ), gradient of the loss at w.
"""
def compute_mse_gradient(y, tx, w):
    return -np.dot(tx.T, y - np.dot(tx, w)) / y.shape[0]

"""
    Gradient Descent algorithm

    Args:
        y: numpy array of shape (N, ), data labels
        tx: numpy array of shape (N,D), data points
        initial_w: numpy array of shape=(D, ), model weights initialization
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape=(D, ), final model weights
        best_loss: scalar, final loss
"""
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    best_w = initial_w
    best_loss = mean_squared_error(y, tx, best_w)

    current_w = initial_w

    for n_iter in range(max_iters):
        loss = mean_squared_error(y, tx, current_w)
        g = compute_mse_gradient(y, tx, current_w)
        current_w -= gamma * g

        if loss < best_loss:
            best_loss = loss
            best_w = current_w

        if n_iter % (max_iters // 10) == 0:
          print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters-1, l=loss))

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss

"""
    Generate a minibatch iterator for a dataset.

    Args:
        y: numpy array of shape (N, ), data labels
        tx: numpy array of shape (N,D), data points
        batch_size: scalar, minibatch

    Returns:
        Iterator, iterates over mini-batches of `batch_size` matching elements from `y` and `X`
"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)
    batch_size = min(data_size, batch_size)

    max_batches = int(data_size / batch_size)
    remainder = (data_size - max_batches * batch_size)

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
        yield y[start_index:end_index], tx[start_index:end_index]

"""
    Stochastic Gradient Descent algorithm

    Args:
        y: numpy array of shape (N, ), data labels
        tx: numpy array of shape (N,D), data points
        initial_w: numpy array of shape (D, ), model weights initialization
        batch_size: scalar, number of data points in a mini-batch
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
"""
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    best_w = initial_w
    best_loss = mean_squared_error(y, tx, best_w)

    current_w = initial_w

    n_iter = 0
    for minibatch_y, minibatch_x in batch_iter(y, tx, 1, num_batches=max_iters, shuffle=True):
        loss = mean_squared_error(y, tx, current_w)
        g = compute_mse_gradient(minibatch_y, minibatch_x, current_w)
        current_w -= gamma * g

        if loss < best_loss:
            best_loss = loss
            best_w = current_w

        if n_iter % (max_iters // 10) == 0:
          print("SGD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

        n_iter += 1

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss

"""
    Compute the least squares solution.

    Args:
        y: numpy array of shape (N,), data labels
        tx: numpy array of shape (N,D), data points

    Returns:
        w: numpy array of shape (D,), final model weights
        loss: scalar, final loss
    """
def least_squares(y, tx):
    w = np.linalg.solve(tx.T @ tx + 1e-8 * np.eye(tx.shape[1]), tx.T @ y)
    return w, mean_squared_error(y, tx, w)

"""
    Ridge regression.

    Args:
        y: numpy array of shape (N,), data labels
        tx: numpy array of shape (N,D), data points
        lambda_: scalar, regularization parameter

    Returns:
        w: numpy array of shape (D,), final model weights
        loss: scalar, final loss
"""
def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    return w, mean_squared_error(y, tx, w)

"""
    Logistic regression using GD

    Args:
        y: numpy array of shape (N,), data labels
        tx: numpy array of shape (N,D), data points
        initial_w: numpy array of shape (D, ), model weights initialization
        max_iters: scalar, total number of iterations
        gamma: scalar, stepsize

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
"""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    best_w = initial_w
    best_loss = binary_crossentropy(y, tx, best_w)

    current_w = initial_w

    for n_iter in range(max_iters):
        loss = binary_crossentropy(y, tx, current_w)
        gradient = compute_bc_gradient(y, tx, current_w)
        current_w -= gamma * gradient

        if loss < best_loss:
            best_loss = loss
            best_w = current_w

        if iter % (max_iters // 10) == 0:
            print("GD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    print("Best loss: {l}".format(l=best_loss))
    return best_w, best_loss

"""
    Regularized logistic regression using gradient descent

    Args:
        y: numpy array of shape (N,), data labels
        tx: numpy array of shape (N,D), data points
        lambda_: scalar, regularization parameter
        initial_w: numpy array of shape (D, ), model weights initialization

    Returns:
        best_w: numpy array of shape (D, ), final model weights
        best_loss: scalar, final loss
"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    best_w = initial_w
    best_loss = binary_crossentropy(y, tx, best_w)

    current_w = initial_w

    for n_iter in range(max_iters):
      loss = binary_crossentropy(y, tx, current_w) + lambda_ * np.squeeze(current_w.T @ current_w)
      gradient = compute_bc_gradient(y, tx, current_w) + 2 * lambda_ * current_w
      current_w -= gamma * gradient

      if loss < best_loss:
        best_loss = loss
        best_w = current_w

        # log info
        if n_iter % (max_iters // 10) == 0:
            print("GD iter. {bi}/{ti}: loss (with regularization) = {l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    print("Best loss: {l}".format(l=best_loss))
    return best_w, binary_crossentropy(y, tx, best_w)
