import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    num_samples = y_true.size
    squared_diff = y_true - y_pred
    squared_diff = squared_diff ** 2
    sam_samples = sum(squared_diff)
    return (1 / num_samples) * sam_samples


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    # return np.mean(y_pred != y_true)
    len_y = len(y_true)
    sum = 0
    for i in range(len_y):
        if y_pred[i] != y_true[i]:
            sum += 1
    if normalize:
        return sum / len(y_true)
    return sum


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    classes = []
    len_y_true = len(y_true)
    for j in range(len_y_true):
        if y_true[j] not in classes:
            classes.append(y_true[j])
    good = 0
    for i in range(len_y_true):
        if y_true[i] == y_pred[i]:
            good += 1
    return good / len(classes)

    # neg_count, pos_count, true_pos, true_neg = 0, 0, 0, 0
    #
    # len_y_true = len(y_true)
    # for j in range(len_y_true):
    #     if y_true[j] > 0:
    #         pos_count += 1
    #         if y_pred[j] == y_true[j]:
    #             true_neg += 1
    #     else:
    #         neg_count += 1
    #         if y_pred[j] == y_true[j]:
    #             true_pos += 1
    # ret = (true_pos + true_neg) / (pos_count + neg_count)
    # return ret


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
