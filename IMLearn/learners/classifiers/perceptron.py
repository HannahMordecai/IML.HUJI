from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error, loss_functions


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

     callback_: Callable[[Perceptron, np.ndarray, int], None]
             A callable to be called after each update of the model while fitting to given data
             Callable function should receive as input a Perceptron instance, current sample and current response
     """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response

      """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        len_y = len(y)
        index = 1
        coef_dim = len(X[0])
        org_x = X
        if self.include_intercept_:
            coef_dim += 1
            len_x = len(X)
            new_x = np.zeros((len_x, coef_dim))
            for k in range(len_x):
                new_x[k] = np.r_[[1], X[k]]
            X = new_x
        self.coefs_ = np.zeros(coef_dim, )
        while index < self.max_iter_:
            trans = 0
            for j in range(len_y):
                if (np.dot(self.coefs_.transpose(), X[j]) * y[j]) < 1:
                    trans = 1
                    self.coefs_ = self.coefs_ + y[j] * X[j]
                    self.fitted_ = True
                    self.callback_(self, org_x[j], y[j])
                    break
            if trans == 0:
                return
            index += 1
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        prediction = np.zeros(len(X), )
        if not self.include_intercept_:
            prediction = np.matmul(X, self.coefs_)
        else:
            new_coefs = np.delete(self.coefs_, 0)
            if X.ndim > 1:
                prediction = np.matmul(X, new_coefs) + self.coefs_[0]
            else:
                a = np.matmul(X, new_coefs) + self.coefs_[0]
                a2 = np.zeros(1, )
                a2[0] = a
                prediction = a2
        return np.sign(prediction)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        pred = self._predict(X)
        return loss_functions.misclassification_error(pred, y, True)
