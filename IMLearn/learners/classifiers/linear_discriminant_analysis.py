import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        class_arr = []
        len_y = len(y)
        len_x = len(X)
        for j in range(len_y):
            if y[j] not in class_arr:
                class_arr.append(y[j])

        len_class_arr = len(class_arr)
        self.pi_ = np.zeros(len_class_arr, )
        self.mu_ = np.zeros((len_class_arr, len(X[0])))
        self.classes_ = np.ndarray(class_arr, )
        self.classes_ = class_arr
        self.classes_.sort()
        new_cov = np.zeros((len(X[0]), len(X[0])))

        for k in range(len_class_arr):
            r = 0
            for i in range(len_x):
                if y[i] == self.classes_[k]:
                    self.mu_[k] = self.mu_[k] + X[i]
                    r += 1
            if r != 0:
                self.mu_[k] *= (1 / r)

        for i in range(len_y):
            self.pi_[np.where(self.classes_ == y[i])[0][0]] += 1
        self.pi_ *= (1 / len_y)

        for j in range(len_x):
            y_i_cur = np.where(self.classes_ == y[j])[0][0]
            temp = X[j] - self.mu_[y_i_cur]
            new_cov = new_cov + np.matmul(temp.reshape((-1, 1)),temp.reshape((1, -1)))
        self.cov_ = new_cov * (1 / (len_x - len_class_arr))
        self._cov_inv = inv(self.cov_)

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
        len_x = len(X)
        ret = np.zeros(len_x, )
        l = self.likelihood(X)
        for i in range(len_x):
            ret[i] = self.classes_[np.where(l[i]== np.amax(l[i]))[0][0]]
        return ret

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        len_x = len(X)
        len_classes = len(self.classes_)
        like_l = np.zeros((len_x, len_classes))
        dimention = len(X[0])
        det_val = det(self.cov_)

        for i in range(len_x):
            for j in range(len_classes):
                a = self.pi_[j]
                b = 1 / (math.sqrt(det_val * ((2 * math.pi) ** dimention)))
                temp = (X[i] - self.mu_[j]).reshape((1, -1))
                temp_rev = (X[i] - self.mu_[j]).reshape((-1, 1))
                c = (-0.5) * (np.matmul(np.matmul(temp, self._cov_inv), temp_rev))
                like_l[i][j] = a * (b * math.exp(c))
        return like_l

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
        prediction = self._predict(X)
        return misclassification_error(y, prediction)
