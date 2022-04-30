import math
from typing import NoReturn

from . import linear_discriminant_analysis
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        class_arr = []
        len_y = len(y)

        for j in range(len_y):
            if y[j] not in class_arr:
                class_arr.append(y[j])

        self.classes_ = np.ndarray(class_arr)
        self.classes_ = class_arr
        len_class_arr = len(class_arr)
        self.classes_.sort()
        self.pi_ = np.zeros(len_class_arr, )
        self.mu_ = np.zeros((len_class_arr, (len(X[0]))))
        self.vars_ = np.zeros((len_class_arr, (len(X[0]))))
        for i in range(len_y):
            self.pi_[np.where(self.classes_ == y[i])[0][0]] += 1
        self.pi_ =self.pi_* (1 / len_class_arr)

        for j in range(len_class_arr):
            self.mu_[j] = (X[y == self.classes_[j]]).mean(axis=0)
            self.vars_[j] = (X[y == self.classes_[j]]).var(axis=0)

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
            ret[i] = self.classes_[np.where(l[i] == np.amax(l[i]))[0][0]]
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

        for i in range(len_x):
            for j in range(len_classes):
                prob_j = 1
                a = self.pi_[j]
                for l in range(dimention):
                    temp = (-1/(2 * (self.vars_[j][l])) * ((X[i][l] - self.mu_[j][l])**2))
                    feat_i = 1 / math.sqrt(2 * math.pi * self.vars_[j][l]) * math.exp(temp)
                    prob_j = prob_j*feat_i
                like_l[i][j] = a * prob_j
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
