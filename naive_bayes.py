from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import itertools

class NaiveBayes(object):
    def __init__(self, alpha=1):
        """
        INPUT:
        -alpha: float, laplace smoothing constant.
        """
        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)
        self.class_counts = None
        self.alpha = alpha
        self.p = None

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        Compute the totals for each class and the totals for each feature
        and class.
        '''

        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)

        for i, y_i in enumerate(y):
            self.class_totals[y_i] += len(X[i])

        for i, y_i in enumerate(y):
                self.class_feature_totals[y_i] = merge_dicts(self.class_feature_totals[y_i], Counter(X[i]))

    def fit(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        OUTPUT: None
        '''
        #Compute priors
        self.class_counts = Counter(y)

        #Compute number of features.
        self.p = len(set(itertools.chain(*X)))

        #Compute likelihoods
        self._compute_likelihood(X, y)

    def posteriors(self, X):
        '''
        INPUT:
        - X: List of list of tokens.
        OUTPUT: A list of counters. The keys of the counter
        will correspond to the possible labels, and the values
        will be the likelihood. 
        '''

        posteriors = []
        y_total = sum([self.class_counts[key] for key in self.class_counts])
        for j, x in enumerate(X):
            posterior = defaultdict(Counter)
            for y_i in self.class_counts:
                posterior[y_i] = np.log(self.class_counts[y_i] / float(y_total))
                for word in x:
                    posterior[y_i] += np.log((self.class_feature_totals[y_i][word] + self.alpha) / float(self.class_totals[y_i] + self.alpha * self.p))
            posteriors.append(Counter(posterior))
        return posteriors


    def predict(self, X):
        """
        INPUT:
        - X A list of lists of tokens.
        OUTPUT:
        -predictions: a numpy array with predicted labels.
        """
        return np.array([label.most_common(1)[0][0]
                         for label in self.posteriors(X)])

    def score(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels
        OUTPUT:
        - accuracy: float between 0 and 1
        Calculate the accuracy, the percent predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))

def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
