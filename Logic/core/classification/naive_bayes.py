
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd
import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader

class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.classes, counts = np.unique(y, return_counts=True)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = counts / self.number_of_samples
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        
        for idx, cls in enumerate(self.classes):
            x_cls = x[y == cls]
            feature_sum = np.sum(x_cls, axis=0)
            self.feature_probabilities[idx, :] = (feature_sum + self.alpha) / (np.sum(feature_sum) + self.alpha * self.number_of_features)

        self.log_probs = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_likelihood + self.log_prior
        return self.classes[np.argmax(log_posterior, axis=1)]

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        # I took sentences as labels
        return np.sum(sentences == 1) / len(sentences)


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
                """
    data_loader = ReviewLoader('Logic/tests/IMDB Dataset.csv')
    data_loader.load_data()
    reviews, labels = data_loader.review_tokens, data_loader.sentiments

    reviews = np.array(reviews)
    labels = np.array(labels)
    vectorizer = CountVectorizer(max_features=5000)
    if reviews.shape == labels.shape:
        x = vectorizer.fit_transform(reviews).toarray()
        y = np.array(labels)
        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    nb = NaiveBayes(count_vectorizer=vectorizer)
    nb.fit(x_train, y_train)

    print(nb.prediction_report(x_test, y_test))
