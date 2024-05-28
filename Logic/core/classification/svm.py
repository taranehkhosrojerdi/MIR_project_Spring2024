import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        # super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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
        y_pred[y_pred == 'negative'] = int(0)
        y_pred[y_pred == 'positive'] = int(1)
        y = y.astype(int)
        y_pred = y_pred.astype(int)
        return classification_report(y, y_pred)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = 'Logic/tests/IMDB Dataset.csv'
    loader = ReviewLoader(file_path)
    loader.load_data()
    loader.get_embeddings()
    
    x_train, x_test, y_train, y_test = loader.split_data(test_data_ratio=0.2)
        
    x_train = np.array([loader.fasttext_model.get_query_embedding(text) for text in x_train])
    x_test = np.array([loader.fasttext_model.get_query_embedding(text) for text in x_test])
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    model = SVMClassifier()
    model.fit(x_train, y_train)
    report = model.prediction_report(x_test, y_test_encoded)
    print(report)
    
