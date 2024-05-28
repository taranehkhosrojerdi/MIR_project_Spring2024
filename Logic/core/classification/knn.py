import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        # super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.x_train = x
        self.y_train = y

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
        
        y_pred = np.empty(x.shape[0], dtype=self.y_train.dtype)
        for i, x_test in tqdm(enumerate(x)):
            distances = np.linalg.norm(self.x_train - x_test, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train[nearest_indices]
            unique_classes, counts = np.unique(nearest_classes, return_counts=True)
            y_pred[i] = unique_classes[np.argmax(counts)]
        return y_pred

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
        y_pred[y_pred == 'positive'] = 1
        y_pred[y_pred == 'negative'] = 0
        y = y.astype(int)
        y_pred = y_pred.astype(int)
        return classification_report(y, y_pred)


# F1 Accuracy : 70%
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
    y_train_encoded = np.array(label_encoder.fit_transform(y_train))
    y_test_encoded = np.array(label_encoder.transform(y_test))
    
    model = KnnClassifier(n_neighbors=1000)
    model.fit(x_train, y_train_encoded)
    report = model.prediction_report(x_test, y_test_encoded)
    print(report)
