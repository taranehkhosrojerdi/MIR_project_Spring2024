import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embeddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for embeddings, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{self.num_epochs}"):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        with torch.no_grad():
            for embeddings, _ in test_loader:
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())

        return np.array(all_predictions)

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        eval_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                outputs = model(embeddings)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        eval_loss /= len(dataloader)
        f1_score_macro = f1_score(all_labels, all_predictions, average='macro')
        return eval_loss, all_predictions, all_labels, f1_score_macro   

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = 'Logic/tests/IMDB Dataset.csv'
    loader = ReviewLoader(file_path)
    loader.load_data()
    loader.get_embeddings()

    x_train_texts, x_test_texts, y_train, y_test = loader.split_data(test_data_ratio=0.2)

    x_train = np.array([loader.fasttext_model.get_query_embedding(text) for text in x_train_texts])
    x_test = np.array([loader.fasttext_model.get_query_embedding(text) for text in x_test_texts])

    label_encoder = LabelEncoder()
    y_train_encoded = np.array(label_encoder.fit_transform(y_train))
    y_test_encoded = np.array(label_encoder.transform(y_test))

    classifier = DeepModelClassifier(in_features=x_train.shape[1], num_classes=len(np.unique(y_train_encoded)), batch_size=32, num_epochs=50)

    classifier.set_test_dataloader(x_test, y_test_encoded)

    classifier.fit(x_train, y_train_encoded)

    report = classifier.prediction_report(x_test, y_test_encoded)
    print(report)
