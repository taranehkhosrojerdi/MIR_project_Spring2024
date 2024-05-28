import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import fasttext

from ..word_embedding.fasttext_model import FastText
from ..utility.preprocess import Preprocessor


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText(method='skipgram')
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        self.fasttext_model.model = fasttext.load_model('FastText_model.bin')
        self.review_tokens = df['review'].astype(str).tolist()
        self.review_tokens = Preprocessor(self.review_tokens).preprocess()
        # df['review'] = self.review_tokens
        # df.to_csv(self.file_path, index=False)    
        
        self.sentiments = df['sentiment'].astype(str).tolist()
        # self.fasttext_model.train(self.review_tokens)
        

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        for review in self.review_tokens:
            self.embeddings.append(self.fasttext_model.get_query_embedding(review))
        

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(self.review_tokens, self.sentiments, test_size=test_data_ratio, random_state=42)
        return x_train, x_test, y_train, y_test

