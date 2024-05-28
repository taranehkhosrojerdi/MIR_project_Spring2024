import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import chain

# from Logic.core.utility.preprocess import Preprocessor

import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import json # for test

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        self.stopwords = []
        with open('Logic/core/stopwords.txt', 'r') as file:
            self.stopwords = [line.strip() for line in file]

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        # TODO
        preprocessed_documents = []
        for doc in self.documents:
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            doc = self.normalize(doc)
            doc = self.remove_stopwords(doc)
            doc = doc.lower()
            preprocessed_documents.append(doc)
        return preprocessed_documents
    
    def snippet_preprocess(self):
        preprocessed_documents = []
        for doc in self.documents:
            doc = self.remove_punctuations(doc)
            doc = self.normalize(doc)
            preprocessed_documents.append(doc)
        return preprocessed_documents    
    

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # TODO
        text = text.lower()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        words = self.tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
        normalized_text = stemmed_words[0]
        for i in range(1, len(stemmed_words)):
            normalized_text = normalized_text + " " + stemmed_words[i]

        return normalized_text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        # TODO
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

            Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        # TODO
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in text:
            if ele in punc:
                text = text.replace(ele, "")
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        # TODO
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        # TODO
        words = self.tokenize(text)
        for i, word in enumerate(words):
            if word in self.stopwords:
                words.pop(i)
        text = ' '.join(words)
        return text
# ---------------------------------------------Test-------------------------------------------
# with open('./Logic/core/LSHFakeData.json') as f:
#     data = json.load(f)
# documents = ["Salam., salam, agarga aregta https://google.com/search"]
# preprocessor = Preprocessor(documents)
# documents = preprocessor.preprocess()
# for doc in documents:
#     print(doc)



class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        pass

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        df = pd.read_json(self.file_path)
        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        df['reviews'] = df['reviews'].apply(lambda x: ' '.join(list(chain.from_iterable(x))) if isinstance(x, list) else x)
        df['text'] = df[['synopsis', 'summaries', 'reviews', 'title']].astype(str).agg(' '.join, axis=1)

        preprocessor = Preprocessor(df['text'].astype(str).tolist())
        df['text'] == preprocessor.preprocess()
        
        labelEncoder = LabelEncoder()
        all_genres = [genre for sublist in df['genres'] for genre in sublist]
        labelEncoder.fit(all_genres)
        
        X = []
        y = []
        
        for idx, row in df.iterrows():
            for genre in row['genres']:
                X.append(row['text'])
                y.append(labelEncoder.transform([genre])[0])
        
        output = (np.asarray(X), np.asarray(y))
        return output
                
                
# data_loader = FastTextDataLoader('Logic/test/IMDB_crawled.json')
# X, y = data_loader.create_train_data()
# print(X, y)