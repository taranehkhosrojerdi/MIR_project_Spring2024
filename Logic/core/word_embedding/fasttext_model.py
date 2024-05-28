import fasttext
import re
import numpy as np

from tqdm import tqdm
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    
    if minimum_length > 0:
        tokens = [token for token in tokens if len(token) >= minimum_length]
    if stopword_removal:
        stop_words = set(stopwords.words('english'))
        if stopwords_domain:
            stop_words.update(stopwords_domain)
        tokens = [token for token in tokens if token not in stop_words]
    
    return tokens
    

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None
        self.path = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        tmp_training_file_path = "Logic/tests/preprocessed_texts_ft.txt"
        with open(tmp_training_file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                for token in preprocess_text(text):
                    f.write(token + " ")
        
        self.model = fasttext.load_model('FastText_model.bin')
        # self.model = fasttext.train_unsupervised('Logic/tests/preprocessed_texts_ft.txt', model=self.method)
        

    def get_query_embedding(self, query, do_preprocess=False):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = preprocess_text(query)
        return np.asarray(self.model.get_sentence_vector(query))
        

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)
        vec4 = vec2 - vec1 + vec3

        vocab = {word: self.model.get_word_vector(word) for word in self.model.get_words()}

        del vocab[word1]
        del vocab[word2]
        del vocab[word3]

        min_distance = float('inf')
        closest_word = None
        for word, vec in vocab.items():
            dist = distance.cosine(vec4, vec)
            if dist < min_distance:
                min_distance = dist
                closest_word = word

        return closest_word


    def save_model(self, path='Logic/core/word_embedding/dumb_FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.path = path
        self.model.save_model(path)
        

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        fasttext.load_model(self.path)


    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(self.path)
        if mode == 'save' and save:
            self.save_model(path)
            

# if __name__ == "__main__":
# ft_model = fasttext.load_model('FastText_model.bin')
# ft_model = FastText(method='skipgram')
    
# ft_data_loader = FastTextDataLoader('Logic/tests/dumb_IMDB_Crawled.json')

# out = ft_data_loader.create_train_data()
# X, y = out[0], out[1]
# X = X.astype(str).tolist()
# ft_model.train(X)
# ft_model.prepare(None, mode = "save", save=True)

# print(10 * "*" + "Similarity" + 10 * "*")
# word = 'woman'
# neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

# for neighbor in neighbors:
#     print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

# print(10 * "*" + "Analogy" + 10 * "*")
# word1 = "man"
# word2 = "king"
# word3 = "woman"
# print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
