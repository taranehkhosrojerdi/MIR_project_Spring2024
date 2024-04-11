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
        with open('../stopwords.txt', 'r') as file:
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

