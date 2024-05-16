import json
import re
from .preprocess import Preprocessor

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        stop_words = []
        with open("../Logic/core/stopwords.txt", "r") as f:
            stop_words = [line.strip() for line in f]
        query = [word for word in query if word not in stop_words]
        query = "".join(query)
        return query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.

        
        modified_query = self.remove_stop_words_from_query(query)
        preprocessor = Preprocessor([doc, modified_query])
        preprocessed_doc, preprocessed_query = preprocessor.snippet_preprocess()
        query_tokens = preprocessed_query.split()
        
        windows = []
        max_occurences = 0
        for token in query_tokens:
            token_occurrences = [pos for pos, word in enumerate(preprocessed_doc.split()) if word.lower() == token.lower()]
            if token_occurrences:
                for occurrence in token_occurrences:
                    start_index = max(0, occurrence - self.number_of_words_on_each_side)
                    end_index = min(len(doc.split()), occurrence + self.number_of_words_on_each_side + 1)
                    snippet_words = preprocessed_doc.split()[start_index:end_index]
                    words = doc.split()[start_index:end_index]
                    highlighted_snippet_words = [f'***{word}***' if snippet_words[idx] in query_tokens else word for idx, word in enumerate(words)]
                    window = " ".join(highlighted_snippet_words)
                    if window.count('***') > max_occurences:
                        windows = [window]
                        max_occurences = window.count('***')
                    elif window.count('***') == max_occurences:
                        windows.append(window)
                        
            else:
                not_exist_words.append(token)
                
        for window in windows:      
            final_snippet += window + " ... "
        final_snippet = final_snippet.strip()
        return final_snippet, not_exist_words
# ---------------------------------------------Test-------------------------------------------
# with open('./LSHFakeData.json') as f:
#     data = json.load(f)
# documents = [item['summaries'][0] for item in data]
# snippet = Snippet()
# print(snippet.find_snippet(query="city jkhj Neon", doc=documents[2])[1])
