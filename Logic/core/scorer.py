import numpy as np
from collections import defaultdict, OrderedDict
import json
import math
from .indexer.indexes_enum import Indexes
from .indexer.index_reader import Index_reader



class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            df = len(self.index.get(term, []))
            idf = np.log(self.N / df) if df > 0 else 0
            self.idf[term] = idf
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        #TODO
        query_tfs = defaultdict(int)
        for term in query:
            query_tfs[term] += 1
        return dict(query_tfs)


    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        all_doc_ids = list(set(doc_id for term_dict in self.index.values() for doc_id in term_dict.keys()))
        term_idfs = {term: self.get_idf(term) for term in self.index.keys()}

        term_document_count_matrix = np.zeros((len(self.index), len(all_doc_ids)))
        for i, term in enumerate(list(self.index.keys())):
            for j, doc in enumerate(all_doc_ids):
                if doc in self.index[term].keys():
                    term_document_count_matrix[i, j] = self.index[term][doc]
                
        if method[0] == 'l':
            term_document_count_matrix = np.where(term_document_count_matrix > 0, 1 + np.log(term_document_count_matrix + 1e-10), 0)                
                    
        if method[1] == 't':
            term_document_count_matrix = term_document_count_matrix * np.array(list(term_idfs.values()))[:, None]
     
        if method[2] == 'c':
            doc_norm = 0
            for i in range(term_document_count_matrix.shape[0]):
                for j in range(term_document_count_matrix.shape[1]):
                    doc_norm += (term_document_count_matrix[i, j] * term_document_count_matrix[i, j])
            doc_norm = 1 / math.sqrt(doc_norm)
            for i in range(term_document_count_matrix.shape[0]):
                for j in range(term_document_count_matrix.shape[1]):
                    term_document_count_matrix[i, j] *= doc_norm           
    
        query_matrix = np.zeros(len(self.index))
        query_tfs = self.get_query_tfs(query)
        for i, term in enumerate(self.index.keys()):
            if term in query_tfs:
                query_matrix[i] = query_tfs[term]

        if method[4] == 'l':
            query_matrix = np.where(query_matrix > 0, 1 + np.log(query_matrix + 1e-10), 0)
                
        if method[5] == 't':
            query_matrix = query_matrix * np.array(list(term_idfs.values()))
                
        if method[6] == 'c':
            query_norm = 0
            for i in range(query_matrix.shape[0]):
                query_norm += (query_matrix[i] * query_matrix[i])
            query_norm = 1 / math.sqrt(query_norm) if query_norm != 0 else 0
            if query_norm > 0:
                for ele in query_matrix:
                    ele *= query_norm
       
        score_vector = np.dot(query_matrix, term_document_count_matrix)
        results = {doc_id: score for doc_id, score in zip(all_doc_ids, score_vector)}
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
#       results = OrderedDict(zip(all_doc_ids, score_vector))
#       results = OrderedDict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        return results
                
        
        


    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        #TODO
        pass

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths, k=1.2, b=0.75):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        
        all_doc_ids = list(set(doc_id for term_dict in self.index.values() for doc_id in term_dict.keys()))

        scores = defaultdict(int)
        for i, doc in enumerate(all_doc_ids):
            for term in query:
                if term in self.index.keys():
                    idf = self.get_idf(term)
                    tf = self.index[term][doc] if doc in list(self.index[term].keys()) else 0
                    normalization_factor = (1 - b) + (b * (document_lengths[doc] / average_document_field_length))
                    scores[doc] += (idf * (k + 1) * tf) / ((k * normalization_factor) + tf)
        
        scores = {key: value for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        return scores
                    

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO
        pass
    
# ------------------------------------------------Test-----------------------------------------------

# index = Index_reader(path='Logic/core/indexer/index/', index_name=Indexes.SUMMARIES).index
# all_doc_ids = []
# for term in index.keys():
#     all_doc_ids.extend(index[term].keys())
# all_doc_ids = list(set(all_doc_ids))
# doc_lengths = defaultdict(int)
# for doc_id in all_doc_ids:
#     for term in index.keys():
#         if doc_id in index[term].keys():
#             doc_lengths[doc_id] += index[term][doc_id]
# avdl = sum(doc_lengths.values()) / len(doc_lengths)

# scorer = Scorer(index, len(all_doc_ids))

# query = ['Joker', 'Batman']
# result = scorer.compute_scores_with_vector_space_model(query=query, method = 'lnc.ltc')

# for key, value in {k: result[k] for k in list(result)[:5]}.items():
#     print(key, value)
    