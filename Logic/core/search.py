import json
import numpy as np
from collections import defaultdict
from .preprocess import Preprocessor
from .scorer import Scorer
from .indexer.indexes_enum import Indexes, Index_types
from .indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = '/index'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)

    def search(self, query, method, weights, safe_ranking = True, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        for field in weights:
            field_scores = scores[field]
            for term in field_scores.keys():
                for doc_id in field_scores[term].keys():
                    final_scores[doc_id] += weights[field] * field_scores[term][doc_id]
        final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """

        for field in weights.keys():
            
            scores[field] = {}
            for tier in ["first_tier", "second_tier", "third_tier"]:
                #TODO
                local_index = self.tiered_index[field][tier]
                all_doc_ids = list(set(doc_id for term_dict in local_index.values() for doc_id in local_index.keys()))
                doc_lengths = defaultdict(int)
                for doc_id in all_doc_ids:
                    for term in local_index.keys():
                        if doc_id in local_index[term].keys():
                            doc_lengths[doc_id] += local_index[term][doc_id]
                avdl = sum(doc_lengths.values()) / len(doc_lengths)
                sc = Scorer(local_index, len(all_doc_ids))

                if method == 'OkapiBM25':
                    scores[field] = sc.compute_socres_with_okapi_bm25(query=query, average_document_field_length=avdl, document_lengths=doc_lengths)
                else:
                    scores[field] = sc.compute_scores_with_vector_space_model(query=query, method=method)

            

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """

        for field in weights.keys():
            #TODO
            local_index = self.tiered_index[field]
            all_doc_ids = list(set(doc_id for term_dict in local_index.values() for doc_id in local_index.keys()))
            doc_lengths = defaultdict(int)
            for doc_id in all_doc_ids:
                for term in local_index.keys():
                    if doc_id in local_index[term].keys():
                        doc_lengths[doc_id] += local_index[term][doc_id]
            avdl = sum(doc_lengths.values()) / len(doc_lengths)
            sc = Scorer(local_index, len(all_doc_ids))

            if method == 'OkapiBM25':
                scores[field] = sc.compute_socres_with_okapi_bm25(query=query, average_document_field_length=avdl, document_lengths=doc_lengths)
            else:
                scores[field] = sc.compute_scores_with_vector_space_model(query=query, method=method)
            
        return scores


    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """

        #TODO
        merged_score = {}
        


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "lnc.ltc"
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights)

    print(result)
