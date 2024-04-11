import json
import numpy as np
from collections import defaultdict
from Logic.core.preprocess import Preprocessor
from Logic.core.scorer import Scorer
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = '../Logic/core/indexer/index/'
        self.document_indexes = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS),
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
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
            scores = self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)
        final_scores = {}
        self.aggregate_scores(weights=weights, scores=scores, final_scores=final_scores)
        
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
        for field in weights.keys():
            print(scores[field])
            for doc_id in scores[field]:
                final_scores[doc_id] = final_scores.get(doc_id, 0) + weights[field] * scores[field][doc_id]

        final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return final_scores         

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

        for tier in ["first_tier", "second_tier", "third_tier"]:
            if max_results != -1 and self.find_size_of_results(scores) > max_results:
                self.cut_results(scores, max_results)
                break
            
            tier_scores = {}
            for field in weights.keys():    
                #TODO
                local_index = self.tiered_index[field].index[tier]
                all_doc_ids = list(set(doc_id for term_dict in local_index.values() for doc_id in term_dict.keys()))
                doc_lengths = defaultdict(int)
                for doc_id in all_doc_ids:
                    for term in local_index.keys():
                        if doc_id in local_index[term].keys():
                            doc_lengths[doc_id] += local_index[term][doc_id]
                avdl = sum(doc_lengths.values()) / len(doc_lengths) if len(doc_lengths) > 0 else 0
                sc = Scorer(local_index, len(all_doc_ids))

                if method == 'OkapiBM25' and avdl > 0:
                    tier_scores[field] = sc.compute_socres_with_okapi_bm25(query=query, average_document_field_length=avdl, document_lengths=doc_lengths)
                elif method == 'OkapiBM25':
                    tier_scores[field] = 0
                else:
                    tier_scores[field] = sc.compute_scores_with_vector_space_model(query=query, method=method)
            scores = self.merge_scores(scores, tier_scores)

        return scores

            

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
            local_index = self.document_indexes[field].index
            all_doc_ids = list(set(doc_id for term_dict in local_index.values() for doc_id in term_dict.keys()))
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
        fields = []
        if len(scores1.keys()) != 0:
            fields.extend(list(scores1.keys()))
        if len(scores2.keys()) != 0:
            fields.extend(list(scores2.keys()))
        
        for field in fields:
            merged_score[field] = defaultdict(float)
            field_scores1 = scores1.get(field, defaultdict(float))
            field_scores2 = scores2.get(field, defaultdict(float))
            for doc_id in field_scores1.keys():
                if doc_id in field_scores2.keys():
                    merged_score[field][doc_id] = field_scores1[doc_id] + field_scores2[doc_id]
                else:
                    merged_score[field][doc_id] = field_scores1[doc_id]
            for doc_id in field_scores2.keys():
                if doc_id not in field_scores1.keys():
                    merged_score[field][doc_id] = field_scores2[doc_id]
            
        return merged_score
    
    def find_size_of_results(self, merged_scores):
        unique_ids = []
        for field in merged_scores.keys():
            for doc_id in merged_scores[field].keys():
                if doc_id not in unique_ids:
                    unique_ids.append(doc_id)
        return len(unique_ids)
    
    def cut_results(self, merged_scores, max_results: int):
        for field in merged_scores.keys():
            merged_scores[field] = {k: v for k, v in sorted(merged_scores[field].items(), key=lambda item: item[1], reverse=True)}
            if len(merged_scores[field]) > max_results:
                merged_scores[field] = dict(list(merged_scores[field].items())[:max_results])
        return merged_scores




if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "lnc.ltc"
    weights = {
        Indexes.STARS.value: 1,
        Indexes.GENRES.value: 1,
        Indexes.SUMMARIES.value: 1
    }
    result = search_engine.search(query, method, weights)

    print(result)
