from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json
import numpy as np
from collections import defaultdict

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        
        #TODO
        self.path = path
        self.index, self.documents = self.read_documents()
        self.metadata_index = self.create_metadata_index()
        

    def read_documents(self):
        """
        Reads the documents.
        
        """
        
        #TODO
        index = {}
        # self.index['documents'] = Index_reader(self.path, index_name=Indexes.DOCUMENTS).index
        index[Indexes.STARS.value] = Index_reader(self.path, index_name=Indexes.STARS).index
        index[Indexes.GENRES.value] = Index_reader(self.path, index_name=Indexes.GENRES).index
        index[Indexes.SUMMARIES.value] = Index_reader(self.path, index_name=Indexes.SUMMARIES).index
        document = Index_reader(self.path, index_name=Indexes.DOCUMENTS).index
        return index, document

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length(Indexes.STARS.value),
            'genres': self.get_average_document_field_length(Indexes.GENRES.value),
            'summaries': self.get_average_document_field_length(Indexes.SUMMARIES.value)
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        #TODO
        local_index = self.index[where]
        sum_dict = defaultdict(int)
        for term in local_index.keys():
            for doc_id in local_index[term].keys():
                sum_dict[doc_id] += local_index[term][doc_id]
        
        return np.mean(list(sum_dict.values()))
        
        


    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


if __name__ == "__main__":
    meta_index = Metadata_index(path='index/')
    meta_index.store_metadata_index(path='index/')
