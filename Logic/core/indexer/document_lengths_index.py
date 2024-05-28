import json

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.indexer.indexes_enum import Indexes,Index_types
from Logic.core.indexer.index_reader import Index_reader
from collections import defaultdict

class DocumentLengthsIndex:
    def __init__(self,path='Logic/core/indexer/index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """

        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES.value)
        }
        self.store_document_lengths_index(path, Indexes.STARS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.SUMMARIES)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """

        # TODO:
        document_lengths = defaultdict(int)
        for doc_id, doc_data in self.documents_index.items():
            num_of_tokens = 0
            for item in doc_data.get(where, []):
                num_of_tokens += len(item.split())
            document_lengths[doc_id] = num_of_tokens
        return document_lengths
        
    
    def store_document_lengths_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex()
    print('Document lengths index stored successfully.')
    document_lengths_index.store_document_lengths_index(path='Logic/core/indexer/index/', index_name=Indexes.STARS)
    document_lengths_index.store_document_lengths_index(path='Logic/core/indexer/index/', index_name=Indexes.SUMMARIES)
    document_lengths_index.store_document_lengths_index(path='Logic/core/indexer/index/', index_name=Indexes.GENRES)
