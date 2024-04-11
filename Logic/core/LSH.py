import numpy as np
import itertools
import random
from collections import defaultdict
import json # for testing

class MinHashLSH:
    def __init__(self, documents: list, num_hashes: int):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        # TODO
        shingles = []
        for i in range(len(document) - k + 1):
            shingles.append(document[i:i+k])
        return set(shingles)

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix
        """
        # TODO
        all_shingles = self.shingle_document(self.documents[0])
        for i, doc in enumerate(self.documents[1:]):
            all_shingles = all_shingles.union(self.shingle_document(doc))

        self.all_shingles = all_shingles

        characteristic_matrix = np.zeros((len(self.documents), len(all_shingles)), dtype=int)
        for i, doc in enumerate(self.documents):
            characteristic_matrix[i, :] = [1 if shingle in doc else 0 for shingle in all_shingles]
        
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        # TODO
        char_matrix = self.build_characteristic_matrix()
        num_docs = len(char_matrix)
        num_shingles = len(char_matrix[0])
        minhash_signature_matrix = np.full((self.num_hashes, num_docs), np.inf)
        hash_base = list(range(1, num_shingles + 1))
        random.shuffle(hash_base)
        hash_functions = [np.random.permutation(num_shingles) for _ in range(self.num_hashes)]

        for di in range(num_docs):
            for hi, hash in enumerate(hash_functions):
                one_indices = np.where(char_matrix[di, hash] == 1)[0]
                if len(one_indices) > 0:
                    minhash_signature_matrix[hi, di] = min(one_indices)

        return minhash_signature_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        num_docs = signature.shape[1]
        bucket_dict = defaultdict(list)
        band_size = rows_per_band * bands

        for i in range(0, self.num_hashes, rows_per_band):
            for j in range(num_docs):
                band_signature = tuple(signature[i:i+rows_per_band, j])
                band_hash = hash(band_signature)
                bucket_id = band_hash % band_size
                bucket_dict[bucket_id].append(j)

        return bucket_dict

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        minhash_signature = self.min_hash_signature()
        buckets = self.lsh_buckets(minhash_signature)
        return buckets

    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        # TODO
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        jaccard = intersection / union if union > 0 else 0
        return jaccard

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


# -------------------------------------------Test with dump data------------------------------------
with open('./Logic/core/LSHFakeData.json') as f:
    data = json.load(f)
documents = [item['summaries'][0] for item in data]
lsh = MinHashLSH(documents, num_hashes=10)
buckets = lsh.perform_lsh()
lsh.jaccard_similarity_test(buckets, documents)
