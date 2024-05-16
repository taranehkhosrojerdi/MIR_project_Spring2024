from collections import defaultdict
import json
from .preprocess import Preprocessor

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        word = "$" + word + "$"
        shingles = set()
        
        # TODO: Create shingle here
        for i in range(len(word) - k + 1):
            shingle = word[i:i+k]
            shingles.add(shingle)
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.

        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        jaccard = intersection / union if union > 0 else 0
        return jaccard

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = defaultdict(set)
        word_counter = defaultdict(int)

        # TODO: Create shingled words dictionary and word counter dictionary here.
        for document in all_documents:
            words = document.split()
            for word in words:
                shingles = self.shingle_word(word)
                all_shingled_words[word].update(shingles)
                word_counter[word] += 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        shingles = self.shingle_word(word)
        similarity_scores = {}

        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            similarity_score = self.jaccard_score(shingles, candidate_shingles)
            similarity_scores[candidate_word] = similarity_score

        sorted_candidates = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top5_candidates = [candidate[0] for candidate in sorted_candidates[:5]]

        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        
        # TODO: Do spell correction here.
        
        corrected_query = ""
        words = query.split()

        for word in words:
            nearest_words = self.find_nearest_words(word)
            combined_scores = []
            max_tf_score = max([self.word_counter[cw] for cw in nearest_words])
            for candidate_word in nearest_words:
                shingles = self.shingle_word(candidate_word)
                jaccard_score = self.jaccard_score(shingles, self.shingle_word(word))
                tf_score = self.word_counter[candidate_word] / max_tf_score
                combined_score = jaccard_score * tf_score
                combined_scores.append((candidate_word, combined_score))
            corrected_query = max(combined_scores, key=lambda x: x[1])[0]

        return corrected_query
    
# --------------------------------------------Test----------------------------------------------
# with open('../tests/IMDB_crawled.json') as f:
#     data = json.load(f)
# spell_correction_dataset = [summary for movie in data for summary in movie["summaries"]]
# # spell_correction_dataset.extend(movie["title"] for movie in data if movie["title"] != None)
# # spell_correction_dataset = [star_name for movie in data for star in movie["stars"] for star_name in star.split() if movie["stars"] != None]
# spell_correction_dataset = Preprocessor(spell_correction_dataset).preprocess()

# sample_docs = [item['summaries'][0] for item in data]
# spell_checker = SpellCorrection(spell_correction_dataset)
# print(spell_checker.find_nearest_words("batman"))
# print(spell_checker.spell_check("batman"))
