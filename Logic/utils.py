from typing import Dict, List

import sys
sys.path.append(r"C:\Users\Asus\PycharmProjects\MIR_project_Spring2024")

from Logic.core.search import SearchEngine
from Logic.core.utility.spell_correction import SpellCorrection
from Logic.core.utility.preprocess import Preprocessor
from Logic.core.utility.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
import json

movies_dataset = json.load(open("Logic/tests/IMDB_crawled.json", "r"))
search_engine = SearchEngine()


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!
    text = Preprocessor([text]).preprocess()[0]
    # TODO: uncomment for spell correction
    # spell_correction_obj = SpellCorrection(all_documents)
    # new_text = ""
    # for word in text.split():
    #     new_text += spell_correction_obj.spell_check(word) + " "
    # text = new_text
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
    unigram_smoothing: str = 'naive',
    alpha = None,
    lamda = None
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    dict_weights = {'stars': weights[0], 'genres': weights[1], 'summaries': weights[2]}  # TODO
    output = search_engine.search(
        query, method, dict_weights, max_results=max_result_count, safe_ranking=True, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda
    )
    print(len(output))
    return output
    # weights = ...  # TODO
    # return search_engine.search(
    #     query, method, weights, max_results=max_result_count, safe_ranking=True
    # )
    return None


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    # TODO: self-added code
    result = {}
    print("looook:", id)
    for movie in movies_dataset:
        if movie["id"] == id:
            result = movie
            break
    
    # result = movies_dataset.get(
    #     id,
    #     {
    #         "Title": "This is movie's title",
    #         "Summary": "This is a summary",
    #         "URL": "https://www.imdb.com/title/tt0111161/",
    #         "Cast": ["Morgan Freeman", "Tim Robbins"],
    #         "Genres": ["Drama", "Crime"],
    #         "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
    #     },
    # )

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result
