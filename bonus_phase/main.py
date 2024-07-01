import streamlit as st
import sys
sys.path.append(r"D:\University\MIR_project_Spring2024-Phase-1.2")
import os
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.snippet import Snippet
from Logic.core.preprocess import Preprocessor

snippet_obj = Snippet(
    number_of_words_on_each_side=5
)

class color(Enum):
    RED = "#00BFFF"   # Light blue
    GREEN = "#00CED1" # Cyan
    BLUE = "#1E90FF"  # Light blue
    YELLOW = "#00FFFF" # Cyan
    PURPLE = "#ADD8E6"  # Light blue
    ORANGE = "#87CEEB"  # Light blue
    CYAN = "#F0FFFF"  # Light cyan
    MAGENTA = "#E0FFFF"  # Light cyan

def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary

def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))

def toggle_star_state(movie_id):
    # session_state = st.session_state
    # if 'star_states' not in session_state:
    #     session_state.star_states = {}
    
    # if movie_id in session_state.star_states:
    #     session_state.star_states[movie_id] = not session_state.star_states[movie_id]
    # else:
    #     session_state.star_states[movie_id] = True
    pass

def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
):
    if search_button:
        spell_correction_dataset = [summary for movie in utils.movies_dataset for summary in movie["summaries"]]
        
        spell_correction_dataset = Preprocessor(spell_correction_dataset).preprocess()
        corrected_query = utils.correct_text(search_term, spell_correction_dataset)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
            )
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                movie_id = result[i][0]
                relevance_score = result[i][1]
                 
                with card[0].container():
                    title_string = f"[{info['title']}]({info['URL']}) - {info['score']}"
                    st.title(title_string)
                    
                    if relevance_score > 0:
                        st.markdown("<span style='font-size: 20px; color: yellow;'>⭐</span>", unsafe_allow_html=True)

                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    with st.expander("Details"):
                        st.write(f"Relevance Score: {result[i][1]}")
                        st.markdown("**Directors:**")
                        if info["directors"] is not None:
                            for director in info["directors"]:
                                st.text(director)

                        st.markdown("**Stars:**")
                        stars = ", ".join(info["stars"])
                        st.text(stars)

                        st.markdown("**Genres:**")
                        genre_colors = iter([color.RED.value, color.GREEN.value, color.BLUE.value, color.YELLOW.value, color.PURPLE.value, color.ORANGE.value, color.CYAN.value, color.MAGENTA.value])
                        for genre in info["genres"]:
                            genre_color = next(genre_colors)
                            st.markdown(
                                f"<span style='color:{genre_color}'>{genre}</span>",
                                unsafe_allow_html=True,
                            )
                with card[1].container():
                    st.image(info["Image_URL"], use_column_width=True)

                st.divider()

def main():
    st.title("IMDB Movie Search Engine")
    st.write(
        "Search through IMDB dataset and find the most relevant movies to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("Search Term", help="Enter the term you want to search for.")
    
    with st.sidebar:
        st.header("Advanced Search")
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5, help="Set the maximum number of search results."
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Adjust the weight given to stars in the search results."
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Adjust the weight given to genres in the search results."
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Adjust the weight given to the summary in the search results."
        )

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method",
            ("ltn.lnn", "ltc.lnc", "OkapiBM25"),
            help="Choose the search method."
        )

    search_button = st.button("Search", key="search_button")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
    )

    # Custom CSS for search button
    custom_css = """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
    }

    .stButton button:hover {
        background-color: #3e8e41;
        color: white;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
