import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(
    page_title="AI Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "books.csv"

@st.cache_data
def load_books(file_path):
    # The supplied CSV contains a small number of malformed rows.
    # The Python parser and on_bad_lines='skip' allow valid rows to load.
    data = pd.read_csv(
        file_path,
        engine="python",
        on_bad_lines="skip",
        skipinitialspace=True
    )

    data.columns = data.columns.str.strip()

    required = [
        "bookID", "title", "authors", "average_rating",
        "language_code", "publisher", "ratings_count"
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in ["title", "authors", "publisher", "language_code"]:
        data[column] = data[column].fillna("").astype(str).str.strip()

    for column in ["average_rating", "ratings_count", "num_pages"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["average_rating"] = data["average_rating"].fillna(0)
    data["ratings_count"] = data["ratings_count"].fillna(0)

    data = data[
        (data["title"] != "") &
        (data["authors"] != "")
    ].copy()

    # Combine descriptive fields to create a content profile for each book.
    data["content"] = (
        data["title"] + " " +
        data["authors"] + " " +
        data["publisher"] + " " +
        data["language_code"]
    ).str.lower()

    data = data.drop_duplicates(subset=["bookID"]).reset_index(drop=True)
    return data


@st.cache_resource
def build_recommender(content_values):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )
    matrix = vectorizer.fit_transform(content_values)
    similarity = cosine_similarity(matrix, dense_output=False)
    return similarity


def recommend_books(book_index, books, similarity_matrix, number_of_recommendations,
                    minimum_rating, minimum_ratings):
    similarity_scores = similarity_matrix[book_index].toarray().flatten()

    ranked_indices = similarity_scores.argsort()[::-1]
    ranked_indices = [
        index for index in ranked_indices
        if index != book_index
        and books.iloc[index]["average_rating"] >= minimum_rating
        and books.iloc[index]["ratings_count"] >= minimum_ratings
    ]

    results = books.iloc[ranked_indices[:number_of_recommendations]].copy()
    results["similarity_score"] = similarity_scores[ranked_indices[:number_of_recommendations]]
    return results


st.title("📚 AI-Based Book Recommendation System")
st.write(
    "Discover books similar to the ones you enjoy. "
    "This application uses TF-IDF text features and cosine similarity "
    "to identify books with related titles, authors, publishers, and languages."
)

try:
    books = load_books(DATA_PATH)
except Exception as error:
    st.error(f"Could not load books.csv: {error}")
    st.info("Place books.csv in the same folder as app.py.")
    st.stop()

if len(books) < 2:
    st.error("The dataset must contain at least two valid books.")
    st.stop()

similarity_matrix = build_recommender(tuple(books["content"].tolist()))

with st.sidebar:
    st.header("Recommendation settings")
    number_of_recommendations = st.slider(
        "Number of recommendations", min_value=3, max_value=15, value=5
    )
    minimum_rating = st.slider(
        "Minimum average rating", min_value=0.0, max_value=5.0,
        value=3.5, step=0.1
    )
    minimum_ratings = st.number_input(
        "Minimum number of user ratings",
        min_value=0, max_value=1000000, value=0, step=100
    )
    st.caption("Filters are applied to recommended books, not the selected book.")

col1, col2, col3 = st.columns(3)
col1.metric("Books available", f"{len(books):,}")
col2.metric("Unique authors", f"{books['authors'].nunique():,}")
col3.metric("Average book rating", f"{books['average_rating'].mean():.2f} / 5")

st.divider()
st.subheader("Find a book")

search_text = st.text_input(
    "Search by book title",
    placeholder="Type a title, for example: Harry Potter"
)

if search_text.strip():
    matches = books[
        books["title"].str.contains(search_text.strip(), case=False, na=False)
    ].head(100)
else:
    matches = books.sort_values("ratings_count", ascending=False).head(100)

if matches.empty:
    st.warning("No matching books were found. Try another title.")
    st.stop()

def format_book(row):
    return f"{row['title']} — {row['authors']}"

selected_label = st.selectbox(
    "Select a book",
    options=matches.index.tolist(),
    format_func=lambda index: format_book(books.loc[index])
)

selected_position = books.index.get_loc(selected_label)
selected_book = books.loc[selected_label]

left, right = st.columns([2, 1])
with left:
    st.markdown(f"### {selected_book['title']}")
    st.write(f"**Author(s):** {selected_book['authors']}")
    st.write(f"**Publisher:** {selected_book['publisher'] or 'Not available'}")
    st.write(f"**Language:** {selected_book['language_code'] or 'Not available'}")
with right:
    st.metric("Average rating", f"{selected_book['average_rating']:.2f} / 5")
    st.metric("Number of ratings", f"{int(selected_book['ratings_count']):,}")

if st.button("Recommend similar books", type="primary", use_container_width=True):
    recommendations = recommend_books(
        selected_position,
        books,
        similarity_matrix,
        number_of_recommendations,
        minimum_rating,
        minimum_ratings
    )

    st.subheader("Recommended books")
    if recommendations.empty:
        st.warning(
            "No books match the current filters. Lower the minimum rating "
            "or minimum number of ratings and try again."
        )
    else:
        for rank, (_, book) in enumerate(recommendations.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {book['title']}**")
                st.write(f"Author(s): {book['authors']}")
                st.write(
                    f"Rating: {book['average_rating']:.2f}/5  |  "
                    f"Ratings: {int(book['ratings_count']):,}  |  "
                    f"Publisher: {book['publisher'] or 'Not available'}"
                )
                st.progress(
                    min(max(float(book["similarity_score"]), 0.0), 1.0),
                    text=f"Content similarity: {book['similarity_score']:.1%}"
                )

        csv_output = recommendations[
            ["title", "authors", "average_rating", "ratings_count",
             "publisher", "similarity_score"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download recommendations as CSV",
            data=csv_output,
            file_name="book_recommendations.csv",
            mime="text/csv"
        )

with st.expander("About this recommendation method"):
    st.write(
        "This is a content-based recommendation approach. The application "
        "combines each book's title, authors, publisher, and language into "
        "one text profile. TF-IDF converts those words into numerical features. "
        "Cosine similarity compares the selected book's feature vector with "
        "other books and ranks the closest matches."
    )
    st.write(
        "The system does not use individual user histories or learn personal "
        "preferences from multiple users. Its recommendations are based on "
        "metadata similarity, so two books by the same author or with similar "
        "metadata may receive higher scores."
    )

st.caption("Dataset: user-provided books.csv. Verify and cite the original dataset source in your project report.")
