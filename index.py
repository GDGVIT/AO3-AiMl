import ast
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI,Cookie,Path
from typing import Annotated
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

def extract_bookmark_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    bookmarks = []

    for bookmark in soup.select('li.bookmark.blurb.group'):
        # Extract the author's name
        author = bookmark.select_one('div.header.module h4 a[rel="author"]').text if bookmark.select_one(
            'div.header.module h4 a[rel="author"]') else 'Unknown'

        # Extract tags and user-set tags
        tags = [tag.text for tag in bookmark.select('.tags.commas li a.tag')]
        user_set_tags = [tag.text for tag in bookmark.select('.meta.tags.commas li a.tag')]
        all_tags = tags + user_set_tags

        # Extract genres
        genres = [genre.text for genre in bookmark.select('h5.fandoms.heading a.tag')]

        bookmarks.append({
            "author_name": author,
            "tags": all_tags,
            "genres": genres
        })

    return bookmarks

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return x if isinstance(x, list) else []

def safe_transform(vectorizer, text):
    try:
        return vectorizer.fit_transform(text)
    except ValueError as e:
        print(f"Error transforming text: {e}")
        print("Problematic text:", text)
        return None

def get_top_stories(tag):
    tag_query = tag.replace(" ", "%20")
    search_url = f"https://archiveofourown.org/works/search?work_search%5Bquery%5D=&work_search%5Btitle%5D=&work_search%5Bcreators%5D=&work_search%5Brevised_at%5D=&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bword_count%5D=&work_search%5Blanguage_id%5D=&work_search%5Bfandom_names%5D={tag_query}&work_search%5Brating_ids%5D=&work_search%5Bcharacter_names%5D=&work_search%5Brelationship_names%5D=&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=&work_search%5Bcomments_count%5D=&work_search%5Bbookmarks_count%5D=&work_search%5Bsort_column%5D=_score&work_search%5Bsort_direction%5D=desc&commit=Search"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the top 5 stories' URLs
    story_urls = []
    for work in soup.select('li.work.blurb.group')[:5]:
        story_url = "https://archiveofourown.org" + work.select_one('h4.heading a')['href']
        story_urls.append(story_url)
    
    return story_urls

@app.route('/recommendations/{username}')
async def similarity(username):
    userid = username.path_params["username"]
    url = f"https://archiveofourown.org/users/{userid}/bookmarks"
    # print(username.path_params["username"])
    response = requests.get(url)
    data = response.content
    # print(data)

    # Get extracted bookmark data
    bookmark_data = extract_bookmark_data(data)
    # print(bookmark_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(bookmark_data)

    # Print the DataFrame
    # print(df)

    df['author_combined'] = df['author_name']
    df['tags_combined'] = df['tags'].apply(lambda x: ' '.join(safe_eval(x)))
    df['genres_combined'] = df['genres'].apply(lambda x: ' '.join(safe_eval(x)))

    vectorizer_author = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b')
    vectorizer_tags = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b')
    vectorizer_genres = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b')

    tfidf_author_matrix = safe_transform(vectorizer_author, df['author_combined'])
    tfidf_tags_matrix = safe_transform(vectorizer_tags, df['tags_combined'])
    tfidf_genres_matrix = safe_transform(vectorizer_genres, df['genres_combined'])

    if tfidf_author_matrix is None or tfidf_tags_matrix is None or tfidf_genres_matrix is None:
        return{"msg":"One or more transformations failed. Cannot proceed."}
    else:
        tfidf_author_array = tfidf_author_matrix.toarray()
        tfidf_tags_array = tfidf_tags_matrix.toarray()
        tfidf_genres_array = tfidf_genres_matrix.toarray()

        combined_vectors = np.hstack([tfidf_author_array, tfidf_tags_array, tfidf_genres_array])

        dimension_author = tfidf_author_array.shape[1]
        dimension_tags = tfidf_tags_array.shape[1]
        dimension_genres = tfidf_genres_array.shape[1]

        index_author = faiss.IndexFlatL2(dimension_author)
        index_tags = faiss.IndexFlatL2(dimension_tags)
        index_genres = faiss.IndexFlatL2(dimension_genres)

        index_author.add(tfidf_author_array.astype(np.float32))
        index_tags.add(tfidf_tags_array.astype(np.float32))
        index_genres.add(tfidf_genres_array.astype(np.float32))

        query_tags_vector = tfidf_tags_array[0].reshape(1, -1).astype(np.float32)

        k = 5

        distances_tags, tags_indices = index_tags.search(query_tags_vector, k)

        # print("\nDistances for tags:")
        # print(distances_tags.flatten())

        inverse_tags_vectors = vectorizer_tags.inverse_transform(tfidf_tags_array[tags_indices.flatten()])

        similar_tags = [', '.join(tags) for tags in inverse_tags_vectors]

        # Get the top 5 tags based on the distances
        top_5_tags = []
        for tags in inverse_tags_vectors:
            top_5_tags.extend(tags)
        top_5_tags = list(set(top_5_tags))[:5]  # Get unique tags and limit to top 5

        # print("\nTop 5 tags:")
        # print(top_5_tags)

    # Create a dictionary to store the top stories for each top tag
    top_stories = {tag: get_top_stories(tag) for tag in top_5_tags}
    # print(top_stories)
    json_compatible_item_data = jsonable_encoder(top_stories)
    return JSONResponse(content=json_compatible_item_data)
    return top_stories

    # Print the top stories for each tag
    # for tag, stories in top_stories.items():
    #     print(f"Top stories for tag '{tag}':")
    #     for story in stories:
    #         print(story)
    #     print("\n")

