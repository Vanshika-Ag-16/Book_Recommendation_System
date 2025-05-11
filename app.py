import pandas as pd
import numpy as np

book = pd.read_csv(r'C:\Users\lenovo\Desktop\book recom\Books\Books.csv')
rating = pd.read_csv(r'C:\Users\lenovo\Desktop\book recom\Ratings.csv\Ratings.csv')
user = pd.read_csv(r'C:\Users\lenovo\Desktop\book recom\Users.csv (1)\users.csv')

print(book.tail())
print(user.tail())
print(user.tail())

print(book.shape)
print(user.shape)
print(rating.shape)

book.isnull().sum()
user.isnull().sum()
rating.isnull().sum()

book.duplicated().sum()
user.duplicated().sum()
rating.duplicated().sum()

user.info()
book.info()
rating.info()

rating_with_name = rating.merge(book,on='ISBN')

num_rating_df = rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'Num_rating'},inplace=True)
print(num_rating_df)

avg_rating_df = rating_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'Avg_rating'},inplace=True)
print(avg_rating_df)


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
print(popular_df)

pbr_df=popular_df[popular_df['Num_rating']>=300].sort_values('Avg_rating',ascending=False).head(100)
pbr_df= pbr_df.merge(book,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Publisher','Image-URL-M','Num_rating','Avg_rating']]
print(pbr_df)

b = rating_with_name.groupby('User-ID').count()['Book-Rating']>250
users_with_ratings = b[b].index

filtered_rating = rating_with_name[rating_with_name['User-ID'].isin(users_with_ratings)]

c = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = c[c].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)
print(pt)

from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(pt)
print(similarity_scores.shape)

from fuzzywuzzy import process

def recommendation(search_query, dataset):
    """
    Recommends books with titles similar to the search query.

    Args:
        search_query (str): The book name to search for.
        dataset (pd.DataFrame): The dataset containing book details.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        list: A list of recommended books with their details.
    """
    # Fetch all book titles from the dataset
    book_titles = dataset['Book-Title'].unique()
    
    # Fuzzy matching to find the closest matching titles
    similar_books = process.extract(search_query, book_titles)
    
    # Preparing recommendations
    results = []
    for book_title, score in similar_books:
        temp_df = dataset[dataset['Book-Title'] == book_title]
        results.append(temp_df)
    
    # Combine results and drop duplicates
    recommendations_df = pd.concat(results).drop_duplicates(subset='Book-Title')
    
    # Convert to list format
    recommendations = recommendations_df[['Book-Title', 'Book-Author', 'Image-URL-M']].values.tolist()
    return recommendations



def recommendation2(book_author):
    # Check if the author exists in the dataset
    if book_author not in book['Book-Author'].values:
        return f"'{book_author}' not found in the dataset. Please try another author."
    
    # Filter books by the same author
    temp_df = book[book['Book-Author'] == book_author]
    
    # Prepare the recommendations
    data = []
    for _, row in temp_df.iterrows():
        item = [
            row['Book-Title'],  # Book title
            row['Book-Author'],  # Author
            row['Image-URL-M']   # Book cover image
        ]
        data.append(item)
    
    return data




import streamlit as st
from IPython.display import Image, display

# Title
st.title("Book Recommendation System")

# Sidebar for User ID input
book_name = st.sidebar.text_input("Enter Book name", "")

# Recommend books button
if st.sidebar.button("Get Recommendations"):
    recommendations = recommendation(book_name, book)
    st.write("Recommended Books:")
    book_df = pd.DataFrame(recommendations)
    # Display book covers with titles
    for index, row in book_df.iterrows():
        st.write(f"Title: {row[0]}, Author: {row[1]}")
        st.image(row[2], caption=row[0], width=200)
        
book_author = st.sidebar.text_input("Enter Author name", "")

# Recommend books button
if st.sidebar.button("Show Recommendations"):
    recommendations = recommendation2(book_author)
    st.write("Recommended Books:")
    book_df = pd.DataFrame(recommendations)
    # Display book covers with titles
    for index, row in book_df.iterrows():
        st.write(f"Title: {row[0]}, Author: {row[1]}")
        st.image(row[2], caption=row[0], width=200)
