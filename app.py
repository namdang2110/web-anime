import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import string

@st.cache(allow_output_mutation=True)
def load_data():

    anime = pd.read_csv("D:\\Do an thuc tap chuyen nganh\\web anime\\data\\anime2023.csv")
    anime = anime[['Name', 'Type', 'Score', 'Studio', 'Episodes', 'Genres', 'Theme', 'Demographic']]
    
    anime["clean_Name"] = anime["Name"].apply(clean)
    anime["clean_Genres"] = anime["Genres"].apply(clean)
    anime["clean_Theme"] = anime["Theme"].apply(clean)
    anime["clean_Studio"] = anime["Studio"].apply(clean)
    anime["clean_Demographic"] = anime["Demographic"].apply(clean)
    
    anime["clean_Genres_Theme_Demographic"] = anime["clean_Genres"] + " " + anime["clean_Theme"] + " " + anime["clean_Demographic"]
    anime["clean_Feature"] = anime["clean_Name"] + " " + anime["clean_Genres_Theme_Demographic"]
    
    indices = pd.Series(anime.index,index=anime['clean_Feature']).drop_duplicates()
    return anime, indices

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def get_similarity(title, anime, indices):
    new_title = False
    feature = anime["clean_Feature"].tolist()
    
    if not(title in feature):
        new_title = True
        feature.append(title)
        
    tfidf = text.CountVectorizer()
    tfidf_matrix = tfidf.fit_transform(feature)
    
    similarity = cosine_similarity(tfidf_matrix)
    
    del tfidf
    del tfidf_matrix
    
    if new_title:
        del feature
        return similarity[len(similarity) - 1]
    else:
        del feature
        index = pd.Series(indices[title])
        return similarity[index[0]]

def anime_recommendation(name, genres, anime, indices):
    cleaned_name = clean(name)
    cleaned_genres = clean(genres)
    cleaned_nameGenres = cleaned_name + " " + cleaned_genres
    
    similarity = get_similarity(cleaned_nameGenres, anime, indices)
    
    similarity_scores = list(enumerate(similarity))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similarity_scores = filter(lambda x: x[1] > 0, similarity_scores)
    similarity_scores = list(similarity_scores)
    
    movieindices = [i[0] for i in similarity_scores if i[0] < len(anime)]
    
    scores = [i[1] for i in similarity_scores if i[0] < len(anime)]
    
    result = pd.DataFrame([anime.iloc[i] for i in movieindices])
    
    result['score'] = scores
    
    result = result['Name'].values.tolist()
    
    return result[:10]
    # return result

def main():
    
    st.set_page_config(page_title="Gợi ý Anime", page_icon=":tada:")
    
    st.title("Gợi ý Anime theo sở thích người xem")
    
    st.write("[Link Github của tôi](https://github.com/namdang2110/Do-an-thuc-tap-chuyen-nganh)")

    data = load_data()
    anime = data[0]
    indices = data[1]

    name = st.text_input("Nhập tên Anime bạn thích :")
    genres = st.text_input("Nhập thể loại phim bạn thích :")
    
    if st.button("Gợi ý"):
        with st.spinner(text='Đang xử lý ...'):
            recommendations = anime_recommendation(name, genres, anime, indices)
            
            if recommendations:
                # st.success('Hoàn thành')
                st.header("Dưới đây là các bộ Anime gợi ý cho bạn :")
                
                for i, recommendation in enumerate(recommendations):
                    st.info(f"{i + 1}. {recommendation}")
                    anime_info = anime.loc[anime['Name'] == recommendation]
                    st.write(f"Điểm số đánh giá : {anime_info['Score'].values[0]}")
                    st.write(f"Thể loại : {anime_info['Genres'].values[0]}")
                    st.write(f"Hình thức : {anime_info['Type'].values[0]}")
                    st.write(f"Số tập phim : {anime_info['Episodes'].values[0]}")   
                    st.write(f"Hãng phim : {anime_info['Studio'].values[0]}")   
                    st.write(f"Chủ đề : {anime_info['Theme'].values[0]}")
                    st.write(f"Đối tượng khán giả : {anime_info['Demographic'].values[0]}")     
                    st.write("-------")
                
                # for i, recommendation in enumerate(recommendations):
                #     st.write(f"{i + 1}. {recommendation}")
            else:
                st.write("Không có gợi ý nào phù hợp")
    
if __name__ == "__main__":
    main()
