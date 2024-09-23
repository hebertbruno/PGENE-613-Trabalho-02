import pandas as pd
import numpy as np

def load_data():
    """
    Carrega a matriz de classificações dos usuários e os dados dos filmes.
    """
    ratings_file = 'data/ratings.csv'
    movies_file = 'data/movies.csv'

    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    
    # Converte os ratings para float
    ratings['rating'] = ratings['rating'].astype(float)

    # Cria a matriz de classificações
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Converte os valores para string e troca . por ,
    #user_movie_matrix = user_movie_matrix.applymap(lambda x: str(x).replace('.', ',') if x != 0 else '0')

    # Salvar a matriz de classificações em um arquivo CSV
    #user_movie_matrix.to_csv('results/user_movie_matrix.csv', sep=';', index=True, encoding='utf-8-sig')
    #print('Matriz de classificações salva em results/user_movie_matrix.csv')

    return user_movie_matrix, movies
