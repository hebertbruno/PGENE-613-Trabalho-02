import pandas as pd

# Função para desnormalizar manualmente (mapear valores de 0 a 1 de volta para 0 a 5)
def manual_denormalize(value):
    if value == 0:
        return 0
    else:
        return round(round(value * 5,2)/ 2,2)  # Mapeia 0 a 1 de volta para 0 a 5

def recommend_movies(user, autoencoder, user_movie_matrix, movies):
    user_ratings = user_movie_matrix.loc[user].values.reshape(1, -1)
    predicted_ratings_normalized = autoencoder.predict(user_ratings).flatten()
    
    # Desnormalizar as previsões
    predicted_ratings = [manual_denormalize(rating) for rating in predicted_ratings_normalized]

    user_rated_movies = user_movie_matrix.loc[user] > 0
    unrated_movies = user_rated_movies[user_rated_movies == False].index
    
    movies_df = pd.DataFrame({
        'movieId': user_movie_matrix.columns,
        'predicted_rating': predicted_ratings
    })
    
    recommendations_df = movies_df[movies_df['movieId'].isin(unrated_movies)]
    recommendations_df = recommendations_df.sort_values(by='predicted_rating', ascending=False)
    recommendations = recommendations_df.merge(movies, on='movieId')
    
    return recommendations
