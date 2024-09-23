import numpy as np

def compute_mse(user, autoencoder, user_movie_matrix):
    if user not in user_movie_matrix.index:
        raise ValueError(f'Usuário {user} não encontrado na matriz de classificações.')
    
    user_data = user_movie_matrix.loc[[user]].values
    predicted_ratings = autoencoder.predict(user_data).flatten()
    
    non_zero_indices = user_data.flatten() > 0
    true_ratings = user_data.flatten()[non_zero_indices]
    predicted_ratings_filtered = predicted_ratings[non_zero_indices]
    
    mse = np.mean((true_ratings - predicted_ratings_filtered) ** 2)
    
    return mse
