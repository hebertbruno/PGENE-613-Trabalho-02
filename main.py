from create_model import create_autoencoder
from train import train_autoencoder
from load_data import load_data
from recommendation import recommend_movies
from calculate_mse import compute_mse
import pandas as pd
import matplotlib.pyplot as plt

# Usuários reservados para o teste
test_users = [40, 92, 123, 245, 312, 460, 514, 590]

#FUNCOES PARA NORMALIZAR OS DADOS
# -------------------------------------------------------------------------
# Função para normalizar manualmente (mapear valores de 0 a 5 para 0 a 1)
def manual_normalize(value):
    if value == 0:
        return 0
    else:
        return value / 5  # Mapeia 0 a 5 para 0 a 1

# Função para desnormalizar manualmente (mapear valores de 0 a 1 de volta para 0 a 5)
def manual_denormalize(value):
    if value == 0:
        return 0
    else:
        return round(round(value * 5 * 2) / 2, 2)  # Mapeia 0 a 1 de volta para 0 a 5

# Função para aplicar a normalização em uma matriz de usuários e filmes
def normalize_matrix(user_movie_matrix):
    normalized_matrix = user_movie_matrix.copy()
    for index, row in normalized_matrix.iterrows():
        normalized_matrix.loc[index] = [manual_normalize(val) for val in row]
    return normalized_matrix

# Função para aplicar a desnormalização em uma matriz de usuários e filmes
def save_real_predicted_ratings(user_movie_matrix, user_movie_matrix_normalized, user_ids, best_autoencoder):
    results_list = []

    for user in user_ids:
        # Obter as avaliações reais da matriz original para o usuário
        user_ratings = user_movie_matrix.loc[user]
        real_ratings = user_ratings.values.flatten()

        # Obter as avaliações normalizadas da matriz normalizada
        user_ratings_normalized = user_movie_matrix_normalized.loc[user].values
        normalized_ratings = user_ratings_normalized.flatten()

        # Prever as avaliações usando o autoencoder
        predicted_ratings_normalized = best_autoencoder.predict(user_movie_matrix_normalized.loc[user].values.reshape(1, -1)).flatten()

        # Desnormalizar as avaliações (manter os zeros inalterados)
        desnormalized_ratings = [manual_denormalize(rating) for rating in normalized_ratings]
        predicted_ratings_desnormalized = [manual_denormalize(rating) for rating in predicted_ratings_normalized]

        # Filtrar os valores onde as avaliações reais são diferentes de zero
        non_zero_indices = real_ratings != 0

        # Extrair movieIds associados às notas
        movie_ids = user_ratings.index.values

        # Gerar gráfico para o usuário
        real_ratings_filtered = real_ratings[non_zero_indices]
        predicted_ratings_filtered = [predicted_ratings_desnormalized[i] for i in range(len(real_ratings)) if non_zero_indices[i]]

        plt.figure(figsize=(10, 6))
        plt.scatter(real_ratings_filtered, predicted_ratings_filtered, alpha=0.6)
        plt.plot([min(real_ratings_filtered), max(real_ratings_filtered)], 
                 [min(real_ratings_filtered), max(real_ratings_filtered)], 
                 color='red', linestyle='--')
        plt.title(f'Usuário {user}: Comparação de Avaliações Reais e Preditas Desnormalizadas')
        plt.xlabel('Avaliação Real')
        plt.ylabel('Avaliação Predita Desnormalizada')
        plt.grid(True)
        plt.savefig(f'results/user_{user}_real_vs_predicted.png')
        plt.close()

        print(f'Gráfico salvo para o usuário {user}')

        # Salvar os valores reais, preditos e IDs dos filmes
        for movie_id, real, predicted in zip(movie_ids, real_ratings, predicted_ratings_desnormalized):
            #if real != 0:  # Considerar apenas os filmes que o usuário avaliou
            results_list.append({
                'User': user,
                'Movie ID': movie_id,
                'Real Rating': real,
                'Predicted Rating': round(predicted, 2)
            })

    # Criar um DataFrame e salvar os resultados em CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('results/user_movie_real_predicted_ratings.csv', sep=';', index=False, encoding='utf-8-sig')
    print('CSV com avaliações reais e preditas salvo em results/user_movie_real_predicted_ratings.csv')

# -------------------------------------------------------------------------


def main():
    user_movie_matrix, movies = load_data()
    
    # Normalizando a matriz de usuários e filmes sem alterar os zeros
    user_movie_matrix_normalized = normalize_matrix(user_movie_matrix)
    
    user_movie_matrix_normalized = user_movie_matrix_normalized.round(3)
    #user_movie_matrix_normalized.to_csv('results/user_movie_matrix_normalized.csv', sep=';', index=True, encoding='utf-8-sig')

    # Separando os usuários para treinamento/validação e teste
    train_users = user_movie_matrix.index.difference(test_users)

    train_matrix = user_movie_matrix_normalized.loc[train_users]

    learning_rate = 0.005  # taxa de aprendizagem
    lambda_reg = 0.0005    # taxa de regularização
    dropout_rate = 0

    # Solicitar ao usuário a quantidade de neurônios a ser utilizada
    while True:
        try:
            k = int(input("Escolha a quantidade de neurônios (50, 75 ou 100): "))
            if k in [50, 75, 100]:
                break
            else:
                print("Por favor, escolha um valor válido: 50, 75 ou 100.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número inteiro.")

    # Criar e treinar o autoencoder
    autoencoder = create_autoencoder(user_movie_matrix.shape[1], k, learning_rate, lambda_reg, dropout_rate)
    best_autoencoder = train_autoencoder(autoencoder, train_matrix)

    if best_autoencoder is not None:
        results = pd.DataFrame(index=test_users, columns=['Movie1', 'Rating1', 'Movie2', 'Rating2', 'Movie3', 'Rating3', 'Movie4', 'Rating4', 'Movie5', 'Rating5', 'MSE'])

        for user in test_users:
            user_rated_movies = (user_movie_matrix.loc[user] != 0).sum()
            print(f'Usuário {user} avaliou {user_rated_movies} filmes.')
            
            # Recomendar filmes para o usuário
            recommendations = recommend_movies(user, best_autoencoder, user_movie_matrix_normalized, movies)
            top_movies = recommendations.head(5)
            
            for i, (movie_id, movie, rating) in enumerate(zip(top_movies['movieId'], top_movies['title'], top_movies['predicted_rating'])):
                # Desnormalizando as notas preditas
                desnormalized_rating = manual_denormalize(rating)
                
                # Formatar string no formato "movieId - Título (Ano)"
                formatted_movie = f"{movie_id} - {movie}"
                
                results.at[user, f'Movie{i+1}'] = formatted_movie
                results.at[user, f'Rating{i+1}'] = rating * 2  # Multiplicando por 2 conforme sua lógica

            mse = compute_mse(user, best_autoencoder, user_movie_matrix_normalized)
            results.at[user, 'MSE'] = round(mse, 4)

        # Salvar as recomendações e notas preditas no CSV
        results.to_csv(f'results/user_recommendations_k_{k}_neurons.csv', sep=';', index=True, encoding='utf-8-sig')

        # Salvar as avaliações reais, normalizadas e desnormalizadas
        save_real_predicted_ratings(user_movie_matrix, user_movie_matrix_normalized, test_users, best_autoencoder)

if __name__ == "__main__":
    main()
