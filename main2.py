import numpy as np
import random
from load_data import load_data
import csv
import time  # Importando o módulo de tempo

# Usuários reservados para o teste
test_users = [40, 92, 123, 245, 312, 460, 514, 590]

# FUNÇÕES PARA NORMALIZAR OS DADOS
# -------------------------------------------------------------------------
# Função para normalizar manualmente (mapear valores de 0 a 5 para 0 a 1)
def manual_normalize(value):
    return value / 5 if value != 0 else 0

# Função para desnormalizar manualmente (mapear valores de 0 a 1 de volta para 0 a 5)
def manual_denormalize(value):
    return round(round(value * 5 * 2) / 2, 2) if value != 0 else 0

# Função para aplicar a normalização em uma matriz de usuários e filmes
def normalize_matrix(user_movie_matrix):
    normalized_matrix = user_movie_matrix.copy()
    for index, row in normalized_matrix.iterrows():
        normalized_matrix.loc[index] = [manual_normalize(val) for val in row]
    return normalized_matrix

def fatoracao_matricial(user_movie_matrix, k, alpha, lambda_reg, num_epochs, batch_size=64, mse_threshold=0.1):
    num_users, num_movies = user_movie_matrix.shape
    start_time1 = time.time()  # Início da contagem do tempo
    # Inicialização com valores menores
    W = np.random.uniform(low=-0.01, high=0.01, size=(num_users, k))
    V = np.random.uniform(low=-0.01, high=0.01, size=(num_movies, k))
    
    # Extraindo as entradas (i, j) onde existem avaliações
    S = [(i, j) for i in range(num_users) for j in range(num_movies) if user_movie_matrix.iloc[i, j] > 0]
    mse_list = []
    
    for epoch in range(num_epochs):
        start_time = time.time()  # Início da contagem do tempo
        random.shuffle(S)

        # Dividir as entradas em mini-batches
        for i in range(0, len(S), batch_size):
            batch = S[i:i + batch_size]
            grad_W = np.zeros_like(W)
            grad_V = np.zeros_like(V)

            for (user_id, movie_id) in batch:
                y_ij = user_movie_matrix.iloc[user_id, movie_id]
                pred_ij = np.dot(W[user_id, :], V[movie_id, :])
                e_ij = y_ij - pred_ij
                
                # Atualizar gradientes
                grad_W[user_id] += -2 * e_ij * V[movie_id] + 2 * lambda_reg * W[user_id]
                grad_V[movie_id] += -2 * e_ij * W[user_id] + 2 * lambda_reg * V[movie_id]
            
            # Atualizar W e V
            W -= alpha * grad_W
            V -= alpha * grad_V

            # Clipping
            np.clip(W, -1, 1, out=W)
            np.clip(V, -1, 1, out=V)

        # Calcular MSE para esta época
        mse = np.mean([(user_movie_matrix.iloc[i, j] - np.dot(W[i, :], V[j, :])) ** 2 for (i, j) in S])
        mse_list.append(mse)

        end_time = time.time()  # Fim da contagem do tempo
        elapsed_time = end_time - start_time  # Tempo decorrido

        print(f'Época {epoch + 1}/{num_epochs}, Erro quadrático médio: {mse:.4f}, Tempo: {elapsed_time:.4f} segundos')
        
        if mse < mse_threshold:
            print(f'Treinamento encerrado na época {epoch + 1}, MSE atingiu {mse:.4f}')
            break
    end_time1 = time.time()  # Fim da contagem do tempo
    elapsed_time1 = end_time1 - start_time1  # Tempo decorrido
    print( f'Tempo TOTAL: {elapsed_time1:.4f} segundos')
    return W, V, mse_list

def recomendar_filmes(user_movie_matrix, W, V, user_ids, num_filmes=5):
    """
    Gera recomendações de filmes para os usuários especificados.
    
    Parâmetros:
    - user_movie_matrix: Matriz de avaliações de usuários-filmes.
    - W: Matriz de fatores latentes para usuários.
    - V: Matriz de fatores latentes para filmes.
    - user_ids: Lista de IDs de usuários para gerar recomendações.
    - num_filmes: Número de recomendações por usuário.
    
    Retorno:
    - Tabela com recomendações de filmes e MSE para os usuários.
    """
    
    num_users, num_movies = user_movie_matrix.shape
    recommendations = []
    
    for user_id in user_ids:
        if user_id < num_users:  # Verificação do limite do índice do usuário
            # Filmes que o usuário já avaliou
            user_ratings = user_movie_matrix.iloc[user_id, :]
            avaliados = user_ratings[user_ratings > 0].index.tolist()
            
            # Predições para todos os filmes
            predicoes = np.dot(W[user_id, :], V.T)
            
            # Remover os filmes que o usuário já avaliou
            predicoes_filtradas = [(i, predicoes[i]) for i in range(num_movies) if i not in avaliados]
            predicoes_filtradas.sort(key=lambda x: x[1], reverse=True)
            
            # Selecionar os top N filmes
            top_filmes = [i for i, _ in predicoes_filtradas[:num_filmes]]
            
            # Cálculo do MSE para os filmes que o usuário já avaliou
            mse_user = np.mean([(user_ratings[i] - predicoes[i]) ** 2 for i in avaliados if i < num_movies])
            
            recommendations.append((user_id, top_filmes, mse_user))
    
    return recommendations

# Exemplo de uso:
alpha = 0.005
lambda_reg = 0.0005
k_values = [50, 75, 100]
num_epochs = 5

# Carregue sua matriz de avaliações `user_movie_matrix`
user_movie_matrix, movies = load_data()

# Normalizando a matriz de usuários e filmes sem alterar os zeros
user_movie_matrix_normalized = normalize_matrix(user_movie_matrix)

# Separando os usuários para treinamento/validação e teste
train_users = user_movie_matrix.index.difference(test_users)
train_matrix = user_movie_matrix_normalized.loc[train_users]

for k in k_values:
    print(f"Treinando com k={k}")
    W, V, mse_list = fatoracao_matricial(train_matrix, k, alpha, lambda_reg, num_epochs)
    
    # IDs dos usuários para gerar recomendações
    user_ids = test_users
    
    # Gerar recomendações e calcular o MSE para os usuários especificados
    recommendations = recomendar_filmes(user_movie_matrix_normalized, W, V, user_ids)
    
    # Exibir recomendações e MSE
    with open('recomendacoes.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['User ID', 'Filmes Recomendados', 'MSE'])
        for user_id, top_filmes, mse_user in recommendations:
            writer.writerow([user_id, ','.join(map(str, top_filmes)), mse_user])
