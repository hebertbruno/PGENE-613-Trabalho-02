# Tópicos Avançados em Aprendizado de Máquina e Otimização - Trabalho 02

Este projeto é uma aplicação de recomendações de filmes para usuarios de uma plataforma.

## Descrição

O projeto inclui a construção, treinamento e avaliação de um modelo de rede neural autoencoder para prever notas dos filmes para os usuarios e recomenda-los. Utiliza-se o conjunto de dados de avaliações de usuarios para treinar o modelo e avaliar seu desempenho com base na métrica MSE - Mean Squared Error.

## Estrutura do Projeto

O projeto está estruturado da seguinte forma:

- `main.py`: Script principal que carrega os dados, treina o modelo e avalia seu desempenho.
- `load_data.py`: Módulo responsável por carregar e processar os dados.
- `create_model.py`: Define a arquitetura do modelo de autoencoder.
- `train.py`: Funções para treinamento do modelo.
- `recommendation.py`: Função para recomendar os filmes com base nas notas preditas.
- `calculate_mse.py`: Função para calcular o MSE das prediçoes em relação as avaliações reais.

## Observação
- `main2.py`: Script que utiliza fatoração matricial para a recomendação dos filmes.

O codigo se baseia no seguinte algoritmo:

```plaintext
begin
  Inicializar randomicamente as matrizes W e V
  S = {(i,j); (𝑒𝑛𝑡𝑟𝑎𝑑𝑎 𝑖 ≠ 0 , 𝑠𝑎í𝑑𝑎 𝑗 ≠ 0};
  while not convergence do
  begin
    Misturar randomicamente as entradas de S
    for each (𝑖,𝑗) ∈ 𝑆 𝑖𝑛 𝑠ℎ𝑢𝑓𝑓𝑙𝑒𝑑 𝑜𝑟𝑑𝑒𝑟 do
    begin
      𝑒𝑖𝑗 = 𝑦𝑗 − Σ 𝑤𝑞𝑖 𝑣𝑗𝑞, 𝐾𝑞=1
      for each 𝑞 ∈ {1,…,𝑘} do 
        𝑤𝑞𝑖 += 𝑤𝑞𝑖.(1 − 2𝛼𝜆) + 2𝛼𝑒𝑖𝑗.𝑣𝑗𝑞
      for each 𝑞 ∈ {1,…,𝑘} do 
        𝑣𝑗𝑞 += 𝑣𝑗𝑞.(1 − 2𝛼𝜆) + 2𝛼𝑒𝑖𝑗.𝑤𝑞𝑖
      for each 𝑞 ∈ {1,…,𝑘} do 
        𝑢𝑞𝑖 += ... and 𝑣𝑗𝑞 += ...
    end
    check convergence condition
  end
end
RT = V . W

```
Entretanto o desempenho teve um alto custo de tempo, assim mudando a opçao de uso para os modelos otimizados do tensorflow keras

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/hebertbruno/PGENE-613-Trabalho-02.git
    cd PGENE-613-Trabalho-02
    ```

2. Crie um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```
4. Verifique os arquivos de entrada na pasta `/data`:

- `/data/movies.csv`
- `/data/ratings.csv`

## Uso

1. Execute o script principal para iniciar o treinamento e a avaliação do modelo:

    ```bash
    python main.py
    ```

2. Você será solicitado a escolher a quantidade de neuronios na entrada da rede. As opções disponíveis são:

    - `50`
    - `75`
    - `100`

## Arquivos Gerados

Durante a execução do treinamento, o seguinte arquivo será gerado:

- `/model/best_autoencoder.keras`: Melhor modelo salvo após o treinamento encerrar com MSE menor que 0,1 no conjunto de validação.
- `/results`: ao fim do programa, todos os dados, planilhas e graficos serão salvos nesta pasta.

## Dependências

O projeto utiliza as seguintes bibliotecas:

- TensorFlow
- Matplotlib
- NumPy
- Pandas
- scikit-learn

