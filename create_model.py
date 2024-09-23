from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_autoencoder(n_movies, k, learning_rate, lambda_reg, dropout_rate):
    initializer = RandomUniform(minval=-1, maxval=1)#pesos inicializados entre -1 e +1

    input_layer = Input(shape=(n_movies,))
    encoded = Dense(k, activation='relu', kernel_regularizer=l2(lambda_reg), kernel_initializer=initializer)(input_layer)
    #encoded = Dropout(dropout_rate)(encoded)
    decoded = Dense(n_movies, activation='sigmoid', kernel_initializer=initializer)(encoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return autoencoder
