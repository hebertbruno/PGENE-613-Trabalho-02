from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Função de perda personalizada que ignora os valores 0
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    return K.mean(mask * K.square(y_pred - y_true), axis=-1)

# Callback para encerrar o treinamento quando MSE for menor que 0.1
class TerminateOnMSE(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mse = logs.get('val_loss')
        if mse is not None and mse < 0.1:
            print(f'\nMSE abaixo de 0.1, encerrando treinamento na época {epoch + 1}.')
            self.model.stop_training = True

# Função para treinar o autoencoder com Early Stopping e Checkpoint
def train_autoencoder(autoencoder, user_movie_matrix, epochs=500, batch_size=128):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('model/best_autoencoder.keras', monitor='val_loss', save_best_only=True)
    terminate_on_mse = TerminateOnMSE()

    # Compilar o autoencoder com a função de perda personalizada
    autoencoder.compile(optimizer='adam', loss=masked_mse)

    # Treinamento com validação (20% para validação)
    history = autoencoder.fit(
        user_movie_matrix, user_movie_matrix,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, terminate_on_mse]
    )

    autoencoder.load_weights('model/best_autoencoder.keras')

    return autoencoder
