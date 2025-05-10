import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import joblib
import os
from datetime import datetime

# 1. Configuração Inicial da GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configuração de memória dinâmica
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Callback Personalizado para Monitoramento
class GPUMonitor(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if gpus:
            # Método alternativo para monitorar uso de GPU
            mem_info = tf.config.experimental.get_memory_usage('GPU:0')
            print(f"\nUso de GPU: {mem_info/1e9:.2f}GB")

# 3. Geração de Dados Corrigida e Otimizada
def generate_training_data(numbers, decimal_prob=0.3, max_int=100, max_decimal=100.0):
    X = np.zeros((numbers, 6))
    y = np.zeros(numbers)
    valid_count = 0
    max_value = max(max_int, max_decimal)
    
    while valid_count < numbers:
        # Geração de números
        use_decimal1 = np.random.random() < decimal_prob
        use_decimal2 = np.random.random() < decimal_prob
        
        number1 = round(np.random.uniform(0.1, max_decimal), 2) if use_decimal1 else np.random.randint(1, max_int)
        number2 = round(np.random.uniform(0.1, max_decimal), 2) if use_decimal2 else np.random.randint(1, max_int)
        
        operation = np.random.randint(0, 4)
        try:
            if operation == 0:
                result = number1 + number2
                norm_factor = max_value
            elif operation == 1:
                result = number1 - number2
                norm_factor = max_value
            elif operation == 2:
                result = number1 * number2
                norm_factor = max_value * max_value
            elif operation == 3:
                denominator = number2 if abs(number2) >= 1e-10 else 1.0
                result = number1 / denominator
                norm_factor = max_value
        except Exception:
            continue

        # Normalização
        X[valid_count, 0] = number1 / max_value
        X[valid_count, 1] = number2 / max_value
        X[valid_count, 2 + operation] = 1
        y[valid_count] = result / norm_factor
        valid_count += 1

    return X[:valid_count], y[:valid_count]

# 4. Construção do Modelo Atualizada
def build_model(hp):
    activation = hp.Choice('activation', ['relu', 'tanh', 'selu'])
    num_layers = hp.Int('num_layers', 1, 3)
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', 0.0, 0.3, step=0.1)
    l2_reg = hp.Float('l2_reg', 1e-6, 1e-2, sampling='log')
    
    inputs = keras.layers.Input(shape=(6,))
    x = inputs
    
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', 64, 256, step=64)
        x = keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=losses.Huber(),
        metrics=['mae', 'mse']  # Removido o parâmetro obsoleto
    )
    
    return model

# 5. Configuração de Callbacks
def setup_callbacks(validation_data):
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return [
        callbacks.EarlyStopping(
            monitor='val_mae',
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        GPUMonitor()
    ]

# 6. Pipeline de Dados para GPU
def create_dataset(X, y, batch_size=512, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 7. Função Principal
def main():
    # Configurações de performance
    tf.config.optimizer.set_jit(True)
    
    # 1. Gerar dados
    print("Gerando dados...")
    X, y = generate_training_data(300000)
    
    # 2. Dividir e normalizar
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 3. Criar datasets otimizados
    train_dataset = create_dataset(X_train, y_train, shuffle=True)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)
    
    # 4. Otimização de hiperparâmetros
    print("Otimizando hiperparâmetros...")
    tuner = kt.Hyperband(
        build_model,
        objective='val_mae',
        max_epochs=50,
        factor=3,
        directory='tuning',
        project_name='math_ops_gpu',
        overwrite=True
    )
    
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=setup_callbacks((X_val, y_val)),
        verbose=1
    )
    
    # 5. Treinar modelo final
    print("\nTreinando modelo final...")
    best_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    history = best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=setup_callbacks((X_val, y_val)),
        verbose=1
    )
    
    # 6. Avaliação
    test_loss, test_mae, test_mse = best_model.evaluate(test_dataset, verbose=0)
    print("\nPerformance no teste:")
    print(f"- Loss: {test_loss:.4f}")
    print(f"- MAE: {test_mae:.4f}")
    print(f"- MSE: {test_mse:.4f}")
    
    # 7. Salvar modelo
    best_model.save('modelo_final_gpu.keras')
    joblib.dump(scaler, 'scaler_gpu.pkl')
    print("\nModelo e scaler salvos.")

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    main()