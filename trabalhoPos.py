from tensorflow import keras
import numpy as np

# Criando os dados de treino com subtração correta
X_train = np.random.randint(-100, 100, size=(10000, 2)).astype(float)
y_train = np.array([a - b for a, b in X_train], dtype=float)


# Criando o modelo
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(2, )),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinando o modelo
model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1)

# Testando a rede
test_data = np.array([[7, 3], [4, 1], [8, 5]], dtype=float)
predictions = model.predict(test_data) 

# Exibindo os resultados
for i in range(len(test_data)):
    print(f"Subtração de {test_data[i][0]} e {test_data[i][1]} é aproximadamente {predictions[i][0]:.2f}")


###Multiplicação

# from tensorflow import keras
# import numpy as np

# X_train = np.random.randint(1, 100, size=(10000, 2)).astype(float)
# y_train = np.array([a*b for a, b in X_train], dtype=float)

# X_train = X_train / 100
# y_train = y_train / (100 * 100)


# model = keras.models.Sequential([
#     keras.layers.Dense(128, activation='relu', input_shape=(2, )),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(1, activation='linear')
# ])

# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

# test_data = np.array([[7, 3], [4, 1], [8, 5]], dtype=float) / 100
# predictions = model.predict(test_data) *(100 * 100)

# for i in range(len(test_data)):
#     print(f"Multiplicação de {int(test_data[i][0]*100)} e {int(test_data[i][1]*100)} é aproximadamente {predictions[i][0]:.2f}")
