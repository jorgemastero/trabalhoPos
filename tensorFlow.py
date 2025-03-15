# import tensorflow as tf
# from tensorflow import keras
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# import os
# TF_ENABLE_ONEDNN_OPTS=0

# fashion_mnist = keras.datasets.fashion_mnist
# (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# class_names = ["T-shit/top", "Trouser", "Pullover", "Dress", "Coat",
#                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics =["accuracy"])

# history = model.fit(X_train, y_train, epochs=30,
#                     validation_data=(X_valid, y_valid))

# # pd.DataFrame(history.history).plot(figsize=(8,5))
# # plt.grid(True)
# # plt.gca().set_ylim(0, 1)
# # plt.show()

# # model.evaluate(X_test, y_test)

# X_new = X_test[:3]
# y_proba = model.predict(X_new)
# y_proba.round(2)

# y_pred = model.predict_class(X_new)
# y_pred

# np.array(class_names)[y_pred]
# y_new = y_test[:3]
# y_new

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Desabilitando oneDNN (opcional)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Carregando o dataset Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Preparando os dados
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Nomes das classes
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Construindo o modelo
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Compilando o modelo
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# Selecionando algumas imagens do conjunto de teste
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_pred = np.argmax(y_proba, axis=1)  # Obtendo as classes preditas

# Labels reais
y_new = y_test[:3]

# Exibindo as imagens, previsões e labels reais
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_new[i], cmap="binary")
    plt.title(f"Prev: {class_names[y_pred[i]]}\nReal: {class_names[y_new[i]]}")
    plt.axis("off")
plt.suptitle("Previsões do Modelo vs Labels Reais", fontsize=16)
plt.show()