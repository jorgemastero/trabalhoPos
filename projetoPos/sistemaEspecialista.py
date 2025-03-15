import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

#------------ Função de ativação Swish -------------------------------------
def swish(x):
    return x * keras.activations.sigmoid(x)

# Função para gerar dados de treinamento
def train_datas(numbers):
    X = []
    y = []
    for i in range(numbers):
        number1 = np.random.randint(1, 100)
        number2 = np.random.randint(1, 100)
        operations = np.random.randint(0, 4)

        if operations == 0:
            result = number1 + number2
        elif operations == 1:
            result = number1 - number2
        elif operations == 2:
            result = number1 * number2
        elif operations == 3:
            result = number1 / number2 if number2 != 0 else 0

        number1 = number1 / 100  
        number2 = number2 / 100
        result = result / (100 * 100)
        
        operation_one_hot = [0, 0, 0, 0]
        operation_one_hot[operations] = 1 

        X.append([number1, number2] + operation_one_hot)
        y.append(result)

    return np.array(X), np.array(y)

# Gerar dados
X, y = train_datas(100000)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir o modelo
model = keras.models.Sequential([
    keras.layers.Dense(64, input_shape=(6,), activation=swish),
    keras.layers.Dense(32, activation=swish),
    keras.layers.Dense(1)
])

# Compilar o modelo
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test))

# Função para verificar a resposta do usuário
def sistema_especialista(resposta_correta, resposta_estudante, tolerancia=0.01):
    if abs(resposta_estudante - resposta_correta) <= tolerancia:
        return True, "Parabéns! Resposta correta."
    else:
        return False, f"Resposta incorreta: {resposta_estudante:.2f}. A resposta correta é {resposta_correta:.2f}. Tente novamente!"

# Variáveis globais
resultado_real_global = 0
contador_problemas = 0
respostas_corretas = 0
dificuldade = 1  # 1: Fácil, 2: Médio, 3: Difícil
operacoes = {
    "adição": {"acertos": 0, "tentativas": 0},
    "subtração": {"acertos": 0, "tentativas": 0},
    "multiplicação": {"acertos": 0, "tentativas": 0},
    "divisão": {"acertos": 0, "tentativas": 0}
}
operacao_nome_global = ""  # Variável global para armazenar a operação atual

# Função para gerar um problema e verificar a resposta
def gerar_problema():
    global contador_problemas, respostas_corretas, resultado_real_global, dificuldade, operacao_nome_global

    if contador_problemas >= 10:
        mostrar_desempenho()
        contador_problemas = 0
        respostas_corretas = 0
        for operacao in operacoes:
            operacoes[operacao]["acertos"] = 0
            operacoes[operacao]["tentativas"] = 0
        return

    # Gerar um problema aleatório
    if dificuldade == 1:
        num1 = np.random.randint(1, 50)
        num2 = np.random.randint(1, 50)
    elif dificuldade == 2:
        num1 = np.random.randint(-50, 100)
        num2 = np.random.randint(-50, 100)
    elif dificuldade == 3:
        num1 = np.random.randint(-100, 100)
        num2 = np.random.randint(-100, 100)

    operacao = np.random.randint(0, 4)

    # Normalizar os números de entrada como no treinamento
    num1_norm = num1 / 100
    num2_norm = num2 / 100

    # Criar a representação One-Hot da operação
    operation_one_hot = [0, 0, 0, 0]
    operation_one_hot[operacao] = 1  

    # Criar a entrada do modelo
    entrada = np.array([[num1_norm, num2_norm] + operation_one_hot])

    # Normalizar a entrada com o mesmo scaler usado no treinamento
    entrada = scaler.transform(entrada)

    # Fazer a previsão com a rede neural
    resultado_predito = model.predict(entrada)[0][0] * (100 * 100)

    # Calcular o resultado real
    if operacao == 0:
        resultado_real = num1 + num2
        simbolo = "+"
        operacao_nome_global = "adição"
    elif operacao == 1:
        resultado_real = num1 - num2
        simbolo = "-"
        operacao_nome_global = "subtração"
    elif operacao == 2:
        resultado_real = num1 * num2
        simbolo = "×"
        operacao_nome_global = "multiplicação"
    elif operacao == 3:
        resultado_real = num1 / num2 if num2 != 0 else 0
        simbolo = "÷"
        operacao_nome_global = "divisão"

    # Atualizar a interface com o problema
    problema_label.config(text=f"Quanto é {num1} {simbolo} {num2}?")
    resultado_real_global = resultado_real
    contador_problemas += 1
    operacoes[operacao_nome_global]["tentativas"] += 1

# Função para verificar a resposta do usuário
def verificar_resposta():
    global respostas_corretas, operacao_nome_global
    try:
        resposta_usuario = float(resposta_entry.get())
        correto, resultado = sistema_especialista(resultado_real_global, resposta_usuario)
        if correto:
            respostas_corretas += 1
            operacoes[operacao_nome_global]["acertos"] += 1  # Atualiza diretamente a contagem de acertos
        messagebox.showinfo("Resultado", resultado)
        gerar_problema()
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira um número válido.")

# Função para mostrar o desempenho final
def mostrar_desempenho():
    desempenho = "Desempenho Final:\n"
    for operacao, dados in operacoes.items():
        tentativas = dados["tentativas"]
        acertos = dados["acertos"]
        if tentativas > 0:
            taxa_acerto = (acertos / tentativas) * 100
            desempenho += f"{operacao.capitalize()}: {acertos}/{tentativas} ({taxa_acerto:.2f}%)\n"
        else:
            desempenho += f"{operacao.capitalize()}: Nenhuma tentativa\n"
    messagebox.showinfo("Desempenho", desempenho)

# Função para escolher a dificuldade
def escolher_dificuldade(nova_dificuldade):
    global dificuldade
    dificuldade = nova_dificuldade
    messagebox.showinfo("Dificuldade", f"Dificuldade alterada para {'Fácil' if dificuldade == 1 else 'Médio' if dificuldade == 2 else 'Difícil'}.")

# Interface gráfica com Tkinter
root = tk.Tk()
root.title("Sistema Especialista de Matemática")

# Label para exibir o problema
problema_label = tk.Label(root, text="Clique em 'Gerar Problema' para começar.", font=("Arial", 14))
problema_label.pack(pady=20)

# Campo de entrada para a resposta do usuário
resposta_entry = tk.Entry(root, font=("Arial", 14))
resposta_entry.pack(pady=10)

# Botão para verificar a resposta
verificar_button = tk.Button(root, text="Verificar Resposta", command=verificar_resposta, font=("Arial", 14))
verificar_button.pack(pady=10)

# Botão para gerar um novo problema
gerar_button = tk.Button(root, text="Gerar Problema", command=gerar_problema, font=("Arial", 14))
gerar_button.pack(pady=10)

# Botões para escolher a dificuldade
facil_button = tk.Button(root, text="Fácil", command=lambda: escolher_dificuldade(1), font=("Arial", 14))
facil_button.pack(side=tk.LEFT, padx=10)

medio_button = tk.Button(root, text="Médio", command=lambda: escolher_dificuldade(2), font=("Arial", 14))
medio_button.pack(side=tk.LEFT, padx=10)

dificil_button = tk.Button(root, text="Difícil", command=lambda: escolher_dificuldade(3), font=("Arial", 14))
dificil_button.pack(side=tk.LEFT, padx=10)

# Iniciar a interface gráfica
root.mainloop()