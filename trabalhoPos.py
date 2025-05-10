import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os

def load_assets(model_path='best_math_model.keras', scaler_path='scaler_gpu.pkl'):
    """Carrega o modelo e o scaler salvos"""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Arquivos do modelo ou scaler não encontrados")
    
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def prepare_input(number1, number2, operation, max_value=100.0):
    """Prepara a entrada no formato esperado pelo modelo"""
    operation_map = {'+': 0, '-': 1, '*': 2, '/': 3}
    
    # Cria o vetor de características
    features = np.zeros(6)
    features[0] = number1 / max_value  # Normaliza number1
    features[1] = number2 / max_value  # Normaliza number2
    features[2 + operation_map[operation]] = 1  # One-hot encoding da operação
    
    return features.reshape(1, -1)

def test_model():
    """Função interativa para testar o modelo"""
    try:
        model, scaler = load_assets()
        print("✅ Modelo e scaler carregados com sucesso!")
        print(f"🔢 Formato de entrada esperado: {model.input_shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar os assets: {e}")
        return

    print("\n🧮 TESTADOR DE MODELO MATEMÁTICO")
    print("="*50)
    print("Instruções:")
    print("- Digite operações no formato: NUM1 OPERADOR NUM2")
    print("- Operadores suportados: +, -, *, /")
    print("- Exemplo: 5 + 3 ou 10.5 * 2")
    print("- Digite 'sair' para encerrar")
    print("="*50)

    while True:
        try:
            user_input = input("\n➡️ Digite uma operação: ").strip()
            if user_input.lower() == 'sair':
                print("👋 Encerrando o programa...")
                break

            # Processa a entrada
            parts = user_input.split()
            if len(parts) != 3:
                print("⚠️ Formato inválido. Use: NUMERO OPERADOR NUMERO")
                continue

            try:
                num1 = float(parts[0])
                num2 = float(parts[2])
                op = parts[1]

                if op not in ['+', '-', '*', '/']:
                    print("⚠️ Operador inválido. Use +, -, * ou /")
                    continue

                if op == '/' and abs(num2) < 1e-10:
                    print("⚠️ Divisão por zero não permitida")
                    continue

                # Prepara a entrada
                raw_input = prepare_input(num1, num2, op)
                scaled_input = scaler.transform(raw_input)

                # Faz a predição
                prediction = model.predict(scaled_input, verbose=0)[0][0]

                # Calcula o valor real (considerando normalização)
                max_value = 100.0  # Deve corresponder ao usado no treinamento
                if op == '+':
                    real = (num1 + num2) / max_value
                elif op == '-':
                    real = (num1 - num2) / max_value
                elif op == '*':
                    real = (num1 * num2) / (max_value * max_value)
                elif op == '/':
                    real = (num1 / num2) / max_value

                # Desnormaliza a predição
                if op in ['+', '-', '/']:
                    denormalized_pred = prediction * max_value
                    denormalized_real = real * max_value
                else:  # Multiplicação
                    denormalized_pred = prediction * max_value * max_value
                    denormalized_real = real * max_value * max_value

                # Exibe resultados
                print("\n📊 RESULTADOS:")
                print(f"Operação: {num1} {op} {num2}")
                print(f"Previsão do modelo: {denormalized_pred:.6f}")
                print(f"Valor real:         {denormalized_real:.6f}")
                print(f"Diferença absoluta: {abs(denormalized_pred - denormalized_real):.6f}")
                print(f"Erro percentual:    {abs(denormalized_pred - denormalized_real)/abs(denormalized_real)*100:.2f}%")
                print("="*50)

            except ValueError:
                print("⚠️ Valores numéricos inválidos")
                continue
            except Exception as e:
                print(f"❌ Erro durante a previsão: {e}")
                continue

        except KeyboardInterrupt:
            print("\n👋 Programa encerrado pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            continue

if __name__ == "__main__":
    # Configurações para melhor performance
    tf.config.optimizer.set_jit(True)
    tf.keras.backend.clear_session()
    
    # Verifica se a GPU está disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU detectada: {gpus[0].name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ Erro na configuração da GPU: {e}")
    else:
        print("ℹ️ Nenhuma GPU detectada, usando CPU")
    
    test_model()