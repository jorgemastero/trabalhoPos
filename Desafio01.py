# Introdução
print("Bem-vindo à Aventura Pythonica!")
print("Você acorda em uma floresta onde os animais falam em código...\n")

# Variáveis do jogo
inventario = []
vida = 100

# Função principal
def main():
    nome = input("Como devo te chamar, jovem programador? ")
    print(f"\n{nome}, para escapar desta dimensão, você precisa dominar 3 conceitos Python!")
    
    fase_1()  # Primeiro desafio

def fase_1():
    print("\n--- Fase 1: Variáveis ---")
    print("Um coelho falante te desafia:")
    print("'Declare uma variável chamada 'arma' com o valor 'Espada de Código' para continuar.'\n")
    
    resposta = input("Escreva o código Python aqui: ")
    
    if resposta.strip() == "arma = 'Espada de Código'":
        inventario.append("Espada de Código")
        print("\n✅ Correto! Você ganhou: Espada de Código (agora no seu inventário).")
        fase_2()
    else:
        print("\n❌ Errado! Tente novamente. Dica: use aspas simples ou duplas para strings.")

def fase_2():
    print("\n--- Fase 2: Condicionais ---")
    print("Um dragão bloqueia o caminho! Ele só deixa passar se 'vida' for maior que 50.")
    print(f"Sua vida atual: {vida}\n")
    
    resposta = input("Escreva um 'if' para verificar se você pode passar: ")
    
    if "if vida > 50:" in resposta or "if vida >50:" in resposta:
        print("\n🐉 'Você pode passar!' O dragão se afasta.")
        fase_3()
    else:
        print("\nO dragão cospe fogo! Dica: Use 'if vida > 50:'.")

def fase_3():
    print("\n--- Fase 3: Loops ---")
    print("Você encontra uma porta mágica que só abre com um loop!\n")
    print("Ela exige que você imprima 'Abracadabra' 3 vezes usando 'for'.\n")
    
    resposta = input("Digite o loop: ")
    
    if "for i in range(3):" in resposta and "print('Abracadabra')" in resposta:
        print("\n✨ A porta se abre! Você escapou da dimensão Python!")
        print("Parabéns! Você aprendeu:")
        print("- Variáveis\n- Condicionais\n- Loops")
    else:
        print("\nA porta não reage. Dica: Use 'for i in range(3):' e 'print()'.")

if __name__ == "__main__":
    main()