import struct
import sys
import random

import backpropagation as mind

def read_bmp(path: str) -> list:
    pixel_values = []

    with open(path, "rb") as file:
        file.read(54)

        for _ in range(900):
            pixel_values.append((int.from_bytes(file.read(1))/(255)+int.from_bytes(file.read(1))/(255)+int.from_bytes(file.read(1))/(255))/3)

    return pixel_values


def get_digits_dataset() -> list:
    dataset = []

    for i in range(4):
        expected = [0] * 4
        expected[i] = 1

        pos = [k for k in range(15)]
        random.shuffle(pos)

        for k in range(50):
            path = f"./images/{i}/{k}.bmp"
            dataset.append([read_bmp(path), expected])

    return dataset


def get_max_index(l: list) -> int:
    max_v = -float("inf")
    max_i = 0

    for i in range(len(l)):
        if l[i] > max_v:
            max_v = l[i]
            max_i = i

    return max_i
    

def use_from_file(network: list, file_path: str, actual: int) -> None:
    values = read_bmp(file_path)

    mind.forward(network, values)
    answer = mind.get_propagation_output(network)

    prediction = get_max_index(answer)

    print(f"A previsão é {prediction} com {answer[prediction]*100:.2f}% de confiança")
    print(f"O valor real é {actual}")
    print("Probabilidades completas:")
    print(answer)
    print()

def use_loop(network) -> None:
    print("Insira o nome de um arquivo ou digite /quit para sair.")
    print("Os comandos 'from [path]' e 'back' são usados para trocar de diretório base.")

    path = ""
    while True:
        try:
            if path == "":
                option = input("> ")
            else:
                option = input(f">{path}")
            
            if option == "/quit":
                return
            
            elif option[0:4] == "from":
                path = option[4:].strip()

            elif option[0:4] == "back":
                path = ""

            else:
                actual_path = path + option
                print(actual_path)
                use_from_file(network, actual_path, "None")

        except FileNotFoundError:
            print("Inválido. Tente novamente ou use /quit para sair.")

def main() -> None:
    print("Demo de treinamento de rede neural para reconhecimento de dígitos\n")

    print("Lendo dados.")
    dataset = get_digits_dataset()

    print("Criando rede.")
    network = mind.create_network(4, [900, 100, 100, 4])

    print("Treinando rede.")
    print("---------------\n")
    network = mind.train(network, dataset, 0.002, 500)

    print()

    use_from_file(network, "./images/test/zero_0.bmp", 0)
    use_from_file(network, "./images/test/one_0.bmp", 1)
    use_from_file(network, "./images/test/two_0.bmp", 2)
    use_from_file(network, "./images/test/three_0.bmp", 3)

    use_loop(network)



if __name__ == "__main__":
    main()