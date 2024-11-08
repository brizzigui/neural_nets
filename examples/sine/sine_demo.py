'''
    Código para treinamento de rede neural de aproximação da funçao seno.

    Não necessita de bibliotecas auxiliares.
    Necessita do arquivo mind.py, que contém as funções de treinamento.
'''


import mind
from mind import Neuron
import math

def use(network: list[list[Neuron]], is_sine_test: bool) -> None:
    ''' Função permite que o usuário use a rede treinada, vendo suas saídas '''

    print("Vamos usar nossa rede treinada!")
    while True:
        print("-------------------------------")
        v = [*map(float, input("Digite uma entrada: ").split())]

        mind.forward(network, v)
        answer = mind.get_propagation_output(network)

        if is_sine_test:
            print(f"Resultado previsto = {answer[0]}")
            print(f"Resultado esperado = {math.sin(v[0])}")
            print(f"Diferença absoluta = {abs(math.sin(v[0]) - answer[0])}")

        else:
            print(f"Resultado previsto = {answer}")


def get_basic_sine_dataset() -> list:
    return [
        [[0],[math.sin(0)]],
        [[math.pi/4],[math.sin(math.pi/4)]],
        [[math.pi/2],[math.sin(math.pi/2)]],
        [[3*math.pi/4],[math.sin(3*math.pi/4)]],
        [[math.pi],[math.sin(math.pi)]]
    ]


def get_advanced_sine_dataset() -> list:
    return [[[i/100],[math.sin(i/100)]] for i in range(314)]

def main() -> None:
    basic_sine_dataset = get_basic_sine_dataset()
    sine_dataset = get_advanced_sine_dataset()

    dataset = sine_dataset

    network = mind.create_network(3, [1, 8, 1])
    network = mind.train(network, dataset, 0.25, 1000)

    use(network, True)


if __name__ == "__main__":
    main()