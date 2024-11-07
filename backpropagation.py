import random
import math

class Neuron:
    ''' Classe que representa cada neurônio '''
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = 0
        self.error = 0

    def __repr__(self) -> str:
        string = f"\tValor: {self.output} - Erro: {self.error}\n\tPesos: {self.weights} - Bias: {self.bias}"
        return string


def create_network(n_layers: int, layers_sizes: list) -> list:
    ''' Cria rede neural '''
    
    network = []

    for l in range(n_layers):

        layer = []

        if l == 0:

            bias = 0

            for j in range(layers_sizes[l]):
                weights = []
                neuron = Neuron(weights, bias)

                layer.append(neuron)



        else:
            bias = random.uniform(-1.0, 1.0)
            for j in range(layers_sizes[l]):
                # índice j representa cada neuronio da camada l
                weights = [random.uniform(-1.0, 1.0) for k in range(layers_sizes[l-1])]
                neuron = Neuron(weights, bias)

                layer.append(neuron)

        network.append(layer.copy())

    return network

def sigmoid(v: float) -> float:
    ''' Função de ativação sigmoide '''

    # v is the pure output
    return 1.0 / (1.0 + math.exp(-v))

def sigmoid_prime(v: float) -> float:
    ''' Derivada da função de ativação sigmoide (recebe sigmoid(linearcombination) como argumento)'''


    # v is the sigmoided output
    return v * (1.0 - v)


def forward(network: list[list[Neuron]], values: list) -> None:
    ''' Faz propagação da ativação '''

    for j in range(len(network[0])):
        network[0][j].output = values[j]

    for l in range(1, len(network)):
        for j in range(len(network[l])):
            network[l][j].output = network[l][j].bias
            for k in range(len(network[l-1])):
                network[l][j].output += network[l-1][k].output * network[l][j].weights[k]  

            network[l][j].output = sigmoid(network[l][j].output)


def backpropagate(network: list[list[Neuron]], expected: list) -> None:
    ''' Faz backpropagation do erro '''

    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []

        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += neuron.weights[j] * neuron.error
                errors.append(error)

        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron.output - expected[j])  

        for j in range(len(layer)):
            neuron = layer[j]
            neuron.error = errors[j] * sigmoid_prime(neuron.output)

def update_weights(network: list[list[Neuron]], values: list, learning_rate: float) -> None:
    ''' Atualiza os pesos da rede, fazendo descida de gradiente '''


    for i in range(1, len(network)):
        values = [neuron.output for neuron in network[i-1]]

        for neuron in network[i]:
            for j in range(len(values)):
                neuron.weights[j] -= learning_rate * neuron.error * values[j]

            neuron.bias -= learning_rate * neuron.error
                

def print_network(network: list) -> None:
    ''' Imprime a rede de maneira formatada '''

    for l_idx, layer in enumerate(network):
        print(f"----- Camada #{l_idx} -----")
        for n_idx, neuron in enumerate(layer):
            print(f"Neurônio #{n_idx}:\n{neuron}")
        print()

def get_propagation_output(network) -> list:
    ''' Retorna saídas da rede '''

    return [network[-1][j].output for j in range(len(network[-1]))]

def train(network: list[list[Neuron]], dataset: list, learning_rate: float, n_epoch: int) -> list[list[Neuron]]:
    ''' Comanda o treino da rede '''

    for epoch in range(n_epoch):
        error_acc = 0
        for point in dataset:
            forward(network, point[0])  

            outputs = get_propagation_output(network)
            expected = point[1]

            error_acc += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])

            backpropagate(network, expected)
            update_weights(network, point[0], learning_rate)

        print(f"> Epoch #{epoch:02d} - error = {error_acc}")

    return network

def use(network: list[list[Neuron]], is_sine_test: bool) -> None:
    ''' Função permite que o usuário use a rede treinada, vendo suas saídas '''

    print("Vamos usar nossa rede treinada!")
    while True:
        print("-------------------------------")
        v = [*map(float, input("Digite uma entrada: ").split())]

        forward(network, v)
        answer = get_propagation_output(network)

        if is_sine_test:
            print(f"Resultado previsto = {answer[0]}")
            print(f"Resultado esperado = {math.sin(v[0])}")
            print(f"Diferença absoluta = {abs(math.sin(v[0]) - answer[0])}")

        else:
            print(f"Resultado previsto = {answer}")


def main() -> None:
    sine_dataset = [
        [[0],[math.sin(0)]],
        [[math.pi/4],[math.sin(math.pi/4)]],
        [[math.pi/2],[math.sin(math.pi/2)]],
        [[3*math.pi/4],[math.sin(3*math.pi/4)]],
        [[math.pi],[math.sin(math.pi)]]
    ]

    dataset = sine_dataset

    network = create_network(3, [1, 8, 1])
    network = train(network, dataset, 0.5, 20000)

    use(network, True)


if __name__ == "__main__":
    main()