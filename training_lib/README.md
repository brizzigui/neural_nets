# Módulo para criação, manipulação e treinamento de redes neurais

## Exemplo de uso:

- seja X = [x1, x2, x3, ..., xn] um conjunto de n entradas
- seja Y = [y1, y2, y3, ..., yn] um conjunto de n saídas

Um dataset será:

``` python
dataset =   [
                [x1, y1],
                [x2, y2],
                [x3, y3]
            ]
```

em que **xn e yn são listas** de tamanho igual ao tamanho da camada de entrada 
e o tamanho da camada de saída, respectivamente.



Para criar uma rede use:
``` python
import mind

network = mind.create_network(n_layers=NUMERO_DE_CAMADAS,   layers_sizes=LISTA_DOS_TAMANHOS)
```

A ```LISTA_DOS_TAMANHOS``` terá de ter o mesmo tamanho que o ```NUMERO_DE_CAMADAS```.



Para treinar, use: 
```python
network = mind.train(network, dataset, learning_rate=TAXA_APRENDIZADO, n_epoch=NUMERO_EPOCAS)
```

em que a ```TAXA_APRENDIZADO``` é um valor para o passo de aprendizado *(preferivelmente menor que 1 e maior que 10^-6)* e o ```NUMERO_EPOCAS``` é um valor para as iterações realizadas *(preferivelmente maior que 10 e menor que 1000)*.


Para usar a rede treinada, use:

```python
mind.forward(network, ENTRADA)
```

em que a ```ENTRADA``` é a lista de valores de entrada
e obtenha a saída com:
```python
answer = mind.get_propagation_output(network)
print(answer)
```