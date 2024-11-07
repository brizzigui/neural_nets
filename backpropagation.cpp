// Demo da propagação de uma rede neural
// Os valores são inicializados aleatóriamente

#include <iostream>
#include <vector>
#include <math.h>

#define PI 3.1415

using namespace std;

struct neuron
{
    double value_comb;
    // the value coming out of the neuron not by activation function

    double value_out;
    // the value coming out of the neuron impacted by activation function

    vector<double> weight;
    // weights held by neuron in layer k represents the transition from k-1 to k

    double bias;
    // bias is held by each neuron, but can be set by layer on initialization

    double error;
    // this is the little delta error computed by backprop
};

typedef struct neuron neuron;
typedef vector<vector<neuron>> network_t;
typedef vector<vector<double>> matrix_t;

double rand_double()
{
    return (double)rand()/RAND_MAX/10;
}

double sigmoid(double v)
{
    return 1 / (1 + exp(-v));
}

double sigmoid_dif(double v)
{
    return sigmoid(v) * (1 - sigmoid(v));
}

network_t create_network(int layers, vector<int> layer_sizes)
{
    network_t network;

    for (size_t l = 0; l < layers; l++)
    {
        int layer_size = layer_sizes[l];

        double layer_bias = (l==0) ? 0 : rand_double(); 
        // sets bias to 0 if it's the input layer
        
        vector<neuron> layer;
        for (size_t i = 0; i < layer_size; i++)
        {
            neuron n;
            // inicia valores randomicamente

            n.value_comb = rand_double();
            n.value_out = sigmoid(n.value_comb);
            for (size_t w = 0; l != 0 && w < network[l-1].size(); w++)
            {
                // adiciona peso relativo a cada neuronio da camada l-1
                n.weight.push_back(rand_double());
            }
            
            n.bias = layer_bias; // bias can be changed as desired

            layer.push_back(n);
        }
        
        network.push_back(layer);
    }

    return network;
}

void show(network_t network)
{
    for(size_t l = 0; l < network.size(); l++)
    {
        cout << "Camada #" << l << endl;
        for (size_t n = 0; n < network[l].size(); n++)
        {
            cout << "\tValor do neuronio: " << network[l][n].value_out << "; Bias do neuronio: " << network[l][n].bias << endl;
            cout << "\tValor do erro: " << network[l][n].error << endl;
            
            cout << "\tPesos do neuronio k-1 para k: | ";
            for (auto w : network[l][n].weight)
            {
                cout << w << " | ";
            }

            if(network[l][n].weight.empty())
            {
                cout << "n/a |";
            }

            cout << endl;       
        }

        cout << endl;
    }

    cout << endl;
}

network_t propagate(network_t network)
{
    for (size_t l = 1; l < network.size(); l++)
    {
        for (size_t n = 0; n < network[l].size(); n++)
        {
            double acc = 0;
            for (size_t origin = 0; origin < network[l-1].size(); origin++)
            {
                acc += (network[l-1][origin].value_out * network[l][n].weight[origin] + network[l][n].bias);
            }

            network[l][n].value_comb = acc;
            network[l][n].value_out = sigmoid(acc);
        }
    }
    
    return network;
}


double compare(matrix_t in, matrix_t out, network_t network)
{
    // calculates error for training example batch using square of difference

    double error = 0.0;

    // for each output neuron
    for (size_t ex = 0; ex < out.size(); ex++)
    {
        for (size_t j = 0; j < out[ex].size(); j++)
        {
            double expected = out[ex][j];
            double actual = network[network.size()-1][j].value_out;

            double diff = (actual-expected) * (actual-expected);

            error += diff;
        }
    }
    
    return error;
}

network_t backpropagate(network_t network, matrix_t in, matrix_t out, double learning_rate)
{
    // this function handles the backpropagation algorithm

    // initializes error of every layer with 0.0
    for (size_t l = 0; l < network.size(); l++)
    {
        for (size_t j = 0; j < network[network.size()-1].size(); j++)
        {
            network[l][j].error = 0.0;
        }
    }
    
    for (int ex = 0; ex < in.size(); ex++)
    {      
        for (size_t j = 0; j < network[0].size(); j++)
        {
            network[0][j].value_out = in[ex][j];
            network[0][j].value_comb = in[ex][j];
        }
        network = propagate(network);

        // calculates error for output layer
        for (size_t j = 0; j < out[ex].size(); j++)
        {
            double expected = out[ex][j];
            double actual = network[network.size()-1][j].value_out;

            double diff = (actual-expected);

            network[network.size()-1][j].error += diff * sigmoid_dif(network[network.size()-1][j].value_out);
        }


        // calculates error for every remaining layer, backpropagating
        // for every layer (excluding input (not needed) and output (already computed))
        for (size_t l = network.size()-2; l > 0; l--)
        {
            // for every neuron in the smaller layer
            for (size_t k = 0; k < network[l].size(); k++)
            {
                double sum = 0.0;
                // for every neuron in the bigger layer
                for (size_t j = 0; j < network[l+1].size(); j++)
                {
                    sum += network[l+1][j].weight[k] * network[l+1][j].error;
                }

                network[l][k].error += sum * sigmoid_dif(network[l][k].value_out);
            }
        }
    }

    // adjusts weights
    for (size_t l = 1; l < network.size(); l++)
    {
        // for every neuron in the bigger layer
        for (size_t j = 0; j < network[l].size(); j++)
        {
            // for every neuron in the smaller layer
            for (size_t k = 0; k < network[l-1].size(); k++)
            {
                double delta = network[l][j].error * sigmoid_dif(network[l][j].value_comb);
                network[l][j].weight[k] -= learning_rate * delta * network[l-1][k].value_out;
            }
        }
    }

    // adjusts bias
    for (size_t l = 1; l < network.size(); l++)
    {
        // for every neuron in the bigger layer
        for (size_t j = 0; j < network[l].size(); j++)
        {
            // for every neuron in the smaller layer
            for (size_t k = 0; k < network[l-1].size(); k++)
            {
                double delta = network[l][j].error * sigmoid_dif(network[l][j].value_comb);
                network[l][j].bias -= learning_rate * delta;
            }
        }
    }
    

    return network;
}

void test(network_t network)
{
    // lets user test network
    // not generic! only works for sine demo

    cout << "\n\n\n";
    cout << "----------------------------------------------------" << endl;
    cout << "Insira um valor de entrada: ";

    double in;
    cin >> in;

    network[0][0].value_out = in;
    network[0][0].value_comb = in;

    network = propagate(network);

    double prediction = network[network.size()-1][0].value_out;
    double real = sin(in);
    double example_error = abs(prediction-real);

    cout << "----------------------------------------------------" << endl;
    cout << "\tValor previsto pela rede = " << prediction << endl;
    cout << "\tValor real = " << real << endl;
    cout << "\tErro absoluto = " << example_error << endl;
    cout << "----------------------------------------------------" << endl;


    return;
}

void sine_demo()
{
    // demo for sine aproximation
    // not generic! only works for sine demo

    vector<vector<double>> inputs = {{0.0}, {PI/4}, {PI/2}, {3*PI/4}, {PI}};
    vector<vector<double>> expected_outputs = {{0.0}, {sqrt(2.0)/2.0}, {1.0}, {sqrt(2.0)/2.0}, {0.0}};
    vector<int> layers = {1, 4, 1};
    double error = 1000000.0;
    size_t epoch = 0;
    network_t network;

    network = create_network(3, layers);
    

    for (size_t epoch = 0; epoch < 1000; epoch++)
    {
        network = backpropagate(network, inputs, expected_outputs, 0.5);

        cout << "Epoch #" << epoch << " - error: " << compare(inputs, expected_outputs, network) << endl;

    }
        show(network);


    while (true)
    {
        test(network);
    }
    
}

int main()
{
    srand(time(NULL));
    sine_demo();

    return 0;
}