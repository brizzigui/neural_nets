// Demo da propagação de uma rede neural
// Os valores são inicializados aleatóriamente

#include <iostream>
#include <vector>
#include <math.h>

#define PI 3.1415

using namespace std;

struct neuron
{
    double value;
    // the value held by the neuron (may be input or the result of the linear combination)

    vector<double> weight;
    // weights held by neuron in layer k represents the transition from k-1 to k

    double bias;
    // bias is held by each neuron, but can be set by layer on initialization
};

typedef struct neuron neuron;
typedef vector<vector<neuron>> network_t;

double rand_double()
{
    return (double)rand()/RAND_MAX;
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

            n.value = rand_double();
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
            cout << "\tValor do neuronio: " << network[l][n].value << "; Bias do neuronio: " << network[l][n].bias << endl;
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

double relu(double v)
{
    // not actualy relu
    v = (v > 0) ? v : 0;
    v = (v > 1) ? 1 : v;
    return v;
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
                acc += (network[l-1][origin].value * network[l][n].weight[origin] + network[l][n].bias);
            }

            network[l][n].value = relu(acc);
        }
    }
    
    return network;
}

network_t set_input(network_t network, vector<double> values)
{
    // iterates over the neurons of the input layer, setting their values from the values vector
    for (size_t j = 0; j < values.size(); j++)
    {
        network[0][j].value = values[j];
    }

    return network;
}

double compare(vector<double> in, vector<double> out, network_t network)
{
    // calculates error for training example using square of difference

    double error = 0.0;

    // for each output neuron
    for (size_t j = 0; j < out.size(); j++)
    {
        double expected = out[j];
        double actual = network[network.size()-1][j].value;

        double diff = abs(actual-expected);

        error += diff;
    }
    
    return error;
}

void sine_demo()
{
    vector<vector<double>> inputs = {{0.0}, {PI/4}, {PI/2}, {3*PI/4}, {PI}};
    vector<vector<double>> expected_outputs = {{0.0}, {sqrt(2.0)/2.0}, {1.0}, {sqrt(2.0)/2.0}, {0.0}};
    double error = 1000000.0;
    size_t epoch = 0;
    network_t network;


    while (error > 0.5)
    {
        epoch++;
        //cout << "Epoch" << epoch << endl;

        vector<int> layer_sizes = {1, 8, 1};
        network = create_network(3, layer_sizes);

        error = 0;
        for (size_t in = 0; in < inputs.size(); in++)
        {
            network = set_input(network, inputs[in]);
            network = propagate(network);
            double partial_error = compare(inputs[in], expected_outputs[in], network);
            error += partial_error;
        }
        
        //cout << "Erro: " << error << endl;
        //cout << "-----" << endl << endl;

        // show(network);
    }

    cout << "found after " << epoch << " epochs, with error = " << error << endl;
    
    cout << "testing!" << endl;
    while(true)
    {
        cout << "insert a value: " << endl;
        double in_value;
        cin >> in_value;

        vector<double> arg = {in_value};
        network = set_input(network, arg);

        network = propagate(network);

        cout << "prediction: " << network[network.size()-1][0].value << endl;
        cout << "real: " << sin(in_value) << endl;
        cout << "absolute difference: " << fabs(network[network.size()-1][0].value - sin(in_value)) << endl;
        cout << "-----" << endl << endl;

    }

    
}

int main()
{
    srand(time(NULL));
    // sine_demo();

    vector<int> layer_sizes = {1, 8, 1};
    network_t network = create_network(3, layer_sizes);
    network = propagate(network);
    show(network);

    return 0;
}