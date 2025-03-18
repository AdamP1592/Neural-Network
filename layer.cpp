
#include <vector>
#include <random>
#include <string>
#include "neuron.h"
#include "layer.h"
struct network{


};



int main(){
    
    std::vector<double> inputs = {1.0, 3.0, 1.5};
    std::vector<double> weights = {0.5, -0.5, 1.0};
    double bias = 0.1;

    Neuron n(bias);
    
    double output = n.activate(inputs);

    std::cout << "Enter layer structure\n" 
    << "Ex: 1 2 1, Yields:\n"
    << "\t0 \n 0\t0  0\n";
    std::string layerStructure;

    std::cin >> layerStructure;

    n.printWeights();
    std::cout << "Feed-forward output: " << output << std::endl;
    return 0;
}