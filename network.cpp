#include "network.h"
struct network{
    std::vector<Layer> layers;
    double learningRate = 1;
    int step = 0;
    void setupNetwork(std::vector<int> structure){
        
        Layer inputLayer = Layer(structure[0]);
        layers.push_back(inputLayer);
        
        for(int i = 1; i < structure.size(); i++){
            std::vector<std::reference_wrapper<Neuron>> prevLayerNeuronReferences;

            Layer thisLayer = Layer(structure[i]);
            //building the references for the next layer
            for(int j = 0; j < structure[i-1]; j++){
                Neuron& refNeuron = layers[i-1].getConnection(j);
                prevLayerNeuronReferences.push_back(refNeuron);
            }
            thisLayer.setupReferences(prevLayerNeuronReferences);
            

            layers.push_back(thisLayer);
            
        }

    }
    void updateLearningRate(){
        learningRate = 1/std::exp(0.1 * step);
    }

    //set activationValue for the input layer. Iterate all subsequent layers as a standard pass
    void forwardPass(std::vector<double>& inputValues){
        Logger::log("forwardPass:\n");
        //catch cases for errors
        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return;
        }
        if(inputValues.size() != layers[0].size){
            std::cerr << "Error: input size (" << layers[0].size
            << ") does not match network imports size (" << inputValues.size() << ").\n";
            return;
        }
        //set all the input layer activation values to the inputs
        for (int i = 0; i < layers[0].size; i++){
            layers[0].layer[i].activationValue = inputValues[i];
            
            //logging activation
            std::ostringstream oss;
            oss << "Neuron: (0, " << i << ") Activation:" << inputValues[i];
            Logger::log(oss.str());
        }
        //iterate each non input layer, activate all neurons in the layers
        int size = layers.size();
        for(int i = 1; i < size; i++){
            //get the current layer reference
            Layer& thisLayer = layers[i];

            int layerSize = layers[i].size;
            for(int j = 0; j < layerSize; j++){
                //get current neuron reference
                Neuron& n = thisLayer.layer[j];
                n.activate();
                
                //logging activation
                std::ostringstream oss;
                oss << "Neuron: (" << i << ", " << j << ") Activation:" << n.activationValue;
                Logger::log(oss.str());
            }
        }
    }
    void backPropagate(std::vector<double> expectedValues){
        //catch case for empty network
        Logger::log("BackProp:");
        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return;
        }
        Layer &outputLayer = layers[layers.size() - 1];
        //catch case for size mismatch
        if(expectedValues.size() != outputLayer.size){
            std::cerr << "Error: expectedValues size (" << outputLayer.size
            << ") does not match network output size (" << expectedValues.size() << ").\n";
            return;
        }

        //back prop call for output layer
        for(int i = 0; i < outputLayer.size; i++){
            Neuron &n = outputLayer.layer[i];
            double err = n.backPropagateOutput(expectedValues[i], learningRate);

            //logging backprop
            std::ostringstream oss;
            oss << "Neuron: (" << layers.size() - 1  << ", " << i << ") backProp:" << err;
            Logger::log(oss.str());
        }
        //skip output layer
        for(int i = layers.size() - 2; i >= 0; i--){
            Layer &curLayer = layers[i];
            for(int j = 0; j < curLayer.size; j++){
                Neuron &n = layers[i].layer[j];
                double err = n.backPropagate(learningRate);
                std::ostringstream oss;
                oss << "Neuron: (" << i  << ", " << j << ") backProp Error:" << err;
                Logger::log(oss.str());
            }

        }
        step++;

    }

    void printNetwork(){
        int size = layers.size();
        for(int i = 0; i < size; i++){
            int layerSize = layers[i].size;
            Layer& thisLayer = layers[i];
            for(int j = 0; j < layerSize; j++){
                Neuron& n = thisLayer.layer[j];

                std::cout << n.activationValue << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};

std::vector<int> stringToStructure(std::string layerStructure){
    std::vector<int> structure;
    std::string thisInt;
    for(int i = 0; i < layerStructure.length(); i++){
       
        if(isdigit(layerStructure[i])){
            thisInt.push_back(layerStructure[i]);
        }else if(layerStructure[i] == ',' || i == layerStructure.length() - 1){
            if(!thisInt.empty()){
                structure.push_back(std::stoi(thisInt));
                thisInt.clear();
            }
        }
    }
    if(!thisInt.empty()){
        structure.push_back(std::stoi(thisInt));
        thisInt.clear();
    }
    //catch case for any unknown structure; 
    return structure;
}

void simpleTest(){
    std::vector<double> inputs = {1.0, 3.0, 1.5};
    std::vector<double> weights = {0.5, -0.5, 1.0};
    std::vector<double> expected = {0.5, 0};
    std::vector<int> structure;

    std::cout << "Enter layer structure\n" 
    << "Ex: 1, 2, 1, Yields:\n"
    << "\t0\n0\t\t0\n\t0\n";
    
    std::string layerStructure;
    std::getline(std::cin, layerStructure);
    system("clear");

    structure = stringToStructure(layerStructure);
    network neuralNetwork;
    neuralNetwork.setupNetwork(structure);

    neuralNetwork.printNetwork();

    neuralNetwork.forwardPass(inputs);
    neuralNetwork.printNetwork();

    neuralNetwork.backPropagate(expected);

    neuralNetwork.forwardPass(inputs);
    neuralNetwork.printNetwork();

}

std::vector<double> getLine(std::vector<std::vector<double>> &lines){
    std::vector<double> randomLine;
    if(!lines.empty()){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, lines.size() - 1);

        
        // Get a random index.
        int randomIndex = dist(gen);
        std::string randomLineString = "Random Line" + std::to_string(randomIndex) + "\n";
        Logger::log(randomLineString);
        
        
        // Get the random line.
        randomLine = lines[randomIndex];
        
        // Remove the random line from the vector.
        lines.erase(lines.begin() + randomIndex);

        
    }
    return randomLine;
}
double converToDouble(const std::string& s) {
    try {
        double value = std::stod(s);
        return value;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
    }
    return std::numeric_limits<double>::lowest();
}
std::vector<std::string> splitStringByComma(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    
    while (std::getline(stream, token, ',')) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<double> processDataPoint(const std::string& input){

    std::unordered_map<std::string, int> dict ={
        {"Iris-setosa", 0.0},
        {"Iris-versicolor", 1.0},
        {"Iris-virginica", 2.0}
    };


    std::vector<double> tokens;
    std::istringstream stream(input);
    std::string token;
    
    while (std::getline(stream, token, ',')) {
        double tokenDouble = converToDouble(token);

        if(tokenDouble == std::numeric_limits<double>::lowest()){
            tokenDouble = dict[token];
        }
        tokens.push_back(tokenDouble);
    }
    
    return tokens;
}

void hardTest(){
    std::string dataPath = "./data/iris.data";
    int startSize = 0;

    std::ifstream inFile(dataPath);  // Adjust the file name as needed
    if (!inFile) {
        std::cerr << "Error: Could not open file.\n";
        return;
    }
    std::vector<std::string> lines;
    std::vector<std::vector<std::string>> separatedLines;

    std::vector<std::vector<double>> processedLines;
    std::vector<double> processedLine;

    std::string line;

    while(std::getline(inFile, line)){
        lines.push_back(line);
        processedLine = processDataPoint(line);
        for(int i = 0; i < processedLine.size();i++){
            std::cout << processedLine[i] << std::endl;
        }
        processedLines.push_back(processedLine);
        separatedLines.push_back(splitStringByComma(line));
    }
    inFile.close();
    startSize = lines.size();
    
    network neuralNet;
    //data points are 5 points, 4 of them are inputs 1 is expected
    //there are 3 types of expected values
    std::vector<int> structure = {4, 8, 8, 3};

    neuralNet.setupNetwork(structure);

    //training
    std::cout << "Training: \n";
    while(processedLines.size() > startSize/4){

        std::string numLines = "Number of lines: "  + std::to_string(processedLines.size());
        Logger::log(numLines);

        std::vector<double> randomLine = getLine(processedLines);
        std::vector<double> expectedOutput = {0, 0, 0};

        std::vector<double> trainingData;
        //add all values except the last one to training data
        for(int i = 0; i < randomLine.size() - 1; i++){
            trainingData.push_back(randomLine[i]);
        }
        
        //last value is the ideal output neuron(will get set to an expected value of 1)
        int outputNeuronIndex = randomLine[randomLine.size() - 1];
        expectedOutput[outputNeuronIndex] = 1.0;

        neuralNet.forwardPass(trainingData);

        neuralNet.backPropagate(expectedOutput);

        //hold for text input every 20 steps
        if(neuralNet.step % 5 == 0){
            std::string tmp;

            std::cout << processedLines.size();
            std:: cin >> tmp;
        }
    }


    
}
int main(){
    std::string empty;

    hardTest();

    std::getline(std::cin, empty);
    ///std::cout << "Feed-forward output: " << output << std::endl;
    return 0;
}