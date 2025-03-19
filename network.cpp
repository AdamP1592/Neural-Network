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
        learningRate = 1/std::exp(0.01 * step);
    }

    //set activationValue for the input layer. Iterate all subsequent layers as a standard pass
    void forwardPass(std::vector<double>& inputValues){
        Logger::log("forwardPass:\n");

        Logger::log("Learning Rate: ");
        Logger::log(std::to_string(learningRate));
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
                thisLayer.layer[j].activate();
                Neuron& n = thisLayer.layer[j];

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
        std::string expectedValuesString;
        Logger::log("Expected");
        for(int i = 0; i < expectedValues.size(); i++){
            expectedValuesString += " " + std::to_string(expectedValues[i]);
        }
        Logger::log(expectedValuesString + "\n");

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

                //logging with string builder
                std::ostringstream oss;
                oss << "Neuron: (" << i  << ", " << j << ") backProp Error:" << err;
                Logger::log(oss.str());
            }

        }
        step++;
        updateLearningRate();

    }

    void printNetworkDetailed() {
        std::cout << "Neural Network Visualization:\n";
        int numLayers = layers.size();
        for (int l = 0; l < numLayers; l++){
            std::cout << "Layer " << l << " (" << layers[l].size << " neurons):\n";
            for (int n = 0; n < layers[l].size; n++){
                Neuron& neuron = layers[l].layer[n];
                // Format the numbers with fixed precision
                std::cout << "  Neuron " << std::setw(2) << n 
                        << " | Activation: " << std::fixed << std::setprecision(4) << neuron.activationValue 
                        << " | Error: " << std::fixed << std::setprecision(4) << neuron.delta
                        << "\n";
            }
            std::cout << std::endl;
        }
    }

    void printExpectedOutputs(const std::vector<double>& expected) {
        std::cout << "Expected Outputs:\n";
        for (size_t i = 0; i < expected.size(); i++){
            std::cout << "  Output " << std::setw(2) << i 
                      << " | Value: " << std::fixed << std::setprecision(4) << expected[i] 
                      << "\n";
        }
        std::cout << std::endl;
    }
    void printNetwork(){
        printNetworkDetailed();
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

    for(int i = 0; i < 10; i++){
        neuralNetwork.printNetwork();

        neuralNetwork.forwardPass(inputs);

        neuralNetwork.backPropagate(expected);
    }

}

std::vector<double> getLine(std::vector<std::vector<double>> &lines){
    std::vector<double> randomLine;
    if(!lines.empty()){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, lines.size() - 2);

        
        // Get a random index.
        int randomIndex = dist(gen);
        std::string randomLineString = "Random Line" + std::to_string(randomIndex) + "\n" + "Num Lines Left: " + std::to_string(lines.size());
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
double max(double val1, double val2){
    if(val1 > val2){
        return val1;
    }
    return val2;
}
double min(double val1, double val2){
    if(val1 < val2){
        return val1;
    }
    return val2;
}
std::vector<std::vector<double>> processData(std::string filepath){
    //declared prior to catch case so there can still be a return value
    std::vector<std::vector<double>> processedLines;

    std::ifstream inFile(filepath);  // Adjust the file name as needed
    if (!inFile) {
        std::cerr << "Error: Could not open file.\n";
        return processedLines;
    }
    std::vector<std::string> lines;
    std::vector<double> processedLine;

    std::vector<std::vector<double>> normalizedData;

    std::string line;
    std::vector<double> maxes;
    std::vector<double> mins;
    while(std::getline(inFile, line)){
        lines.push_back(line);
        processedLine = processDataPoint(line);
        
        for(int i = 0; i < processedLine.size();i++){

            if(i >= maxes.size()){
                maxes.push_back(processedLine[i]);
                mins.push_back(processedLine[i]);
            }
            maxes[i] = max(maxes[i], processedLine[i]);
            mins[i] = min(mins[i], processedLine[i]);
            
            std::cout << processedLine[i] << std::endl;
        }
        processedLines.push_back(processedLine);
    }
    for(int i = 0; i < processedLines.size(); i++){
        //each line is [bunch of values, expectedOutputIndex]
        std::vector<double> normalizedLine;
        for(int j = 0; j < processedLines[i].size(); j++){
            //each value in line
            double denom = (maxes[j] - mins[j]);
            double normalizedValue;
            if(denom != 0.0){
                normalizedValue = (processedLines[i][j] - mins[j]) / (maxes[j] - mins[j]);
            }else{
                normalizedValue = 0;
            }
            normalizedLine.push_back(normalizedValue);
        }
        normalizedData.push_back(normalizedLine);
    }
    
    inFile.close();
    return normalizedData;
}

void hold() {
    std::cout << "Press Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}
void hardTest(){
    std::string dataPath = "./data/iris.data";
    std::vector<std::vector<double>> normalizedData = processData(dataPath);

    int startSize = normalizedData.size();
    network neuralNet;
    //data points are 5 points, 4 of them are inputs 1 is expected
    //there are 3 types of expected values
    std::vector<int> structure = {4, 5, 1};

    neuralNet.setupNetwork(structure);

    //training
    Logger::log("Training");
    while(normalizedData.size() > startSize/4){

        //std::string numLines = "Number of lines: "  + std::to_string(normalizedData.size());
        //Logger::log(numLines); **DEBUGGING TOOL**

        std::vector<double> randomLine = getLine(normalizedData);

        std::vector<double> expectedOutput;
        std::vector<double> trainingData;
        //for logging std::string lineData;


        for(int i = 0; i < randomLine.size() - 1; i++){
            trainingData.push_back(randomLine[i]);
            //for logging lineData+= std::to_string(randomLine[i]);
        }
        expectedOutput.push_back(randomLine[randomLine.size() - 1]);
        neuralNet.forwardPass(trainingData);

        neuralNet.backPropagate(expectedOutput);

        //hold for text input every 20 steps **DEBUGGING TOOL**
        /*if(neuralNet.step % 20 == 0){
            std::string tmp;

            std::cout << normalizedData.size();
            std:: cin >> tmp;
        }*/
    }

    //testing

    
    for(int i = 0; i < normalizedData.size()/2; i++){
        system("clear");
        std::vector<double> randomLine = getLine(normalizedData);

        std::vector<double> expectedOutput;
        std::vector<double> trainingData;
        //for logging std::string lineData;


        for(int i = 0; i < randomLine.size() - 1; i++){
            trainingData.push_back(randomLine[i]);
            //for logging lineData+= std::to_string(randomLine[i]);
        }
        expectedOutput.push_back(randomLine[randomLine.size() - 1]);
        neuralNet.forwardPass(trainingData);
        neuralNet.backPropagate(expectedOutput);

        neuralNet.printNetworkDetailed();
        neuralNet.printExpectedOutputs(expectedOutput);

        hold();
    }
    Logger::log(std::to_string(normalizedData.size()));

    hold();
}
int main(){
    //simpleTest(); //for testing basic functionality with fixed data
    hardTest(); //for testing more complicated functionality with variable data

    //hold();
    ///std::cout << "Feed-forward output: " << output << std::endl;
    return 0;
}