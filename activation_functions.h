#ifndef ACTIVATION_FUNCS
#define ACTIVATION_FUNCS

struct ActivationResult {
    double activatedValue;
    double derivative;
};
ActivationResult relu(double value);

ActivationResult leakyRelu(double value);


#endif