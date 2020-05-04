// dllmain.cpp : Определяет точку входа для приложения DLL.
#include "pch.h"
#include "string"
#include <math.h>
#include <time.h>

using namespace std;

BOOL APIENTRY DllMain( HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

/////////////////////////////////////////////////////////////////////////////////////////
//Переменные
/////////////////////////////////////////////////////////////////////////////////////////

int countLayer;
double learnRate;
double* sizes;
double*** weights;
double** outputs;
double** errors;

/////////////////////////////////////////////////////////////////////////////////////////
//Function Export
/////////////////////////////////////////////////////////////////////////////////////////
bool IsInitialize() 
{
    return countLayer > 0 ? true : false;
}
bool CountLayers() 
{
    return countLayer;
}
bool CountNeurons(int IDLayer) 
{
    int result = 0;
    for (int i = 0; i < countLayer; i++)
    {
        result += sizes[i];
    }
    return result;
}
double GetWeight(int layer, int neuron, int weight)
{
    return weights[layer][neuron][weight];
}
double GetOutput(int layer, int neuron)
{
    return outputs[layer][neuron];
}
double GetError(int layer, int neuron)
{
    return errors[layer][neuron];
}
void SetWeight(int layer, int neuron, int weight, double value)
{
    weights[layer][neuron][weight] = value;
}

//Initialization AI
bool Initialize(double learningRate, int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize) 
{
    //weight[LAYER][NEURON][WEIGHT]

    //Init random seed
    srand(time(0));

    learnRate = learningRate;
    countLayer = 1 + hiddenLayersSize + 1; //Count layers
    sizes = new double[countLayer];
    sizes[0] = countInputNeurons;
    sizes[countLayer - 1] = countOutputNeurons;
    for (int i = 0; i < hiddenLayersSize; i++)
    {
        sizes[i + 1] = hiddenLayers[i];
    }

    InitWeights(countInputNeurons, countOutputNeurons, hiddenLayers, hiddenLayersSize);
    InitOutputs(countInputNeurons, countOutputNeurons, hiddenLayers, hiddenLayersSize);
    InitErrors(countInputNeurons, countOutputNeurons, hiddenLayers, hiddenLayersSize);

    return true;
}
//Start
double Forward(double* inputs, int &index) 
{
    for (int i = 0; i < sizes[0]; i++)//Send signals to input neurons
    {
        outputs[0][i] = inputs[i];
    }
    for (int i = 1; i < countLayer; i++)//Send signals to hidden neurons
    {
        double* nextInputs = new double[sizes[i - 1]];
        for (int k = 0; k < sizes[i - 1]; k++)
        {
            nextInputs[k] = outputs[i - 1][k];
        }
        for (int n = 0; n < sizes[i]; n++)
        {
            FeedForward(i, n, nextInputs);
        }
    }
    double result = 0;
    for (int i = 0; i < sizes[countLayer - 1]; i++)//Return result and index output neuron
    {
        if (outputs[countLayer - 1][i] > result) {
            index = i;
            result = outputs[countLayer - 1][i];
        }
    }
    return result;
}
//Learning
double BackPropagation(double excepted, double* inputs) 
{
    int index = 0;
    double actual = Forward(inputs, index);
    double difference = actual - excepted;

    for (int i = 0; i < sizes[countLayer - 1]; i++)
    {
        Learn(countLayer - 1, i, difference);
    }

    for (int i = countLayer - 2; i >= 0; i--)
    {
        for (int n = 0; n < sizes[i]; n++)
        {
            double error = 0;
            for (int k = 0; k < sizes[i+1]; k++)
            {
                double newError = weights[i + 1][k][n] * errors[i + 1][k];
                error += (newError * newError) / 2;
            }
            Learn(i, n, error);
        }
    }

    return difference * difference;
}
//Free memory
bool Dispose() 
{
    for (int i = 0; i < countLayer; i++)
    {
        for (int l = 0; l < sizes[i]; l++)
        {
            delete[] weights[i][l];
        }
        delete[] weights[i];
        delete[] outputs[i];
        delete[] errors[i];
    }
    delete[] weights;
    delete[] outputs;
    delete[] errors;
    delete[] sizes;
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
//Functions
/////////////////////////////////////////////////////////////////////////////////////////
void InitWeights(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize) 
{
    weights = new double** [countLayer]; //Create layers
    weights[0] = new double* [countInputNeurons]; //Create input neurons
    weights[countLayer - 1] = new double* [countOutputNeurons];//Create output neurons
    for (int i = 1; i <= hiddenLayersSize; i++)
    {
        weights[i] = new double* [hiddenLayers[i - 1]];//Create hidden neurons
    }
    for (int i = 0; i < countInputNeurons; i++)//Create input weights
    {
        weights[0][i] = new double[1]{ 1 };
    }
    for (int i = 1; i < countLayer - 1; i++)//Create other weights
    {
        for (int j = 0; j < hiddenLayers[i - 1]; j++)
        {
            if (i == 1) {
                weights[i][j] = new double[countInputNeurons];
                for (int k = 0; k < countInputNeurons; k++)
                {
                    weights[i][j][k] = (double)(rand()) / RAND_MAX;
                }
            }
            else
            {
                weights[i][j] = new double[hiddenLayers[i - 1]];
                for (int k = 0; k < hiddenLayers[i - 1]; k++)
                {
                    weights[i][j][k] = (double)(rand()) / RAND_MAX;
                }
            }
        }
    }
    for (int n = 0; n < countOutputNeurons; n++)//Create output weights
    {
        if (hiddenLayersSize == 0) {
            weights[countLayer - 1][n] = new double[countInputNeurons];
            for (int w = 0; w < countInputNeurons; w++)
            {
                weights[countLayer - 1][n][w] = (double)(rand()) / RAND_MAX;
            }
        }
        else 
        {
            weights[countLayer - 1][n] = new double[hiddenLayers[hiddenLayersSize - 1]];
            for (int w = 0; w < hiddenLayers[hiddenLayersSize - 1]; w++)
            {
                weights[countLayer - 1][n][w] = (double)(rand()) / RAND_MAX;
            }
        }
    }
}
void InitOutputs(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize) 
{
    outputs = new double* [countLayer];
    outputs[0] = new double[countInputNeurons];
    outputs[countLayer - 1] = new double[countOutputNeurons];

    for (int i = 1; i < countLayer - 1; i++)
    {
        outputs[i] = new double[hiddenLayers[i - 1]];
    }
}
void InitErrors(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize) 
{
    errors = new double* [countLayer];
    errors[0] = new double[countInputNeurons];
    errors[countLayer - 1] = new double[countOutputNeurons];

    for (int i = 1; i < countLayer - 1; i++)
    {
        errors[i] = new double[hiddenLayers[i - 1]];
    }
}
void FeedForward(int layer, int neuron, double* inputs) 
{
    double sum = 0;
    for (int i = 0; i < sizes[layer-1]; i++)
    {
        sum += inputs[i] * weights[layer][neuron][i]; //Сounting output
    }
    outputs[layer][neuron] = Activation(sum);
}
void Learn(int layer, int neuron, double error)
{
    errors[layer][neuron] = error;
    double delta = error * ActivationDx(outputs[layer][neuron]);

    for (int i = 0; i < sizes[layer - 1]; i++)
    {
        weights[layer][neuron][i] = weights[layer][neuron][i] - outputs[layer - 1][i] * delta * learnRate; //Сounting new weights
    }
}
double Activation(double x)
{
    if (x < 0) {
        return 0.001;
    }
    if (x > 1) {
        return 1 + 0.001 * (x - 1);
    }
    return x;

    //return 1 / (1 + exp(-x));
}
double ActivationDx(double x)
{
    if (x < 0 || x > 1) {
        return 0.001;
    }
    return 1.0;
    //return Activation(x) * (1 - Activation(x));
}