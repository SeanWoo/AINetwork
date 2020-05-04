#pragma once

#define WIN32_LEAN_AND_MEAN             // Исключите редко используемые компоненты из заголовков Windows

#if defined(AINETWORK_EXPORTS)
#define FUNCTION_EXPORT __declspec(dllexport)
#else
#define FUNCTION_EXPORT
#endif

// Файлы заголовков Windows
#include <windows.h>


void InitWeights(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize);
void InitOutputs(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize);
void InitErrors(int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize);
void FeedForward(int layer, int neuron, double* inputs);
void Learn(int layer, int neuron, double error);
double Activation(double x);
double ActivationDx(double x);

EXTERN_C FUNCTION_EXPORT bool IsInitialize();
EXTERN_C FUNCTION_EXPORT bool CountLayers();
EXTERN_C FUNCTION_EXPORT bool CountNeurons(int IDLayer);

EXTERN_C FUNCTION_EXPORT bool Initialize(double learningRate, int countInputNeurons, int countOutputNeurons, int* hiddenLayers, int hiddenLayersSize);
EXTERN_C FUNCTION_EXPORT double Forward(double* inputs, int &index);
EXTERN_C FUNCTION_EXPORT double BackPropagation(double* excepted, double* inputs);
EXTERN_C FUNCTION_EXPORT double GetWeight(int layer, int neuron, int weight);
EXTERN_C FUNCTION_EXPORT double GetOutput(int layer, int neuron);
EXTERN_C FUNCTION_EXPORT double GetError(int layer, int neuron);
EXTERN_C FUNCTION_EXPORT void SetWeight(int layer, int neuron, int weight, double value);
EXTERN_C FUNCTION_EXPORT bool Dispose();
