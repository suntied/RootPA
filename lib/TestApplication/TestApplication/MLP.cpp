#include "MLP.h"
#include "random"

MLP::MLP(vector<int> npl) {
    this->npl = npl;
    this->inputSize = npl.front();
    this->L = npl.size() - 1;

    random_device rd;
    mt19937 e2 = mt19937(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);


    layers.push_back(new Layer()); 

    for (int i = 1; i < npl.size(); i++) {
        Layer* layer = new Layer();

        for (int j = 0; j < npl[i - 1] + 1; j++) {
            layer->weights.push_back(vector<double>());

            for (int k = 0; k < npl[i] + 1; k++) {
                layer->weights[j].push_back(dist(e2));
            }
        }

        layers.push_back(layer);
    }

    for (int i = 0; i < npl.size(); i++) {
        Layer* layer = layers[i];
        for (int j = 0; j < npl[i] + 1; j++) {
            layer->deltas.push_back(0.0);
        }
    }
    //all layers have 1 biais initialize to 0
    for (int i = 0; i < npl.size(); i++) {
        Layer* layer = layers[i];
        for (int j = 0; j < npl[i] + 1; j++) {
            layer->x.push_back(j == 0 ? 1.0 : 0.0);
        }
    }
}

int MLP::getInputSize()
{
    return this->inputSize;
}
int MLP::getL()
{
    return this->L;
}
vector<Layer*> MLP::getLayers()
{
    return this->layers;
}
vector<int> MLP::getNpl()
{
    return this->npl;
}

