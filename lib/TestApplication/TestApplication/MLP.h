#ifndef UNTITLED2_MLP_H
#define UNTITLED2_MLP_H

#include "vector"
#include "Layer.h"

using namespace std;

class MLP {
    vector<int> npl;
    int L;
    int inputSize;
    vector<Layer*> layers;

public:
    MLP(vector<int> npl);


    int getInputSize();
    int getL();
    vector<Layer*> getLayers();
    vector<int> getNpl();
};


#endif //UNTITLED2_MLP_H
