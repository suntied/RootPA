#pragma once
#ifndef UNTITLED2_LAYER_H
#define UNTITLED2_LAYER_H


#include "vector"

using namespace std;

class Layer {

public:
    vector<vector<double>> weights;

    vector<double> deltas;

    vector<double> x;
};


#endif //UNTITLED2_LAYER_H
