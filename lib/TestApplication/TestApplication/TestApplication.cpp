#include <iostream>


#include "vector"
#include "MLP.h"
#include "MlpService.h"

using namespace std;

int main() {

    double inputs[8] = {
            0, 0,
            1, 0,
            0, 1,
            1, 1
    };

    double labels[4] = { -1, 1, 1, -1 };

    MLP* mlp = new MLP(vector<int>{2, 2, 1});
    MlpService* s = new MlpService();
    s->mlp_train_classification(mlp,inputs, labels, 4, 10000, 0.1);

    double* prediction = s->predict_MLP_Classification(mlp,new double[2]{ 0, 0 });
    cout << prediction[0] << endl;

    prediction = s->predict_MLP_Classification(mlp, new double[2]{ 1, 0 });
    cout << prediction[0] << endl;

    prediction = s->predict_MLP_Classification(mlp, new double[2]{ 0, 1 });
    cout << prediction[0] << endl;

    prediction = s->predict_MLP_Classification(mlp, new double[2]{ 1, 1 });
    cout << prediction[0] << endl;


}
