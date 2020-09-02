#include "MlpService.h"
#include "MLP.h"
MlpService::MlpService()
{
}
double* MlpService::predict_MLP_Classification(MLP* mlp, double* inputs) {
    return MlpService::predict_MLP_InCommon(mlp, inputs, true);
}
double* MlpService::predict_MLP_InCommon(MLP* mlp, double* inputs, bool isClassif) {
    Layer* first = mlp->getLayers().front();

    for (int i = 1; i < mlp->getInputSize() + 1; i++) {
        first->x[i] = inputs[i - 1];
    }

    for (int l = 1; l < mlp->getL() + 1; l++) {
        Layer* layer = mlp->getLayers()[l];
        Layer* previous = mlp->getLayers()[l - 1];

        for (int j = 1; j < mlp->getNpl()[l] + 1; j++) {
            double result = 0.0;
            for (int k = 0; k < mlp->getNpl()[l - 1] + 1; k++) {
                result += layer->weights[k][j] * previous->x[k];
            }

            if (l != mlp->getL() || isClassif) {
                result = tanh(result);
            }

            layer->x[j] = result;
        }
    }

    double* predictions = mlp->getLayers().back()->x.data() + 1;
    return predictions;
}


void MlpService::mlp_train_classification(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha) {
    _mlp_train_common(mlp, inputs, labels, size, iteration, alpha, true);
}
void MlpService::mlp_train_Regression(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha) {
    _mlp_train_common(mlp, inputs, labels, size, iteration, alpha, false);
}
void MlpService::_mlp_train_common(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha, bool isClassif) {
    for (int _ = 0; _ < iteration; _++) {
        for (int it = 0; it < size; it++) {
            predict_MLP_InCommon(mlp, inputs + (mlp->getInputSize() * it), isClassif);

            for (int j = 1; j < mlp->getNpl()[mlp->getL()] + 1; j++) {
                mlp->getLayers().back()->deltas[j] = mlp->getLayers().back()->x[j] - labels[it];

                if (isClassif) {
                    mlp->getLayers().back()->deltas[j] *= (1 - mlp->getLayers().back()->x[j] * mlp->getLayers().back()->x[j]);
                }
            }

            for (int l = mlp->getL(); l > 1; l--) {
                for (int i = 1; i < mlp->getNpl()[l - 1] + 1; i++) {
                    double result = 0.0;

                    for (int j = 1; j < mlp->getNpl()[l] + 1; j++) {
                        result += mlp->getLayers()[l]->weights[i][j] * mlp->getLayers()[l]->deltas[j];
                    }

                    result *= 1 - (mlp->getLayers()[l - 1]->x[i] * mlp->getLayers()[l - 1]->x[i]);
                    mlp->getLayers()[l - 1]->deltas[i] = result;
                }
            }

            for (int l = 1; l < mlp->getL() + 1; l++) {
                for (int i = 0; i < mlp->getNpl()[l - 1] + 1; i++) {
                    for (int j = 0; j < mlp->getNpl()[l] + 1; j++) {
                        mlp->getLayers()[l]->weights[i][j] -= alpha * mlp->getLayers()[l - 1]->x[i] * mlp->getLayers()[l]->deltas[j];
                    }
                }
            }

        }
    }
}


