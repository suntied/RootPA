#pragma once
#include "MLP.h"
class MlpService
{
public:
	MlpService();
	double* predict_MLP_Classification(MLP* mlp, double* inputs);

	double* predict_MLP_InCommon(MLP* mlp, double* inputs, bool isClassif);

	void mlp_train_classification(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha);

	void mlp_train_Regression(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha);

	void _mlp_train_common(MLP* mlp, double* inputs, double* labels, int size, int iteration, double alpha, bool isClassif);
};


