#ifndef INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LIBRARY_H
#define INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LIBRARY_H

#include "LinearModel.h"

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
DLLEXPORT LinearModel *linear_create_model(int nb_features);
DLLEXPORT void linear_dispose_model(LinearModel *model);
DLLEXPORT double linear_predict_regression(LinearModel *model, const double *inputs, int inputs_size);
DLLEXPORT double linear_predict_classification(LinearModel *model, const double *inputs, int inputs_size);
DLLEXPORT void linear_train_regression(
        LinearModel *model,
        double *dataset_inputs,
        double *dataset_expected_outputs,
        int dataset_samples_count,
        int dataset_sample_features_count
);
DLLEXPORT void linear_train_classification(
        LinearModel *model,
        const double *dataset_inputs,
        const double *dataset_expected_outputs,
        int dataset_samples_count,
        int dataset_sample_features_count,
        double alpha, // learning rate
        int iteration_count
);


}

#endif //INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LIBRARY_H
