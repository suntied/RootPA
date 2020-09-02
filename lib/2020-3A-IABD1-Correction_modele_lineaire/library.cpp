#include <random>
#include "library.h"


extern "C" {
DLLEXPORT LinearModel *linear_create_model(int nb_features) {
    return new LinearModel(nb_features);
}

DLLEXPORT void linear_dispose_model(LinearModel *model) {
    delete model;
}

DLLEXPORT double linear_predict_regression(LinearModel *model, const double *inputs, int inputs_size) {
    return model->PredictRegression(inputs, inputs_size);
}

DLLEXPORT double linear_predict_classification(LinearModel *model, const double *inputs, int inputs_size) {
    return model->PredictClassification(inputs, inputs_size);
}

DLLEXPORT void linear_train_regression(
        LinearModel *model,
        double *dataset_inputs,
        double *dataset_expected_outputs,
        int dataset_samples_count,
        int dataset_sample_features_count
) {
    model->TrainRegression(dataset_inputs, dataset_expected_outputs, dataset_samples_count,
                           dataset_sample_features_count);
}

DLLEXPORT void linear_train_classification(
        LinearModel *model,
        const double *dataset_inputs,
        const double *dataset_expected_outputs,
        int dataset_samples_count,
        int dataset_sample_features_count,
        double alpha, // learning rate
        int iteration_count
) {
    model->TrainClassification(dataset_inputs, dataset_expected_outputs, dataset_samples_count,
                               dataset_sample_features_count, alpha, iteration_count);
}

}