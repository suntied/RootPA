//
// Created by vidal on 6/11/2020.
//

#include <random>
#include <Eigen>
#include "LinearModel.h"

LinearModel::LinearModel(int nb_features) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    w = new double[nb_features + 1];
    for (auto i = 0; i < nb_features + 1; i++) {
        w[i] = dist(mt);
    }
}

LinearModel::~LinearModel() {
    delete[]w;
}

double LinearModel::PredictRegression(const double *inputs, int inputs_size) {
    auto sum = w[0];
    for (auto i = 0; i < inputs_size; i++) {
        sum += w[i + 1] * inputs[i];
    }
    return sum;
}

double LinearModel::PredictClassification(const double *inputs, int inputs_size) {
    auto sum = PredictRegression(inputs, inputs_size);
    auto return_val = sum >= 0 ? 1.0 : -1.0;
    return return_val;
}

void LinearModel::TrainRegression(double *dataset_inputs, double *dataset_expected_outputs,
                                  int dataset_samples_count, int dataset_sample_features_count) {
    Eigen::MatrixXd mat_inputsTransposed(dataset_sample_features_count + 1, dataset_samples_count);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_inputs_without_bias(
            dataset_inputs,
            dataset_samples_count,
            dataset_sample_features_count);

    mat_inputsTransposed
            << Eigen::VectorXd::Ones(dataset_samples_count).transpose(), mat_inputs_without_bias.transpose();

    Eigen::MatrixXd mat_inputs = mat_inputsTransposed.transpose();

    Eigen::MatrixXd pseudoInverse;
    if (dataset_samples_count >= dataset_sample_features_count + 1)
        pseudoInverse = (mat_inputsTransposed * mat_inputs).inverse() * mat_inputsTransposed;
    else
        pseudoInverse = mat_inputsTransposed * (mat_inputs * mat_inputsTransposed).inverse();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_outputs(
            dataset_expected_outputs, dataset_samples_count, 1);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >>(w,
                                                                                        dataset_sample_features_count +
                                                                                        1, 1) =
            pseudoInverse * mat_outputs;
}

void
LinearModel::TrainClassification(const double *dataset_inputs, const double *dataset_expected_outputs,
                                 int dataset_samples_count, int dataset_sample_features_count,
                                 double alpha, int iteration_count) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, dataset_samples_count - 1);

    for (auto it = 0; it < iteration_count; it++) {
        auto k = dist(mt);
        auto inputs_k = dataset_inputs + k * dataset_sample_features_count;

        auto expected_output_k = dataset_expected_outputs[k];

        auto predicted_output_k = PredictClassification(inputs_k, dataset_sample_features_count);

        auto semi_grad = alpha * (expected_output_k - predicted_output_k);
        for (auto i = 0; i < dataset_sample_features_count; i++) {
            w[i + 1] += semi_grad * inputs_k[i];
        }
        w[0] += semi_grad * 1.0;
    }
}
