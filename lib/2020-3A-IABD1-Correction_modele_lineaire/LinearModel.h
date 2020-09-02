//
// Created by vidal on 6/11/2020.
//

#ifndef INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LINEARMODEL_H
#define INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LINEARMODEL_H


class LinearModel {
    double *w;

public:
    explicit LinearModel(int nb_features);

    ~LinearModel();

    double PredictRegression(const double *inputs, int inputs_size);

    double PredictClassification(const double *inputs, int inputs_size);

    void TrainRegression(
            double *dataset_inputs,
            double *dataset_expected_outputs,
            int dataset_samples_count,
            int dataset_sample_features_count
    );

    void TrainClassification(
            const double *dataset_inputs,
            const double *dataset_expected_outputs,
            int dataset_samples_count,
            int dataset_sample_features_count,
            double alpha, // learning rate
            int iteration_count
    );
};


#endif //INC_2020_3A_IABD1_CORRECTION_MODELE_LINEAIRE_LINEARMODEL_H
