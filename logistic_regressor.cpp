#include <iostream>
#include <fstream>
#include <cmath>
#include "logistic_regressor.h"
using namespace std;

inline double sigma(double z)
{
    return 1/(1+exp(-z));
}

inline double log_loss(double y, double y_hat)
{
    return -y*log(y_hat) - (1-y)*log(1-y_hat);
}

vector <double> LogisticRegressor::train(vector <vector<double>> X, vector <double> y,
                           double learning_rate, unsigned epoch, unsigned batch_size)
{
    vector <double> losses, sample;
    double loss, prediction, target;
    unsigned dataset_size = y.size();
    b1 = (double) rand() / RAND_MAX;
    b2 = (double) rand() / RAND_MAX;
    b3 = (double) rand() / RAND_MAX;
    for (int i = 0; i < epoch; i++) {
        loss = 0;
        for (int j = 0; j < dataset_size / batch_size; j++) {
            double b1_grad = 0, b2_grad = 0, b3_grad = 0;
            for (int k = 0; k < batch_size; k++) {
                sample = X[k + j * batch_size];
                target = y[k + j * batch_size];
                prediction = predict_value(sample);
                loss += log_loss(target, prediction);
                b1_grad += (prediction - target) * sample[0];
                b2_grad += (prediction - target) * sample[1];
                b3_grad += (prediction - target);
            }
            b1 -= learning_rate * b1_grad;
            b2 -= learning_rate * b2_grad;
            b3 -= learning_rate * b3_grad;
        }
        cout << "Loss for epoch " << i + 1 << ": " << loss << endl;
        losses.push_back(loss);
    }
    return losses;
}

double LogisticRegressor::predict_value(vector<double> X) const
{
    return sigma(b1*X[0] + b2*X[1] + b3);
}

vector <double> LogisticRegressor::predict_vector(vector <vector <double>> X) const
{
    vector<double> predictions;
    for (int i = 0; i < X.size(); i++)
        predictions.push_back(sigma(b1*X[i][0] + b2*X[i][1] + b3));
    return predictions;
}

vector <double> LogisticRegressor::get_coefficients() const
{
    return vector<double> {b1, b2, b3};
}

void LogisticRegressor::save_model() const
{
    ofstream model_file;
    model_file.open("saved_logistic_regressor.txt");
    model_file << "b1: " << b1 << endl << "b2: " << b2 << endl << "b3: " << b3;
    model_file.close();
}
