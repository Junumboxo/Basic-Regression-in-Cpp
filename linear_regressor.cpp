#include <iostream>
#include <fstream>
#include <cmath>
#include "linear_regressor.h"
using namespace std;

vector <double> LinearRegressor::train(vector <double> X, vector <double> y,
                                       double learning_rate, unsigned epoch, unsigned batch_size)
{
    vector <double> losses = {};
    double loss, sample, prediction, target;
    unsigned dataset_size = y.size();
    W = (double) rand() / RAND_MAX;
    b = (double) rand() / RAND_MAX;
    for (int i = 0; i < epoch; i++)
    {
        loss = 0;
        for (int j = 0; j < dataset_size / batch_size; j++)
        {
            double W_grad = 0, b_grad = 0;
            for (int k = 0; k < batch_size; k++)
            {
                sample = X[k + j*batch_size];
                target = y[k + j*batch_size];
                prediction = predict_value(sample);
                loss += pow(target - prediction, 2);
                W_grad += 2*(prediction - target) * sample;
                b_grad += 2*(prediction - target);
            }
            W -= W_grad * learning_rate/batch_size;
            b -= b_grad * learning_rate/batch_size;
        }
        cout << "Loss for epoch " << i + 1 << ": " << loss << endl;
        losses.push_back(loss/dataset_size);
    }
    return losses;
}

double LinearRegressor::predict_value(double X) const
{
    return W*X + b;
}

vector <double> LinearRegressor::predict_vector(vector <double> X) const
{
    vector <double> y(X.size());
    for (int i = 0; i < X.size(); i++)
        y[i] = predict_value(X[i]);
    return y;
}

vector <double> LinearRegressor::get_coefficients() const
{
    return vector <double> {W, b};
}

//solve for name!!!?????
void LinearRegressor::save_model() const {
    ofstream model_file;
    model_file.open("saved_linear_regressor.txt");
    model_file << "Slope: " << W << endl << "Intercept: " << b;
    model_file.close();
}