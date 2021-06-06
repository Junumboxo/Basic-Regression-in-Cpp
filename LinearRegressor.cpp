#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include "LinearRegressor.h"
#include "RegressionExceptions.h"
using namespace std;

vector <double> LinearRegressor::train(vector <double> X, vector <double> y,
                                       double learning_rate, double epoch, double batch_size)
{
    if (learning_rate <= 0 || epoch <= 0 || batch_size <= 0)
    {
        cout << "One of the hyperparameters is negative! Aborting..." << endl;
        abort();
    }
    vector <double> losses = {};
    double loss, sample, prediction, target;
    unsigned dataset_size = y.size();
    //initialize coefficients
    W = (double) rand() / RAND_MAX;
    b = (double) rand() / RAND_MAX;
    for (int i = 0; i < epoch; i++)
    {
        loss = 0;
        //# of iterations in an epoch = # of batches = dataset_size/batch_size
        for (int j = 0; j < dataset_size / batch_size; j++)
        {
            //start accumulating gradients for each new batch
            double W_grad = 0, b_grad = 0;
            for (int k = 0; k < batch_size; k++)
            {
                sample = X[k + j*batch_size];
                target = y[k + j*batch_size];
                prediction = predict_value(sample);
                //calculate loss for a given prediction
                loss += pow(target - prediction, 2);
                //calculate the gradients for a given prediction and add it to the gradient for the given batch
                W_grad += 2*(prediction - target) * sample;
                b_grad += 2*(prediction - target);
            }
            //update the coefficients - gradient descent
            //introduce the division by the batch size necessary by the formula
            W -= W_grad * learning_rate/batch_size;
            b -= b_grad * learning_rate/batch_size;
        }
        cout << "Loss for epoch " << i + 1 << ": " << loss << endl;
        //when iterated through ALL the elements, append the accumulated loss
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

void LinearRegressor::save_model() const {
    ofstream model_file;
    model_file.open("saved_linear_regressor.txt");
    model_file << "Slope: " << W << endl << "Intercept: " << b;
    model_file.close();
}

vector <double> LinearRegressor::get_model_from_file(string filename) {
    vector<double> coefficients;
    ifstream model_file;
    model_file.open(filename);
    if (!model_file)
        throw FileOpenException();
    string coeff;
    double value;
    while (model_file >> coeff >> value) {
        coefficients.push_back(value);
    }
    if (!model_file && !model_file.eof())
        throw FileReadException();
    model_file.close();
    W = coefficients[0];
    b = coefficients[1];

    return coefficients;
}