#include <iostream>
#include <fstream>
#include <cmath>
#include "LogisticRegressor.h"
#include "RegressionExceptions.h"
using namespace std;

inline double sigmoid(double z)
{
    return 1/(1+exp(-z));
}

//loss for the logistic regression
//y is the target
//y_hat is the prediction
inline double log_loss(double y, double y_hat)
{
    return -y*log(y_hat) - (1-y)*log(1-y_hat);
}

vector <double> LogisticRegressionClassificator::train(vector <vector<double>> X, vector <double> y,
                                                       double learning_rate, double epoch, double batch_size)
{
    if (learning_rate <= 0 || epoch <= 0 || batch_size <= 0)
    {
        cout << "One of the hyperparameters is negative! Aborting..." << endl;
        exit(EXIT_FAILURE);
    }
    vector <double> losses, sample;
    double loss, prediction, target;
    unsigned dataset_size = y.size();
    //initialize coefficients
    b1 = (double) rand() / RAND_MAX;
    b2 = (double) rand() / RAND_MAX;
    b3 = (double) rand() / RAND_MAX;
    for (int i = 0; i < epoch; i++) {
        loss = 0;
        //# of iterations in an epoch = # of batches = dataset_size/batch_size
        for (int j = 0; j < dataset_size / batch_size; j++)
        {
            //start accumulating gradients for each new batch
            double b1_grad = 0, b2_grad = 0, b3_grad = 0;
            for (int k = 0; k < batch_size; k++)
            {
                sample = X[k + j * batch_size];
                target = y[k + j * batch_size];
                prediction = sigmoid(b1 * sample[0] + b2 * sample[1] + b3);
                //calculate loss for a given prediction
                loss += log_loss(target, prediction);
                //calculate the gradients for a given prediction and add it to the gradient for the given batch
                b1_grad += (prediction - target) * sample[0];
                b2_grad += (prediction - target) * sample[1];
                b3_grad += (prediction - target);
            }
            //update the coefficients - gradient descent
            b1 -= learning_rate * b1_grad;
            b2 -= learning_rate * b2_grad;
            b3 -= learning_rate * b3_grad;
        }
        cout << "Loss for epoch " << i + 1 << ": " << loss << endl;
        //when iterated through ALL the elements, append the accumulated loss
        losses.push_back(loss);
    }
    return losses;
}

double LogisticRegressionClassificator::predict_value(vector<double> X) const
{
    double sigma_value = sigmoid(b1*X[0] + b2*X[1] + b3);
    if (sigma_value < 0.5)
        return 0;
    else
        return 1;
}

vector <double> LogisticRegressionClassificator::predict_vector(vector <vector <double>> X) const
{
    vector<double> predictions;
    for (int i = 0; i < X.size(); i++)
        predictions.push_back(predict_value(X[i]));
    return predictions;
}

vector <double> LogisticRegressionClassificator::get_coefficients() const
{
    return vector<double> {b1, b2, b3};
}

void LogisticRegressionClassificator::save_model() const
{
    ofstream model_file;
    model_file.open("saved_logistic_regression_classificator.txt");
    model_file << "b1: " << b1 << endl << "b2: " << b2 << endl << "b3: " << b3;
    model_file.close();
}

vector <double> LogisticRegressionClassificator::get_model_from_file(string filename)
{
    vector <double> coefficients;
    ifstream model_file;
    model_file.open(filename);
    if(!model_file)
        throw FileOpenException();
    string coeff; double value;
    while( model_file >> coeff >>  value)
        coefficients.push_back(value);
    model_file.close();
    if (!model_file && !model_file.eof())
        throw FileReadException();
    b1 = coefficients[0];
    b2 = coefficients[1];
    b3 = coefficients[2];

    return coefficients;
}
