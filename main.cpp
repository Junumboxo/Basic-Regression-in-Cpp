#include <iostream>
#include "DataFileHandler.h"
#include "LinearRegressor.h"
#include "PolynomialRegressor.h"
#include "LogisticRegressor.h"
#include <sciplot/sciplot.hpp>
using namespace std;
using namespace sciplot;

int main() {
    srand(42); //a lucky number in Data Science:)
    int test_case;
    cout << "Choose a test case to run (1, 2 or 3). Enter 0 to exit" << endl;
    cin >> test_case;
    while (test_case != 1 && test_case != 2 && test_case != 3) {
        cout << "Invalid test case! Please, enter a number again" << endl;
        cin >> test_case;
    }
    while (test_case != 0)
    {
        switch (test_case) {
            //test 1 - linear regression, testing the main functionalities
            case 1:
            {
                vector<vector <double>> dataset = DataFileHandler::getVectorFromFile("data.txt", "targets.txt");

                LinearRegressor model1;
                model1.train(dataset[0], dataset[1]);
                //Example of prediction
                cout << "Model's prediction for 5.5: " << model1.predict_value(5.5) << endl;

                //Validating the model with the data it has never seen
                vector <double> z = {2.3, 7.7, 20};
                vector <double> z_pred = model1.predict_vector(z);
                cout << "Predictions of the model1:" << endl;
                for (int i = 0; i < z.size(); i++)
                    cout << "For " << z[i] << ": "<< z_pred[i] << endl;
                model1.save_model();

                vector <double> coeff_of_model1 = model1.get_coefficients();
                cout << "Testing accessibility of the coefficients of model1. Predictions:" << endl;
                for (int i = 0; i < z.size(); i++)
                    cout << "For " << z[i] << ": "<< coeff_of_model1[0]*z[i] + coeff_of_model1[1] << endl;

                LinearRegressor model2;
                try {
                    model2.get_model_from_file("saved_linear_regressor.txt");
                    //the outputs should be exactly the same as the prediction of model1
                    //as model2 is the very same model, loaded from a file
                    vector<double> z_pred_2 = model2.predict_vector(z);
                    cout << "Testing the save_model() and get_model_from_file()." << endl;
                    cout << "Predictions of the model2:" << endl;
                    for (int i = 0; i < z.size(); i++)
                        cout << "For " << z[i] << ": "<< z_pred[i] << endl;
                }
                catch (exception& ex)
                {
                    cout << ex.what() << endl;
                }
                break;
            }
            //test 2 - linear vs. polynomial regression
            case 2: {
                vector<double> X_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                vector<double> y_2 = {1, 1.3, 1.7, 2.3, 2.5, 3.1, 4.3, 5.5, 7.5, 10};

                Regressor *models[2];
                vector<vector<double>> losses;

                models[0] = new LinearRegressor;
                losses.push_back(models[0]->train(X_2, y_2, 0.002, 500, 5));

                models[1] = new PolynomialRegressor;
                losses.push_back(models[1]->train(X_2, y_2, 0.0001, 2500, 5));

                for (int i = 0; i < 2; i++) {
                    cout << "Prediction of model " << i << " for 5.5: " << models[i]->predict_value(5.5) << endl;
                    models[i]->save_model();
                }

                Plot plot, plot2;
                //plot the original data and the model results
                plot.xlabel("X");
                plot.ylabel("y");

                plot.xrange(0.0, 10.0);
                plot.yrange(0.0, 10.0);
                plot.legend()
                        .atOutsideBottom()
                        .displayHorizontal();
                plot.drawCurve(X_2, y_2).label("The initial dataset").lineWidth(3);
                plot.drawCurve(X_2, models[0]->predict_vector(X_2)).label("Linear Regression model").lineWidth(3);
                plot.drawCurve(X_2, models[1]->predict_vector(X_2)).label("Polynomial Regression model").lineWidth(3);

                plot.size(749, 600);
                plot.show();
                plot.save("test2.pdf");

                //plot the losses of the trained PolynomialRegressor
                valarray<double> epochs = linspace(0.0, 2500, 2500); //the number of epochs
                plot2.size(1000, 600);
                plot2.xlabel("Epochs");
                plot2.ylabel("Losses of Polynomial Regressor");

                plot2.xrange(0.0, 2500);
                plot2.yrange(0.0, 20);
                plot2.legend()
                        .atOutsideBottom()
                        .displayHorizontal();
                plot2.drawCurve(epochs, losses[1]).label("MSE").lineWidth(3);
                plot2.show();
                plot2.save("losses_polyn.pdf");

                for (int i = 0; i < 2; i++)
                    delete models[i];
                break;
            }
            //test 3 - logistic regression for classification
            case 3: {
                vector<vector<double>> X = {{2.7810836,   2.550537003},
                                            {1.465489372, 2.362125076},
                                            {3.396561688, 4.400293529},
                                            {1.38807019,  1.850220317},
                                            {3.06407232,  3.005305973},
                                            {7.627531214, 2.759262235},
                                            {5.332441248, 2.088626775},
                                            {6.922596716, 1.77106367},
                                            {8.675418651, -0.2420686549},
                                            {7.673756466, 3.508563011}};
                vector<double> y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

                LogisticRegressionClassificator model_class;
                model_class.train(X, y, 0.005, 15000, 5);
                cout << "Prediction for {8,2}: "<< model_class.predict_value(vector<double>{8, 2}) << endl;
                cout << "Prediction for {2,4}: "<< model_class.predict_value(vector<double>{2, 4}) << endl;
                cout << "Prediction for {7,3}: "<< model_class.predict_value(vector<double>{7, 3}) << endl;
                cout << "Prediction for {6,1}: "<< model_class.predict_value(vector<double>{6, 1}) << endl;
                cout << "Prediction for {4,2}: "<< model_class.predict_value(vector<double>{4, 2}) << endl;
                model_class.save_model();
                break;
            }
    }
    cout << "Choose a new test case to run: ";
    cin >> test_case;
    }
    return 0;
}