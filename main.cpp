#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include "linear_regressor.h"
#include "polynomial_regressor.h"
#include "logistic_regressor.h"
#include <sciplot/sciplot.hpp>
using namespace std;
using namespace sciplot;

vector<vector<double>> getVectorFromFile(string file_x, string file_y);

int main() {
    /*
    //Test 3
    vector<vector <double>> X = {{2.7810836, 2.550537003}, {1.465489372, 2.362125076},
                                 {3.396561688, 4.400293529}, {1.38807019,1.850220317},
                                 {3.06407232,3.005305973}, {7.627531214, 2.759262235},
                                 {5.332441248, 2.088626775}, {6.922596716,1.77106367},
                                 {8.675418651,-0.2420686549}, {7.673756466, 3.508563011}};
    vector<double> y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    LogisticRegressor model_class;
    model_class.train(X, y, 0.001, 2000, 5);
    */
    //Test 1
    /*
    vector<vector <double>> dataset = getVectorFromFile("data.txt", "targets.txt");

    LinearRegressor model1;
    model1.train(dataset[0], dataset[1], 0.002, 250, 5);
    cout << "Model prediction: " << model1.predict_value(5.5) << endl;

    vector <double> z = {2.3, 7.7, 20};
    vector <double> z_pred = model1.predict_vector(z);
    for (int i = 0; i < z.size(); i++)
        cout << z_pred[i] << endl;*/

    //Test 2
    vector <double> X_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector <double> y_2 = {1, 1.3, 1.7, 2.3, 2.5, 3.1, 4.3, 5.5, 7.5, 10};

    Regressor* models[2];
    vector <vector <double>> losses;

    models[0] = new LinearRegressor;
    losses[0] = models[0]->train(X_2, y_2, 0.002, 500, 5);

    models[1] = new PolynomialRegressor;
    losses[1] = models[1]->train(X_2, y_2, 0.0001, 2500, 5);

    for (int i = 0; i < 2; i++) {
        cout << "Prediction of model " << i << ": " << models[i]->predict_value(5.5) << endl;
        models[i]->save_model();
    }

    Plot plot, plot2;
    plot.xlabel("X");
    plot.ylabel("y");

    plot.xrange(0.0, 10.0);
    plot.yrange(0.0, 10.0);
    plot.legend()
            .atOutsideBottom()
            .displayHorizontal();
    plot.drawCurve(X_2, y_2).label("Targets").lineWidth(3);
    plot.drawCurve(X_2, models[0]->predict_vector(X_2)).label("LinearR").lineWidth(3);
    plot.drawCurve(X_2, models[1]->predict_vector(X_2)).label("PolynomialR").lineWidth(3);

    plot.size(749, 600);
    plot.show();
    plot.save("eg.pdf");

    for (int i = 0; i < 2; i++)
        delete models[i];
    return 0;
}

vector<vector<double>> getVectorFromFile(string file_x, string file_y)
{
    vector<vector<double>> dataset(2);
    ifstream data_file, target_file;

    data_file.open(file_x);
    if (!data_file.is_open())
    {
        cout << "Failed to open the file!" << endl;
        //raise an exception
        return {{}, {}};
    }
    double x_data;
    while(data_file >> x_data)
        dataset[0].push_back(x_data);
    data_file.close();

    target_file.open(file_y);
    if (!target_file.is_open())
    {
        cout << "Failed to open the file!" << endl;
        //raise an exception
        return {{}, {}};
    }
    double y_data;
    while(target_file >> y_data)
        dataset[1].push_back(y_data);
    target_file.close();

    return dataset;
}