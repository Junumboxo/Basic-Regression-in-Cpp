#include "DataFileHandler.h"
#include <iostream>
#include <fstream>
using namespace std;

vector<vector<double>> DataFileHandler::getVectorFromFile(string file_x, string file_y)
{
    vector<vector<double>> dataset(2);
    ifstream data_file, target_file;

    data_file.open(file_x);
    if (!data_file.is_open())
    {
        cout << "Failed to open the file!" << endl;
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
        return {{}, {}};
    }
    double y_data;
    while(target_file >> y_data)
        dataset[1].push_back(y_data);
    target_file.close();

    return dataset;
}
