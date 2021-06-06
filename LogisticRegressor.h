#ifndef HW_LOGISTICREGRESSOR_H
#define HW_LOGISTICREGRESSOR_H

#include "regressor.h"
#include <vector>
#include <string>

class LogisticRegressionClassificator
{
    double b1; //coefficient for x1
    double b2; //coefficient for x2
    double b3; //free parameter
public:
    LogisticRegressionClassificator(): b1(0), b2(0), b3(0) {}
    std::vector <double> train(std::vector <std::vector<double>> X, std::vector <double> y,
                               double learning_rate = 0.005, double epoch = 15000, double batch_size = 5);
    double predict_value(std::vector<double> X) const;
    std::vector <double> predict_vector(std::vector <std::vector <double>> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
    std::vector <double> get_model_from_file(std::string filename);
};

#endif //HW_LOGISTICREGRESSOR_H
