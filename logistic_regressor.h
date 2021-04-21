#ifndef HW_LOGISTIC_REGRESSOR_H
#define HW_LOGISTIC_REGRESSOR_H

#include "regressor.h"
#include <vector>

class LogisticRegressor
{
    double b1;
    double b2;
    double b3;
public:
    LogisticRegressor(): b1(0), b2(0), b3(0) {}
    std::vector <double> train(std::vector <std::vector<double>> X, std::vector <double> y,
                               double learning_rate, unsigned epoch, unsigned batch_size);
    double predict_value(std::vector<double> X) const;
    std::vector <double> predict_vector(std::vector <std::vector <double>> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
};

#endif //HW_LOGISTIC_REGRESSOR_H
