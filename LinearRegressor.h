#ifndef HW_LINEARREGRESSOR_H
#define HW_LINEARREGRESSOR_H

#include "regressor.h"
#include <vector>
#include <string>

class LinearRegressor: public Regressor
{
    double W; //slope
    double b; //intercept
public:
    LinearRegressor(): W(0), b(0) {}
    std::vector <double> train(std::vector <double> X, std::vector <double> y,
                               double learning_rate = 0.005, double epoch = 500, double batch_size = 5);
    double predict_value(double X) const;
    std::vector <double> predict_vector(std::vector <double> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
    std::vector <double> get_model_from_file(std::string filename);
};

#endif //HW_LINEARREGRESSOR_H
