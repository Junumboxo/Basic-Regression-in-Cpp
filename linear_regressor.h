#ifndef HW_LINEAR_REGRESSOR_H
#define HW_LINEAR_REGRESSOR_H

#include "regressor.h"
#include <vector>

class LinearRegressor: public Regressor
{
    double W; //slope
    double b; //intercept
public:
    LinearRegressor(): W(0), b(0) {}
    std::vector <double> train(std::vector <double> X, std::vector <double> y,
                               double learning_rate, unsigned epoch, unsigned batch_size);
    double predict_value(double X) const;
    std::vector <double> predict_vector(std::vector <double> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
};

#endif //HW_LINEAR_REGRESSOR_H
