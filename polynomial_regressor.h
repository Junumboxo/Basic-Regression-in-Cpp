#ifndef HW_POLYNOMIAL_REGRESSOR_H
#define HW_POLYNOMIAL_REGRESSOR_H

#include "regressor.h"
#include <vector>

class PolynomialRegressor: public Regressor
{
    double a; //coefficient for x^2
    double b; //coefficient for x
    double c; //free parameter
public:
    PolynomialRegressor(): a(0), b(0), c(0) {}
    std::vector <double> train(std::vector <double> X, std::vector <double> y,
                               double learning_rate, unsigned epoch, unsigned batch_size);
    double predict_value(double X) const;
    std::vector <double> predict_vector(std::vector <double> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
};


#endif //HW_POLYNOMIAL_REGRESSOR_H
