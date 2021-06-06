#ifndef HW_POLYNOMIALREGRESSOR_H
#define HW_POLYNOMIALREGRESSOR_H

#include "regressor.h"
#include <vector>
#include <string>

class PolynomialRegressor: public Regressor
{
    double a; //coefficient for x^2
    double b; //coefficient for x
    double c; //free parameter
public:
    PolynomialRegressor(): a(0), b(0), c(0) {}
    std::vector <double> train(std::vector <double> X, std::vector <double> y,
                               double learning_rate = 0.0001, double epoch = 2500, double batch_size = 5);
    double predict_value(double X) const;
    std::vector <double> predict_vector(std::vector <double> X) const;
    std::vector <double> get_coefficients() const;
    void save_model() const;
    std::vector <double> get_model_from_file(std::string filename);
};


#endif //HW_POLYNOMIALREGRESSOR_H
