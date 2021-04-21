#ifndef HW_REGRESSOR_H
#define HW_REGRESSOR_H
#include <vector>

class Regressor{
public:
    virtual ~Regressor() {}
    virtual std::vector <double> train(std::vector <double> X, std::vector <double> y,
                               double learning_rate, unsigned epoch, unsigned batch_size) = 0;
    virtual double predict_value(double X) const = 0;
    virtual std::vector <double> predict_vector(std::vector <double> X) const = 0;
    virtual std::vector <double> get_coefficients() const = 0;
    virtual void save_model() const = 0;
};

#endif //HW_REGRESSOR_H
