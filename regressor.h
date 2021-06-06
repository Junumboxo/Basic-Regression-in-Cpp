#ifndef HW_REGRESSOR_H
#define HW_REGRESSOR_H
#include <vector>
#include <string>

class Regressor{
public:

    /* trains the current object (model)
   vector <double> X – input data, vector <double> y – target data
   must be of the same size, otherwise program crashes
   learning_rate, epoch, batch_size – hyperparameters
   must be positive integers, otherwise program terminates execution

    returns the vector of losses shown during the training (its size is equal to the number of epochs) */
    virtual std::vector <double> train(std::vector <double> X, std::vector <double> y, double learning_rate, double epoch, double batch_size) = 0;

    /* calculates and returns the output (prediction) for a single input value */
    virtual double predict_value(double X) const = 0;

    /* calculates and returns outputs (predictions) for several input values simultaneously given in a vector */
    virtual std::vector <double> predict_vector(std::vector <double> X) const = 0;

    /* returns the coefficients of the model. This method is necessary, because the coefficients are private attributes of the class */
    virtual std::vector <double> get_coefficients() const = 0;

    /* saves a model’s coefficients to an external text file */
    virtual void save_model() const = 0;

    /* overwrites the actual model’s coefficients with the new ones read from and external text file. The format of a line in the file must be the following:
    name_of_the_coefficient: value of the coefficient \n
    The order of the coefficients in the file must be the same as their order in the attribute declaration of the class! */
    virtual std::vector <double> get_model_from_file(std::string filename) = 0;

    /*a virtual destructor necessary to destruct the objects of derived classes as well*/
    virtual ~Regressor() {};
};

#endif //HW_REGRESSOR_H
