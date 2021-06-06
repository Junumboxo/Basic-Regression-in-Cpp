#ifndef HW_REGRESSIONEXCEPTIONS_H
#define HW_REGRESSIONEXCEPTIONS_H

#include <exception>

class FileOpenException: public std::exception
{
public:
    virtual const char* what() const throw()
    {
        return "Cannot open the file to read!";
    }
};

class FileReadException: public std::exception
{
public:
    virtual const char* what() const throw()
    {
        return "Failed to read the file! The data in the file isn't written in the appropriate format to read them.";
    }
};


#endif //HW_REGRESSIONEXCEPTIONS_H
