//
// Created by tatia on 5/1/2021.
//

#ifndef HW_DATAFILEHANDLER_H
#define HW_DATAFILEHANDLER_H

#include <vector>
#include <string>

class DataFileHandler{
public:
    static std::vector<std::vector<double>> getVectorFromFile(std::string file_x, std::string file_y);
};

#endif //HW_DATAFILEHANDLER_H
