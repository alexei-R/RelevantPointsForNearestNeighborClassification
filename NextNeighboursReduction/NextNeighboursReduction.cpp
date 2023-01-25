#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "sdlp.hpp"

using namespace std;
using namespace Eigen;

void read_dataset(vector<vector<double>>& dataset, vector<int>& labels) {
    
    ifstream in_file("winequality-white.csv");
    string line;
    getline(in_file, line); // skip header

    while (getline(in_file, line)) {
        stringstream stream(line);
        vector<double> split_line;
        string token;
        while (getline(stream, token, ';')) {
            split_line.push_back(atof(token.c_str()));
        }
        int class_label = split_line.back();
        split_line.pop_back();
        dataset.push_back(split_line);
        labels.push_back(class_label);
    }
}

int main() {

    vector<vector<double>> dataset;
    vector<int> labels;

    read_dataset(dataset, labels);

    /*for (long i = 0; i < dataset.size(); i++) {
        for (long j = 0; j < dataset[i].size(); j++) cout << dataset[i][j] << " ";
        cout << labels[i] << "\n";
    }*/

    int m = 2 * 7;
    Eigen::Matrix<double, -1, 1> x(7, 1);        // decision variables
    Eigen::Matrix<double, -1, 1> c(7, 1);        // objective coefficients
    Eigen::Matrix<double, -1, 7> A(m, 7); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    c << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    A << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0;
    b << 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;

    double minobj = sdlp::linprog(c, A, b, x, 7);

    std::cout << "prob:\n"
        << std::endl;
    std::cout << "     min x1 + ... + x7," << std::endl;
    std::cout << "     s.t. x1 <=  6,  x2 <=  5, ..., x7 <= 0," << std::endl;
    std::cout << "          x1 >= -1,  x2 >= -2,  ..., x7 >= -7.\n"
        << std::endl;
    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;

    return 0;
}