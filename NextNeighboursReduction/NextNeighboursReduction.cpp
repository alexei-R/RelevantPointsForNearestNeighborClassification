#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "sdlp.hpp"

using namespace std;
using namespace Eigen;

const double eps = 1.0e-10;

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

bool find_witness_vector(vector<double> p, vector<vector<double>> q, vector<double>& v) {

    static const int vars = p.size();
    int constraints = q.size();

    Eigen::Matrix<double, Dynamic, 1> x(vars, 1);                 // decision variables
    Eigen::Matrix<double, Dynamic, 1> c(vars, 1);                 // objective coefficients
    Eigen::Matrix<double, Dynamic, Dynamic> A(constraints, vars); // constraint matrix
    Eigen::VectorXd b(constraints);                               // constraint bounds

    // the objective coefficients do not matter, any feasible solution is ok
    for (long i = 0; i < vars; i++) c(i) = 0; 
    // adding a small epsilon as constraint bound, as we look for strict inequality,
    // while the solver solves with non strict inequalities in the constraints ( < 0 replaced by <= -eps)
    for (long i = 0; i < constraints; i++) b(i) = -eps;
    for (long row = 0; row < constraints; row++) {
        for (long col = 0; col < vars; col++) {
            // coefficients for v q - v p < 0 
            A(row, col) = q[row][col] - p[col];
        }
    }

    double objective = sdlp::linprog(c, A, b, x, vars);

    if (objective == INFINITY) return false; // infeasible LP problem

    for (long i = 0; i < vars; i++) v.push_back(x(i, 1));

    return true;
}

int main() {

    vector<vector<double>> dataset;
    vector<int> labels;

    read_dataset(dataset, labels);

    /*for (long i = 0; i < dataset.size(); i++) {
        for (long j = 0; j < dataset[i].size(); j++) cout << dataset[i][j] << " ";
        cout << labels[i] << "\n";
    }*/

    vector<vector<double>> q;
    q.push_back({ 1, 1, 1 });
    q.push_back({ 1, 1, 2 });
    q.push_back({ 1, 2, 1 });
    q.push_back({ 2, 1, 1 });
    q.push_back({ 1, 2, 2 });
    q.push_back({ 2, 1, 2 });
    q.push_back({ 2, 2, 1 });
    q.push_back({ 2, 2, 2 });

    vector<double> v;

    bool feasible = find_witness_vector({1.5, 1.5, 1.5}, q, v);

    cout << "feasible: " << feasible << "\n";
    if (feasible) {
        for (long i = 0; i < v.size(); i++) cout << v[i] << " ";
        cout << "\n";
    }

    feasible = find_witness_vector({ 0, 0, 0 }, q, v);

    cout << "feasible: " << feasible << "\n";
    if (feasible) {
        for (long i = 0; i < v.size(); i++) cout << v[i] << " ";
        cout << "\n";
    }

    return 0;
}