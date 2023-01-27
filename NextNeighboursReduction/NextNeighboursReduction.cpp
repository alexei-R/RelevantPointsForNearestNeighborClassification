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

bool find_witness_vector(vector<double>& p, vector<vector<double>>& q, vector<double>& v) {

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

    double norm = 0;
    for (long i = 0; i < vars; i++) {
        v.push_back(x(i, 1));
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);
    for (long i = 0; i < v.size(); i++) v[i] /= norm; // scale to unit vector

    return true;
}

double dot_product(vector<double>& p, vector<double>& q) {
    
    double res = 0;
    for (long i = 0; i < p.size(); i++) res += p[i] * q[i];
    return res;
}

int find_extreme_point_from_witness_vector(vector<vector<double>>& dataset, vector<bool>& is_extreme, vector<double>& witness) {
    int imax = 0;
    double valmax = 0;
    
    for (long i = 0; i < dataset.size(); i++) {
        double prod = dot_product(witness, dataset[i]);
        // initialize or capture new maximum or pick a point not already identified as extreme in case of a tie (within epsilon interval)
        if (i == 0 || prod > valmax + eps || (!(prod < valmax - eps) && is_extreme[imax])) {
            imax = i;
            valmax = prod;
        }
    }

    return imax;
}

void find_extreme_points(vector<vector<double>>& dataset, vector<int>& extreme_point_indices) {

    int d = dataset[0].size();
    vector<bool> is_extreme(dataset.size(), false);
    // the first witness vector can be arbitrary
    vector<double> witness(d, 0);
    witness[0] = 1;
    vector<vector<double>> extreme_points;

    int imax = find_extreme_point_from_witness_vector(dataset, is_extreme, witness);
    extreme_points.push_back(dataset[imax]);
    is_extreme[imax] = true;

    for (long i = 0; i < dataset.size(); i++) {
        witness.clear();      
        bool feasible = find_witness_vector(dataset[i], extreme_points, witness);
        if (feasible) {
            int imax = find_extreme_point_from_witness_vector(dataset, is_extreme, witness);
            if (!is_extreme[imax]) extreme_points.push_back(dataset[imax]);
            is_extreme[imax] = true;
        }
    }

    for (long i = 0; i < is_extreme.size(); i++) {
        if (is_extreme[i]) extreme_point_indices.push_back(i);
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

    vector<vector<double>> q;
    // points inside
    q.push_back({ 2, 2, 1 });
    q.push_back({ 1.5, 1.5, 1 });
    q.push_back({ 2.5, 2.5, 1 });
    q.push_back({ 2.25, 1.75, 1 });
    q.push_back({ 1.75, 2.25, 1 });
    q.push_back({ 2, 2, 1.5 });
    q.push_back({ 1.5, 1.5, 1.5 });
    q.push_back({ 2.5, 2.5, 1.5 });
    q.push_back({ 2.25, 1.75, 1.5 });
    q.push_back({ 1.75, 2.25, 1.5 });
    q.push_back({ 2, 2, 2 });
    q.push_back({ 1.5, 1.5, 2 });
    q.push_back({ 2.5, 2.5, 2 });
    q.push_back({ 2.25, 1.75, 2 });
    q.push_back({ 1.75, 2.25, 2 });
    q.push_back({ 2, 2, 0.5 });
    q.push_back({ 2, 2, 2.5 });
    // points from the convex hull
    q.push_back({ 1, 1, 1 });
    q.push_back({ 1, 2, 1 });
    q.push_back({ 2, 1, 1 });
    q.push_back({ 2, 3, 1 });
    q.push_back({ 3, 2, 1 });
    q.push_back({ 3, 3, 1 });
    q.push_back({ 1, 1, 2 });
    q.push_back({ 1, 2, 2 });
    q.push_back({ 2, 1, 2 });
    q.push_back({ 2, 3, 2 });
    q.push_back({ 3, 2, 2 });
    q.push_back({ 3, 3, 2 });
    q.push_back({ 2, 2, 0 });
    q.push_back({ 2, 2, 3 });

    vector<int> indices;
    find_extreme_points(q, indices);

    for (long i = 0; i < indices.size(); i++) cout << indices[i] << " ";
    cout << "\n";

    return 0;
}