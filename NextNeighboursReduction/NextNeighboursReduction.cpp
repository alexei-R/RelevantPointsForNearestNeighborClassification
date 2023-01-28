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

void invert_wrt_sphere(vector<vector<double>>& dataset, int center_index, vector<vector<double>>& inverted_dataset) {

    for (long i = 0; i < dataset.size(); i++) {
        inverted_dataset.push_back(dataset[i]);
        if (i != center_index) {
            // shift the origin of the coordinate system to the center of the sphere
            for (long j = 0; j < dataset[i].size(); j++)
                inverted_dataset[i][j] = dataset[i][j] - dataset[center_index][j];

            // get the norm of the vector pointing to the original point
            double norm = 0;
            for (long j = 0; j < dataset[i].size(); j++)
                norm += inverted_dataset[i][j] * inverted_dataset[i][j];
            norm = sqrt(norm);

            // get the norm of the vector pointing to the inverted point
            double norm_inverted = 1 / norm; // sphere radius = 1
            // get the coordinates of the inverted point
            for (long j = 0; j < dataset[i].size(); j++)
                inverted_dataset[i][j] = inverted_dataset[i][j] * norm_inverted / norm;

            // shift the coordinate system back
            for (long j = 0; j < dataset[i].size(); j++)
                inverted_dataset[i][j] += dataset[center_index][j];
        }
    }
}

void find_boundary_points_from_inversion(vector<vector<double>>& dataset, vector<int>& labels, int center_index, vector<int>& boundary_point_indices) {
    
    // remove points of the same class as the center point
    vector<vector<double>> reduced_dataset;
    vector<int> index_mapping(dataset.size(), -1);
    int new_center;
    for (long i = 0; i < dataset.size(); i++) {
        if (i == center_index) new_center = reduced_dataset.size();
        if (labels[i] != labels[center_index] || i == center_index) {
            index_mapping[reduced_dataset.size()] = i;
            reduced_dataset.push_back(dataset[i]);
        }             
    }
    // find the extreme points of the inverted set
    vector<vector<double>> inverted_dataset;
    invert_wrt_sphere(reduced_dataset, new_center, inverted_dataset);
    vector<int> extreme_point_indices;
    find_extreme_points(inverted_dataset, extreme_point_indices);

    // restore the original indices
    for (long i = 0; i < extreme_point_indices.size(); i++) {
        if (extreme_point_indices[i] != new_center)
            boundary_point_indices.push_back(index_mapping[extreme_point_indices[i]]);
    }       
}

void capture_new_boundary_points_from_inversion_step(
    vector<vector<double>>& dataset,
    vector<int>& labels,
    int center_index,
    vector<bool>& is_boundary_point,
    vector<int>& boundary_point_indices) {

    vector<int> boundary_point_indices_from_one_step;
    find_boundary_points_from_inversion(dataset, labels, center_index, boundary_point_indices_from_one_step);
    for (long i = 0; i < boundary_point_indices_from_one_step.size(); i++) {
        // skip points already found before, to avoid duplicates
        if (!is_boundary_point[boundary_point_indices_from_one_step[i]]) {
            is_boundary_point[boundary_point_indices_from_one_step[i]] = true;
            boundary_point_indices.push_back(boundary_point_indices_from_one_step[i]);
        }
    }
}

void flores_velazco(vector<vector<double>>& dataset, vector<int>& labels, vector<int>& boundary_point_indices) {
    vector<bool> is_boundary_point(dataset.size(), false);
    // the first inversion center point can be arbitrary
    capture_new_boundary_points_from_inversion_step(dataset, labels, 0, is_boundary_point, boundary_point_indices);

    // loop through the boundary points
    for (long i = 0; i < boundary_point_indices.size(); i++) {
        capture_new_boundary_points_from_inversion_step(dataset, labels, boundary_point_indices[i], is_boundary_point, boundary_point_indices);
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

    vector<vector<double>> inverted;
    invert_wrt_sphere(q, 17, inverted);

    for (long i = 0; i < q.size(); i++) {
        cout << "original:";
        for (long j = 0; j < q[i].size(); j++) cout << " " << q[i][j];
        cout << " inverted:";
        for (long j = 0; j < q[i].size(); j++) cout << " " << inverted[i][j];
        cout << "\n";
    }

    vector<int> qlabels(q.size(), 1);
    qlabels[29] = 2; qlabels[30] = 2;

    vector<int> boundary_point_indices;
    flores_velazco(q, qlabels, boundary_point_indices);
    sort(boundary_point_indices.begin(), boundary_point_indices.end());
    for (long i = 0; i < boundary_point_indices.size(); i++) cout << boundary_point_indices[i] << " ";
    cout << "\n";

    return 0;
}