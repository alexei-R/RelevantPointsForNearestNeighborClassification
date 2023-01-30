#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

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

double square_euclidian_distance(vector<double>& a, vector<double>& b) {
    double square_dist = 0;
    for (long i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        square_dist += diff * diff;
    }
    return square_dist;
}

void jarnik_prim_euclidian_mst(vector<vector<double>>& dataset, vector<vector<int>>& mst_edges) {
    vector<bool> visited(dataset.size(), false);
    vector<bool> reachable(dataset.size(), false);
    vector<double> cost(dataset.size(), 0);
    vector<int> cost_from_which_neighbour(dataset.size(), -1);
    
    // start from arbitrary point
    reachable[0] = true;
    cost[0] = 0;

    // add edges one by one according to priority
    for (long k = 0; k < dataset.size(); k++) {
        // look for the node with the smallest cost of inclusion to the tree
        bool found_reachable = false;
        double min_cost = 0;
        int min_index = 0;
        for (long i = 0; i < dataset.size(); i++) {
            if (!visited[i] && reachable[i] && (!found_reachable || cost[i] < min_cost)) {
                min_cost = cost[i];
                min_index = i;
                found_reachable = true;
            }
        }
        // visit found node and capture the edge added to the MST
        visited[min_index] = true;
        if (cost_from_which_neighbour[min_index] != -1) 
            mst_edges.push_back({ cost_from_which_neighbour[min_index], min_index});
        // update the neighbours
        for (long i = 0; i < dataset.size(); i++) {
            if (!visited[i]) {
                double square_dist = square_euclidian_distance(dataset[i], dataset[min_index]);
                if (!reachable[i] || square_dist < cost[i]) {
                    cost[i] = square_dist;
                    cost_from_which_neighbour[i] = min_index;
                    reachable[i] = true;
                }
            }
        }
    }
}

void eppstein(vector<vector<double>>& dataset, vector<int>& labels, vector<int>& boundary_point_indices) {
    // initial phase calculating boundary point pairs based on the MST
    vector<bool> is_boundary_point(dataset.size(), false);
    vector<vector<int>> mst_edges;
    jarnik_prim_euclidian_mst(dataset, mst_edges);
    for (long i = 0; i < mst_edges.size(); i++) {
        if (labels[mst_edges[i][0]] != labels[mst_edges[i][1]]) {
            for (long j = 0; j < 2; j++) {
                if (!is_boundary_point[mst_edges[i][j]]) {
                    is_boundary_point[mst_edges[i][j]] = true;
                    boundary_point_indices.push_back(mst_edges[i][j]);
                }
            }
        }
    }

    // loop through the boundary points
    for (long i = 0; i < boundary_point_indices.size(); i++) {
        capture_new_boundary_points_from_inversion_step(dataset, labels, boundary_point_indices[i], is_boundary_point, boundary_point_indices);
    }
}

void trim_dataset_dimensionality(vector<vector<double>>& dataset, int d, vector<vector<double>>& trimmed_dataset) {
    for (long i = 0; i < dataset.size(); i++) {
        vector<double> trimmed_datapoint;
        for (long j = 0; j < d; j++) trimmed_datapoint.push_back(dataset[i][j]);
        trimmed_dataset.push_back(trimmed_datapoint);
    }
}

void dataset_portion(vector<vector<double>>& dataset, int n, vector<vector<double>>& partial_dataset) {
    for (long i = 0; i < dataset.size() && i < n; i++) {
        partial_dataset.push_back(dataset[i]);
    }
}

// class frequencies:
// label:    0    1    2    3    4    5    6    7    8    9   10
// freq:     0    0    0   20  163 1457 2198  880  175    5    0
void consolidate_labels(vector<int>& labels) {
    for (long i = 0; i < labels.size(); i++) {
        if (labels[i] < 5) labels[i] = 1;
        else if (labels[i] < 8) labels[i] = 2;
        else labels[i] = 3;
    }
}

int main(int argc, const char** argv) {

    vector<vector<double>> dataset;
    vector<int> labels;
    read_dataset(dataset, labels);

    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int consolidate = atoi(argv[3]);

    vector<vector<double>> partial_dataset;
    dataset_portion(dataset, n, partial_dataset);
    vector<vector<double>> trimmed_dataset;
    trim_dataset_dimensionality(partial_dataset, d, trimmed_dataset);
    if (consolidate) consolidate_labels(labels);

    vector<int> boundary_point_indices;
    cout << trimmed_dataset.size() << ";" << trimmed_dataset[0].size() << ";" << consolidate << ";";
    auto start = chrono::high_resolution_clock::now();
    flores_velazco(trimmed_dataset, labels, boundary_point_indices);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << elapsed.count() << ";" << boundary_point_indices.size() << ";";
    boundary_point_indices.clear();
    auto start1 = chrono::high_resolution_clock::now();
    eppstein(trimmed_dataset, labels, boundary_point_indices);
    auto end1 = chrono::high_resolution_clock::now();
    auto elapsed1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout << elapsed1.count() << ";" << boundary_point_indices.size() << endl;
    
    return 0;
}