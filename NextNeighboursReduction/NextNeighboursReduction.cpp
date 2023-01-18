#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

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

    for (long i = 0; i < dataset.size(); i++) {
        for (long j = 0; j < dataset[i].size(); j++) cout << dataset[i][j] << " ";
        cout << labels[i] << "\n";
    }

    return 0;
}