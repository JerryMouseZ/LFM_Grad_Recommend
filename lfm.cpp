#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "lfm.hpp"
using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " train\ttraining\n"
            << argv[0] << " predict\tpredicting\n";
        exit(0);
    }
    string cmd = argv[1];
    if (cmd.find("train") != -1) {
        ios::sync_with_stdio(false);
        vector<unordered_map<int, double>> scores;
        int items = read_file("./realtrain.txt", scores);
        vector<vector<double>> P, Q;
        LFM_model(scores, items, 30, 0.0002, 0.0001, 300, 1, P, Q);
        double mse = validate(P, Q);
        ofstream out("mse.txt");
        out << mse << endl;
        out.close();
        predict(P, Q);
    } else {
        vector<vector<double>> P, Q;
        read_matrix("./models/P.txt", P);
        read_matrix("./models/Q.txt", Q);
        predict(P, Q);
    }
    return 0;
}

