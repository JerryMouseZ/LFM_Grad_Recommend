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
        LFM_model(scores, items, 30, 0.0002, 0.0001, 300, 1);
        double mse = validate(642960);
        ofstream out("mse.txt");
        out << mse << endl;
        out.close();
    } else {
        predict(64296);
    }
    return 0;
}

