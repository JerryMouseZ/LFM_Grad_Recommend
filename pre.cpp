#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include "lfm.hpp"

using namespace std;

static inline void preprocess(int percentage, vector<unordered_map<int, double>> &scores)
{
    ofstream validate("validate.txt");
    ofstream realtrain("realtrain.txt");
    int m = scores.size();
    for (int i = 0; i < m; ++i) {
        unordered_map<int, double> &map = scores[i];
        int count = map.size() * percentage / 100;
        validate << i << "|" << count << endl;
        realtrain << i << "|" << map.size() - count << endl;
        int dd = 0;
        for (auto &kv : map) {
            if (dd < count) {
                validate << kv.first << "  " << kv.second << endl;
                dd++;
            } else {
                realtrain << kv.first << "  " << kv.second << endl;
            }
        }
    }
    validate.close();
    realtrain.close();
}

int main()
{
    vector<unordered_map<int, double>> scores;
    read_file("../data/train.txt", scores);
    preprocess(20, scores);
    return 0;
}

