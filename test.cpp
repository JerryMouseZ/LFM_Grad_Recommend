#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include "lfm.hpp"
using namespace std;


static double test(int items)
{
    ifstream input("realtrain.txt");
    ofstream output("test.txt");
    string line;
    int user = -1;
    int item_id = -1;
    double score;
    vector<double> user_data(items);
    double mse = 0;
    int count = 0;
    while (getline(input, line)) {
        istringstream iss(line);
        if (line.find('|') != -1) {
            iss >> user;
            output << user << "|" << endl;
            // read the matrix line of the user
            read_user(user, user_data);
        } else {
            istringstream iss(line);
            iss >> item_id >> score;
            mse += (user_data[item_id] - score) * (user_data[item_id] - score);
            count++;
            output << item_id << "  " << user_data[item_id] << endl;
        }
    }
    
    mse /= count;
    mse = sqrt(mse);
    input.close();

    cout << "mse : " << mse << endl;
    return mse;
}

int main()
{
    test(642590);
    cout << "Hello world" << std::endl;
    return 0;
}

