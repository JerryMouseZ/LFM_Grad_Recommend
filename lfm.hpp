#ifndef LFM_H
#define LFM_H
#include <cstdio>
#include <cstdlib>
/* #include <omp.h> */
#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <random>

using namespace std;

static inline int read_file(const char *filename, vector<unordered_map<int, double>> &scores)
{
    int max_iterm = -1;
    ifstream input(filename);
    string line;
    int user = -1;
    int item_id = -1, score = 0;
    while (getline(input, line)) {
        istringstream iss(line);
        if (line.find('|') != -1) {
            iss >> user;
            scores.emplace_back(unordered_map<int, double>());
        } else {
            iss >> item_id >> score;
            if (item_id > max_iterm) {
                max_iterm = item_id;
            }
            scores[user][item_id] = score;
        }
    }
    input.close();
    return max_iterm + 1;
}


static void random_array(int x, int y, vector<vector<double>> &array)
{
    srand(time(0));
    array.resize(x);
    for (int i = 0; i < x; ++i) {
        array[i].resize(y);
        for (int j = 0; j < y; ++j) {
            int num = rand() % 100000;
            array[i][j] = (double)(num) / 100000;
        }
    }
}

static void matrix_mul(vector<vector<double>> &P, vector<vector<double>> &Q, int x, int y)
{
#pragma omp parallel for
    for (int i = 0; i < x; ++i) {
        vector<double> line(y);
        for (int k = 0; k < Q.size(); ++k) {
            for (int j = 0; j < y; ++j) {
                line[j] += P[i][k] * Q[k][j];
            }
        }
        
        // 内存不够了，写入文件
        char filename[24];
        sprintf(filename, "models/%d.txt", i);
        ofstream output(filename);
        for (int j = 0; j < y; ++j) {
            output << line[j] << " ";
        }
        output << endl;
        output.close();
    }
}

static double norm(vector<vector<double>> &matrix)
{
    double sum_of_sq = 0;
    for(int i = 0; i < matrix.size(); i++)
    {
        for(int j = 0; j < matrix[0].size(); j++)
        {
            sum_of_sq += (matrix[i][j] * matrix[i][j]);
        }
    }

    double out;
    return out;
}

static double train_iter(vector<unordered_map<int, double>> &R, vector<vector<double>> &P, vector<vector<double>> &Q, double alpha, double lamda, int m, int n)
{
    for (int i = 0; i < m; ++i) {
        unordered_map<int, double> &map = R[i];
        for (auto & kv : map) {
            // 计算估计值和R中的差
            int user = i, item = kv.first;
            double score = kv.second;
            double error = score;
            // 计算估计值
            for (int k = 0; k < Q.size(); ++k) {
                error -= P[user][k] * Q[k][item];
            }
            for (int k = 0; k < Q.size(); ++k) {
                P[user][k] -= alpha * (-2 * error * Q[k][item] + 2 * lamda * P[user][k]);
                Q[k][item] -= alpha * (-2 * error * P[user][k] + 2 * lamda * Q[k][item]);
            }
        }
    }

    // 计算损失函数eui
    double eui = 0;
    for (int i = 0; i < m; ++i) {
        unordered_map<int, double> &map = R[i];
        for (auto & kv : map) {
            int user = i, item = kv.first;
            double score = kv.second;
            double error = score;
            for (int k = 0; k < Q.size(); ++k) {
                error -= P[user][k] * Q[k][item];
            }
            eui += error * error;
        }
    }

    // 正则化项
    eui += lamda * (norm(P) + norm(Q));
    return eui;
}

static void LFM_model(vector<unordered_map<int, double>> &R, int max_iterm, int factors, double alpha, double lamda, int max_step, double min_eui)
{
    auto start = clock();
    int m = R.size();
    vector<vector<double>> P;
    vector<vector<double>> Q;
    random_array(m, factors, P);
    random_array(factors, max_iterm, Q);

    for (int i = 0; i < max_step; ++i) {
        double eui = train_iter(R, P, Q, alpha, lamda, m, max_iterm);
        if (eui <= min_eui)
            break;
    }

    matrix_mul(P, Q, m, max_iterm);
    auto end = clock();
    cout<< "training time = " << double(end-start) / CLOCKS_PER_SEC << "s" << endl;
}

static void read_user(int user, vector<double> &array)
{
    char filename[24];
    sprintf(filename, "models/%d.txt", user);
    ifstream input(filename);
    for (int j = 0; j < array.size(); ++j) {
        input >> array[j];
    }
    input.close();
}

static double validate(int items)
{
    ifstream input("validate.txt");
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
            // read the matrix line of the user
            read_user(user, user_data);
        } else {
            istringstream iss(line);
            iss >> item_id >> score;
            mse += abs(user_data[item_id] - score);
            count++;
        }
    }
    
    mse /= count;
    input.close();

    cout << "mse : " << mse << endl;
    return mse;
}

static void predict(int items)
{
    auto start = clock();
    ifstream input("../data/test.txt");
    ofstream output("result.txt");
    string line;
    int user = -1;
    int item_id = -1;
    vector<double> user_data(items);
    while (getline(input, line)) {
        istringstream iss(line);
        if (line.find('|') != -1) {
            iss >> user;
            output << user << "|6" << endl;
            // read the matrix line of the user
            read_user(user, user_data);
        } else {
            istringstream iss(line);
            iss >> item_id;
            output << item_id << "  " << user_data[item_id] << endl;
        }
    }
    output.close();
    input.close();

    auto end = clock();
    cout<< "training time = " << double(end-start) / CLOCKS_PER_SEC << "s" << endl;
}

static inline void print_maps(vector<unordered_map<int, double>> &scores) {
    int n = scores.size();
    for (int i = 0; i < n; ++i) {
        unordered_map<int, double> &map = scores[i];
        cout << i << "|" << map.size() << endl;
        for (auto & kv : map) {
            cout << kv.first << " " << kv.second << endl;
        }
    }
}
#endif
