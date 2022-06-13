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
#pragma omp parallel for
    for (int i = 0; i < x; ++i) {
        array[i].resize(y);
        for (int j = 0; j < y; ++j) {
            int num = rand() % 100000;
            array[i][j] = (double)(num) / 100000;
        }
    }
}

static void model_persist(vector<vector<double>> &P, vector<vector<double>> &Q)
{
    char filename[24];
    sprintf(filename, "models/P.txt");
    // 内存不够了，写入文件
    ofstream output(filename);
    for (int i = 0; i < P.size(); ++i) {
        for (int j = 0; j < P[i].size(); ++j) {
            output << P[i][j] << " ";
        }
        output << endl;
    }
    output.close();

    sprintf(filename, "models/Q.txt");
    output.open(filename);
    for (int i = 0; i < Q.size(); ++i) {
        for (int k = 0; k < Q[i].size(); ++k) {
            output << Q[i][k] << " ";
        }
        output << endl;
    }
    output.close();
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

static void LFM_model(vector<unordered_map<int, double>> &R, int max_iterm, int factors, double alpha, double lamda, int max_step, double min_eui, vector<vector<double>> &P, vector<vector<double>> &Q)
{
    auto start = clock();
    int m = R.size();
    random_array(m, factors, P);
    random_array(factors, max_iterm, Q);

    for (int i = 0; i < max_step; ++i) {
        double eui = train_iter(R, P, Q, alpha, lamda, m, max_iterm);
        if (eui <= min_eui)
            break;
    }

    auto end = clock();
    cout<< "training time = " << double(end-start) / CLOCKS_PER_SEC << "s" << endl;
    model_persist(P, Q);
}

static void read_matrix(const char *filename, vector<vector<double>> &matrix)
{
    ifstream input(filename);
    double value;
    string line;
    int i = 0;
    while (getline(input, line)) {
        matrix.emplace_back(vector<double>());
        istringstream iss(line);
        vector<double> &row = matrix[i];
        while (iss >> value){
            row.emplace_back(value);
        }
        i++;
    }
    input.close();
}

static double predict_score(int user, int item, vector<vector<double>> &P, vector<vector<double>> &Q)
{
    int l = Q.size();
    double res = 0;
    for (int i = 0; i < l; ++i) {
        res += P[user][i] * Q[i][item];
    }
    return res;
}

static double validate(vector<vector<double>> &P, vector<vector<double>> &Q)
{
    ifstream input("validate.txt");
    string line;
    int user = -1;
    int item_id = -1;
    double score;
    double mse = 0;
    int count = 0;
    while (getline(input, line)) {
        istringstream iss(line);
        if (line.find('|') != -1) {
            iss >> user;
        } else {
            istringstream iss(line);
            iss >> item_id >> score;
            double value = predict_score(user, item_id, P, Q);
            mse += abs(value - score);
            count++;
        }
    }

    mse /= count;
    input.close();

    cout << "mse : " << mse << endl;
    return mse;
}

static void predict(vector<vector<double>> &P, vector<vector<double>> &Q)
{
    auto start = clock();
    ifstream input("../data/test.txt");
    ofstream output("result.txt");
    string line;
    int user = -1;
    int item_id = -1;
    while (getline(input, line)) {
        istringstream iss(line);
        if (line.find('|') != -1) {
            iss >> user;
            output << user << "|6" << endl;
        } else {
            istringstream iss(line);
            iss >> item_id;
            output << item_id << "  " << predict_score(user, item_id, P, Q) << endl;
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
