#ifndef UTILITY_H_
#define UTILITY_H_

#include <string>
#include <cmath>
#include <iostream>
#include <deque>

namespace Optimizer
{

typedef std::vector<double>* DBvec;

struct Sparsematrix{
    std::vector<int> sparse_matrix;
    std::vector<int> start_idx;
    std::vector<int> end_idx;
};

typedef struct DBmatrix{

    std::deque<DBvec> records;
    ~DBmatrix(){
        std::deque<DBvec>::iterator iter = records.begin();
        for (; iter != records.end(); ++iter){ delete *iter;}
    }
    int push_back(DBvec vec){
        records.push_back(vec);
        return 0;
    }
    int pop_front(DBvec vec){
        records.pop_front();
        return 0;
    }

} dbmatrix;

    static class LOGOUT {
    public:
        LOGOUT() {}
        std::ostream&  info() {
            std::cout << "[info      ] ";
            return std::cout;
        }
    } logout;


int load_file(
       const std::string& path,
       const std::string& file,
       int& sample_nums,
       int feat_num,
       Sparsematrix& dataset,
       std::vector<int>* labels);

int string_split(
       const std::string& origin, 
       const std::string& spliter, 
       std::vector<std::string>& words);

int compute_signa(
         const Sparsematrix& sample,
         const int& start,
         const int& end,
         bool positive_sample,
         DBvec new_para,
         double& h_sita,
         double& loss);

int compute_fo_deriv(
           DBvec fo_deriv, 
           const double& h_sita,
           const Sparsematrix& sample,
           const int& start,
           const int& end,
           bool positive_sample);

double dot_product(DBvec a, DBvec b, double c = 1.0);

int vec_minus_to(DBvec a, DBvec b, DBvec c);

int scale(DBvec a, double b);

int add_scale_vec(DBvec a,DBvec b, double c);

int sum_two_scale_vec(DBvec a,DBvec b, DBvec c, double d);

void swap_ptr(void* a, void* b);

bool abs_cmp(const double& a,const double& b);

}
#endif
