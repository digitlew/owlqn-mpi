#ifndef WORKER_H_
#define WORKER_H_

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include "server.h"
#include "utility.h"
#include "mpi.h"

namespace Optimizer
{

class Worker{

friend class server_run_Test;

public:

    Worker(int iters, std::string file, int feat_nums, MPI_Comm comm)
         :_iters(iters), _train_file(file), _feat_nums(feat_nums), _communicator(comm) {}

    int run();

    DBvec getter_f_deriv(){return _fo_deriv;}

    const double& getter_loss(){return _loss;}

    ~Worker(){
        delptr(_x);
        delptr(_fo_deriv);
        delptr(_train_labels);
        delptr(_h_sita_vec);
    }

private:

    int _load_train_data();

    int _init_interal_parameter();

    int _compute_loss_and_grad();

    int _compute_f_deriv();

    bool _go_bts_condition();

    int _send_loss_and_grad();

private:

    double _loss;

    int  _sample_nums, _feat_nums , _iters;

    DBvec _fo_deriv, _x;

    std::string _train_file;

    Sparsematrix _train_set;

    std::vector<int>* _train_labels;
   
    std::vector<double>* _h_sita_vec;

    MPI_Comm _communicator;

};

}
#endif
