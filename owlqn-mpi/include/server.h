#ifndef SERVER_H_
#define SERVER_H_

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <math.h>
#include "mpi.h"
#include "utility.h"

#define delptr(x) {\
    if (NULL != x) {\
        delete x; \
    }\
}

namespace Optimizer
{

class NodeWork{
public:

    NodeWork(){}

    virtual int run()=0;

    virtual ~NodeWork(){}
};


class Server{

friend class server_run_Test;

public:

    Server(int m, int iters, double c1, double c2, std::string file, int feat_nums, MPI_Comm comm)
         :_m(m), _iters(iters), _c1(c1), _c2(c2), _validation_file(file), _feat_nums(feat_nums), _auc(1), _communicator(comm) {}

    int run();

    DBvec getter_x(){return _x;}

    DBvec getter_newx(){return _newX;}

    ~Server(){
        delptr(_x);
        delptr(_newX);
        delptr(_grad);
        delptr(_newGrad);
        delptr(_fo_deriv);
        delptr(_newton_deriv); 
        
        delptr(_validation_labels); 
        delptr(_sList); 
        delptr(_yList); 
        delptr(_roList); 
        /*
        delptr(_validation_set); 
        */
    }

private:

    int _load_test_data();

    int _init_interal_parameter();

    int _check_test_loss();

    int _assemble_train_loss(double& loss);

    int _update_internal_parameter();

    int _generate_newton_direction();

    int _correction_direction();

    int _back_tracking_line_search(const double& alpha);

    bool _go_bts_condition(double o, double n, double& alpha, const double& orig);

    int _add_regterm_grad();

    int _get_loss_and_grad_from_slaves(double&);

private:

    int _m, _iters, _feat_nums, _sample_nums, _auc;

    double _c1; //const coefficient for L1 regular term

    double _c2; //const coefficient for L2 regular term

    bool _using_l1; // owlqn for L1 or L-BFGS for L2

    std::string _validation_file;

    DBvec _x, _newX, _grad, _newGrad, _fo_deriv, _newton_deriv, _alphas;

    std::deque<double>* _roList;

    dbmatrix* _sList;

    dbmatrix* _yList;

    Sparsematrix _validation_set;

    std::vector<int>* _validation_labels;

    MPI_Comm _communicator;

};

}
#endif
