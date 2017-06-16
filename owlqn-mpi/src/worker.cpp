#include <vector>
#include <math.h>
#include "worker.h"
#include "glog/logging.h"

namespace Optimizer
{

int Worker::_load_train_data() {
    PLOG(INFO) << "start loading train data";
    std::string path="./data";
    std::string file=_train_file;
    if (0 != load_file(path, file, _sample_nums, 
               _feat_nums, _train_set, _train_labels)) {
        PLOG(ERROR) << "work load test data error,exit!";
        return -1;
    }
    _h_sita_vec = new std::vector<double>(_sample_nums);
    PLOG(INFO) << "load succ: sample_num is " << _sample_nums;
    return 0;
}

int Worker::_init_interal_parameter() {
    _x = new std::vector<double>(_feat_nums);
    _fo_deriv = new std::vector<double>(_feat_nums);
    _train_labels = new std::vector<int>;
    return 0;
}

int Worker::_compute_loss_and_grad() {
    std::vector<int>::iterator s_iter = _train_set.start_idx.begin();
    std::vector<int>::iterator e_iter = _train_set.end_idx.begin();
    std::vector<int>::iterator l_iter = _train_labels->begin();
    std::vector<double>::iterator h_iter = _h_sita_vec->begin();
    _loss = 0.0;

    for(;s_iter!=_train_set.start_idx.end()&&l_iter!=_train_labels->end(); ++s_iter, ++l_iter, ++h_iter, ++e_iter){
        
        if (0 != compute_signa(_train_set, *s_iter, *e_iter, *l_iter, _x, *h_iter, _loss)) {
            return -1;
        }
        // compute logloss
        //_loss += log(*h_iter);
    }  
    logout.info() << "worker loss is :" << _loss / _sample_nums << std::endl;
 
    return 0;
}

int Worker::_compute_f_deriv() {
    std::vector<int>::iterator s_iter = _train_set.start_idx.begin();
    std::vector<int>::iterator e_iter = _train_set.end_idx.begin();
    std::vector<double>::iterator h_iter = _h_sita_vec->begin();
    std::vector<int>::iterator l_iter = _train_labels->begin();

    for (int j = 0; j < _fo_deriv->size(); ++j) {_fo_deriv->at(j) = 0.0;}

    for(;s_iter!=_train_set.start_idx.end()&&h_iter!=_h_sita_vec->end()&&l_iter!=_train_labels->end(); ++s_iter, ++h_iter, ++e_iter, ++l_iter){
        // compute deriv
        compute_fo_deriv(_fo_deriv, *h_iter, _train_set, *s_iter, *e_iter, *l_iter);
    }

    int count = 0;
    int hit = 0;
    for (int j = 0; j < _fo_deriv->size(); ++j) {
        if (_fo_deriv->at(j) != 0) {
            ++count;
        }
        if (_fo_deriv->at(j) >= -1 and _fo_deriv->at(j) <= 1) {
            ++hit;
        }
    }

    return 0;
}

bool Worker::_go_bts_condition() {
    // fetch information
    int go_bts = 0;
    MPI_Barrier(_communicator);
    MPI_Bcast(&go_bts, 1, MPI_INT, 0, _communicator);
    MPI_Barrier(_communicator);

    if (1 == go_bts) {
        return true;
    }
    return false;
}

int Worker::_send_loss_and_grad() {
    MPI_Barrier(_communicator);
    MPI_Bcast(&(_x->front()), _feat_nums, MPI_DOUBLE, 0, _communicator);
    MPI_Barrier(_communicator);
    //compute train loss and new loss-term gradient
    if (0 != _compute_loss_and_grad()) {
        PLOG(WARNING) << "worker compute loss and grad error!";
    }

    //send loss and sampl number
    MPI_Reduce(&_loss, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, _communicator);
   
    //compute first_order derivative
    if (0 != _compute_f_deriv()) {
        PLOG(WARNING) << "worker compute first-order derivative error!";
    } 
    MPI_Reduce(&(_fo_deriv->front()), NULL, _feat_nums, MPI_DOUBLE, MPI_SUM, 0, _communicator);

    return 0;
}

int Worker::run() {
    //init x
    if (0 != _init_interal_parameter()) {
        PLOG(ERROR) << "worker initial parameter error!";
        return -1;
    }
    //load train data
    if (0 != _load_train_data()) {
        PLOG(ERROR) << "worker load train data error!";
        return -1;
    }

    LOG(INFO) << "worker initialize complete";

    if (0 != _send_loss_and_grad()) {
        PLOG(ERROR) << "worker compute loss and grad error!";
        return -1;
    }

    int count = 0;
    while (count < _iters) {
        
        do {
            // wait for new parameter
            if (0 != _send_loss_and_grad()) {
                PLOG(WARNING) << "worker compute loss and grad error!";
            }
        } while (_go_bts_condition());

        int resume_signal = 0;
        MPI_Bcast(&resume_signal, 1, MPI_INT, 0, _communicator);
        // logic for resume_signal
        if (1 == resume_signal) {
            break;
        }

        ++count;
    }

} 

}
