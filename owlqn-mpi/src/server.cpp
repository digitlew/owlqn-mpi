#include <vector>
#include <algorithm>
#include <math.h>
#include "server.h"
#include "glog/logging.h"

namespace Optimizer
{

int Server::_load_test_data() {
    PLOG(INFO) << "start loading test data";
    std::string path="./data";
    std::string file=_validation_file;
    if (0 != load_file(path, file, _sample_nums, 
               _feat_nums, _validation_set, _validation_labels)) {
        PLOG(ERROR) << "server load test data error,exit!";
        return -1;
    }
    PLOG(INFO) << "load succ: sample_num is " << _sample_nums;
    return 0;
}


int Server::_init_interal_parameter(){
    PLOG(INFO) << "server init parmeter";
    _x = new std::vector<double>(_feat_nums);
    _newX = new std::vector<double>(_feat_nums);
    _grad = new std::vector<double>(_feat_nums);
    _newGrad = new std::vector<double>(_feat_nums);
    _fo_deriv = new std::vector<double>(_feat_nums);
    _newton_deriv = _fo_deriv;
    _alphas = new std::vector<double>(_m, 0.0);
    _validation_labels = new std::vector<int>;
    
    _sList = new dbmatrix{};
    _yList = new dbmatrix{};
    _roList = new std::deque<double>;
    
    return 0;
}

int Server::_check_test_loss(){
     
    std::vector<int>::iterator s_iter = _validation_set.start_idx.begin();
    std::vector<int>::iterator e_iter = _validation_set.end_idx.begin();
    std::vector<int>::iterator l_iter = _validation_labels->begin();
    double _loss = 0.0;
    DBvec h_sita_vec = NULL;
    int pos_sample = 0;
    int neg_sample = 0;
    if (_auc) {
        h_sita_vec = new std::vector<double>;
    }
   
    for (;s_iter!=_validation_set.start_idx.end()&&l_iter!=_validation_labels->end(); ++s_iter, ++l_iter, ++e_iter) {
        double h_sita = 0.0;
        if (!_auc) { 
            if (0 != compute_signa(_validation_set, *s_iter, *e_iter, *l_iter, _newX, h_sita, _loss)) {
                return -1;
            }
        }else {
            //compute auc
            for (int i = *s_iter; i < *e_iter; ++i) {
                h_sita += _newX->at(_validation_set.sparse_matrix.at(i));
            }
            double temp = 1.0 + exp(-h_sita);
            h_sita = 1.0 / temp;
            if (!*l_iter) {h_sita *= -1.0;}
            h_sita_vec->push_back(h_sita);
            if (*l_iter) {
                ++pos_sample;
            } else {
                ++neg_sample;
            }
        }    
    }
    if (_auc) {
        std::sort(h_sita_vec->begin(), h_sita_vec->end(), abs_cmp);
        std::vector<double>::iterator p_iter = h_sita_vec->begin();
        double delta_x = 1.0 / neg_sample;
        double delta_y = 1.0 / pos_sample;
        int inc_y = 0;
        double cur_auc = 0.0;
        for (; p_iter != h_sita_vec->end(); ++p_iter) {
            if (*p_iter > 0) {
                ++inc_y;
            } else {
                cur_auc += (delta_x * inc_y * delta_y);
            }
        }
        logout.info() << "auc is :" << cur_auc << std::endl;

    } 

    if (NULL != h_sita_vec) {delete h_sita_vec;}
    
    PLOG(INFO) << "current iter:" << _iters << ", validation_loss:" << _loss / _sample_nums;
    //logout.info() << "test loss is :" << _loss / _sample_nums  << std::endl;
    return 0;
}

int Server::_assemble_train_loss(double& loss){
    PLOG(INFO) << "total logloss from train is " << loss;
    // add regulation loss
    for(int j = 0; j < _feat_nums; ++j) {
        loss += (_c1 * fabs(_x->at(j)) + 0.5 * _c2 * _x->at(j) * _x->at(j));
        //_fo_deriv[j] = 2 * _c2 * _newX[j];
    }
    PLOG(INFO) << "add reg-term loss ,total loss is" << loss / _sample_nums;
    logout.info() << "add reg_term loss is :" << loss / _sample_nums  << std::endl;
    return 0;
}

int Server::_add_regterm_grad() {
    if (0.0 == _c1) {
        PLOG(INFO) << "optimizer do not add l1 reg-term!";
    }
    //L-BFGS for l1 
    int count = 0;
    for(int i =0; i < _feat_nums; ++i) {
        // add l2 reg-term if _c2 <> 0
        _grad->at(i) += (_c2 * _x->at(i));
        // add l1 reg-term if _c1 <> 0
        if (_x->at(i) > 0) {
            _newGrad->at(i) = _grad->at(i) + _c1;
        } else if (_x->at(i) < 0) {
            _newGrad->at(i) = _grad->at(i) - _c1;
        } else if (_grad->at(i) < -1.0 * _c1) {
            _newGrad->at(i) = _grad->at(i) + _c1;
        } else if (_grad->at(i) > _c1) {
            _newGrad->at(i) = _grad->at(i) - _c1;
        } else {
            _newGrad->at(i) = 0.0;
        }
        _fo_deriv->at(i) = -_newGrad->at(i);
        if (_fo_deriv->at(i) == 0) {
            ++count;
        }
    }
    logout.info() << " subdirection case [" << count << " ] newton_deriv to be 0!" << std::endl;
    return 0;
}

int Server::_back_tracking_line_search(const double& alpha){
    if (0 != sum_two_scale_vec(_newX, _x, _newton_deriv, alpha)) {return -1;}
    int count = 0;
    for (int i = 0; i < _newX->size(); ++i) {
        if (_newX->at(i) * _x->at(i) < 0) {
            _newX->at(i) = 0.0;
            ++count;
        }
    }
    logout.info() << "line search case [" << count << " ] newton_deriv to be 0!" << std::endl;
    return 0;
}

int Server::_generate_newton_direction(){
    // add l1 and l2 grad
    if (0 != _add_regterm_grad()) {
        PLOG(ERROR) << "server add reg-term grad error!";
    }
    //L-BFGS
    int iter = _m < _sList->records.size() ?  _m : _sList->records.size(); 
    if (0 == iter) {return 0;}
    

    for (int i = iter - 1; i >= 0; --i) {
        _alphas->at(i) = dot_product(_sList->records[i], _newton_deriv) / _roList->at(i);  
        if (0 != sum_two_scale_vec(_newton_deriv, _newton_deriv, _yList->records[i], -_alphas->at(i))) {return -1;}
    }
    // H = S * Y(-1) =  <S, Y> / <Y,Y>
    DBvec y_iter = _yList->records[iter -1];
    double ydoty = dot_product(y_iter, y_iter);
    double scalar = _roList->at(iter - 1)/ydoty;
    scale(_newton_deriv, scalar);
    logout.info() << "scalar is " << scalar << "ydoty" << ydoty <<std::endl;
    
    for (int i = 0; i < iter; ++i) {
        double beta =  dot_product(_yList->records[i], _newton_deriv) / _roList->at(i);
        add_scale_vec(_newton_deriv, _sList->records[i], _alphas->at(i) - beta);
    }
    return 0;
}

int Server::_correction_direction() {
    if (_newton_deriv->size() != _newGrad->size()) {
        PLOG(ERROR) << "size of newton_derive and grad not same!";
        return -1;
    }
    // make sure newton direction is same with steepdescent direction(which is opposite of gradient)
    int count = 0;
    for (int i = 0; i < _newton_deriv->size(); ++i) {
        if (_newton_deriv->at(i) *  _newGrad->at(i) >= 0) {
            _newton_deriv->at(i) = 0.0;
            ++count;
        }
    }
    logout.info() << "correction direction case [" << count << " ] newton_deriv to be 0!" << std::endl;
    return 0;
}

int Server::_update_internal_parameter() {

    // update slist
    DBvec new_s, new_y;
    if (_m == _sList->records.size()) {
        new_s = _sList->records.front();
        _sList->records.pop_front();
        new_y = _yList->records.front();
        _yList->records.pop_front();
        _roList->pop_front();
    }

    if (_sList->records.size() < _m && _yList->records.size() < _m) {
        new_s = new std::vector<double>(_feat_nums);
        if (0 != vec_minus_to(_newX, _x, new_s)) {return -1;}
        _sList->records.push_back(new_s);
        
        new_y = new std::vector<double>(_feat_nums);
        if (0 != vec_minus_to(_newGrad, _grad, new_y)) {return -1;}
        _yList->records.push_back(new_y);

        double new_ro = dot_product(new_s, new_y);
        _roList->push_back(new_ro); 

    }

    DBvec temp = _x;
    _x = _newX;
    _newX = temp;
    temp = _grad;
    _grad = _newGrad;
    _newGrad = temp;

    return 0;
}

bool Server::_go_bts_condition(double old_value, double new_value, double& alpha, const double& orig) {

    int go_bts = 1;
    bool flag = true;

    double c = 1e-4;
    double backoff = 0.5;

    // logic for bts
    double cond = c * alpha * orig;
    if (old_value + cond >= new_value) {
        go_bts = 0;
        flag = false;
    }

    alpha *= backoff;

    // inform worker
    MPI_Barrier(_communicator);
    MPI_Bcast(&go_bts, 1, MPI_INT, 0, _communicator);
    MPI_Barrier(_communicator);

    return flag;
}

int Server::_get_loss_and_grad_from_slaves(double& train_loss){
    MPI_Barrier(_communicator);
    MPI_Bcast(&(_newX->front()), _feat_nums, MPI_DOUBLE, 0, _communicator);
    MPI_Barrier(_communicator);

    MPI_Reduce(MPI_IN_PLACE, &train_loss, 1, MPI_DOUBLE, MPI_SUM, 0, _communicator);
    //train loss for t:wqhis iter
    if (0 != _assemble_train_loss(train_loss)) {
        PLOG(WARNING) << "server compute train loss error!";
        return -1;
    }
    
    //set zero 
    for (int i = 0; i < _feat_nums; ++ i) {
        _newGrad->at(i) = 0.0;
    }
    //get first-order derivative
    MPI_Reduce(MPI_IN_PLACE, &(_newGrad->front()), _feat_nums, MPI_DOUBLE, MPI_SUM, 0, _communicator);

    return 0;
}

int Server::run() {
    //init x
    if (0 != _init_interal_parameter()) {
        PLOG(ERROR) << "server initial parameter error!";
        return -1;
    }
    //load test data set
    if (0 != _load_test_data()) {
        PLOG(ERROR) << "server load test data error!";
        return -1;
    }

    LOG(INFO) << "server initialize complete";

    if (0 != _check_test_loss()) {
        PLOG(WARNING) << "server check test loss error!";
    }

    double train_loss;
    if (0 != _get_loss_and_grad_from_slaves(train_loss)) {
        PLOG(ERROR) << "server initial grad and loss error!";
        return -1;
    }
    //swap_ptr(_grad, _newGrad);
    for (int i = 0; i < _feat_nums; ++ i) {
        _grad->at(i) = _newGrad->at(i);
    }

    //iterator compute
    int count = 0;
    while (count < _iters) {

        //compute newton direction, L-BFGS
        if (0 != _generate_newton_direction()) {
            PLOG(WARNING) << "server compute newton direction error!";
        }
        
        //fix newton directin and loss function derivative
        if (0 != _correction_direction()) {
            PLOG(WARNING) << "server correction direction error!";
        }

        //back tracking line search
        double old_value;
        double orig_dir_deriv = dot_product(_newton_deriv, _newGrad);
        double alpha = 1.0 / sqrt(dot_product(_newton_deriv, _newton_deriv));
        logout.info() << "alpha is: " << alpha << " orig is" << orig_dir_deriv << std::endl;

        old_value = train_loss;
        train_loss = 0.0;
        do {
            //get next point
            if (0 != _back_tracking_line_search(alpha)) {
                PLOG(WARNING) << "server doing line search error!";
            }
            // compute loss
            if (0 != _get_loss_and_grad_from_slaves(train_loss)) {
                PLOG(WARNING) << "server compute train loss error!";
            }
        } while(_go_bts_condition(old_value, train_loss, alpha, orig_dir_deriv));

        if (0 != _check_test_loss()) {
            PLOG(WARNING) << "server check test loss error!";
        }
       /*
            for(int i = 0; i < 100; ++i) {
                logout.info() << "newX" << _newX->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "X" << _x->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "newgrad" << _newGrad->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "grad" << _grad->at(i) << std::endl;
            } */
        if (0 != _update_internal_parameter()) {
            PLOG(WARNING) << "server update internal error!";
        }
        
        int resume_signal = 0;
        // logic for resume_signal
        MPI_Bcast(&resume_signal, 1, MPI_INT, 0, _communicator);
        if (1 == resume_signal) {
            break;
        }
        
        ++count;
    }

}


}
