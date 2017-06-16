#include <fstream>
#include <vector>
#include <string>
#include "utility.h"
#include "glog/logging.h"
#include <stdio.h>


namespace Optimizer
{

int load_file(
       const std::string& path,
       const std::string& file,
       int& sample_nums,
       int feat_num,
       Sparsematrix& dataset,
       std::vector<int>* labels) {
    std::string fn;
    fn = path + "/" + file;

    std::ifstream fin(fn.c_str(), std::ios_base::in);

    if (fin.fail()) {
        PLOG(WARNING) << "load file [" << fn << "] failed!";
        return -1;
    }
   

    // consider label
    //++feat_num;

    std::string line;
    int pos_num = 0;
    std::string spliter = " ";
    while(getline(fin, line)) {
        std::vector<std::string> words;
        if (0 != string_split(line, spliter, words)
             /*|| feat_num != words.size()*/) {
            PLOG(WARNING) << "line [ "<< line << "] is not meet specifications";
            continue;
        }
        //add logic for data
        ++sample_nums;

        std::vector<std::string>::iterator iter = words.begin();
        //add label
        int lab = atoi((*iter).c_str());
        labels->push_back(lab);
        if (1 == lab) {
            //logout.info() << "samlpe is " << sample_nums << std::endl;
            //logout.info() << "pos sample label is  " << labels->at(sample_nums-1) << " lab is " << lab << std::endl;
            ++pos_num;
        }
        ++iter;
        //add feats
        //DBvec one_sample = new std::vector<double>;
        dataset.start_idx.push_back(dataset.sparse_matrix.size());
        for (; iter != words.end(); ++iter) {
            std::string::size_type p;
            p = (*iter).find(":");
            int feat_id = atoi((*iter).substr(0, p).c_str());
            dataset.sparse_matrix.push_back(feat_id);
        }
        dataset.end_idx.push_back(dataset.sparse_matrix.size());
    }

    logout.info() << "get [ "<< pos_num << " ]pos sample" << std::endl;
    logout.info() << "load file succ with sparse_matrix size :" << dataset.sparse_matrix.size()  << std::endl;

    fin.close();

    return 0;
}

int string_split(
       const std::string& origin,
       const std::string& spliter,
       std::vector<std::string>& words) {
    if (0 == origin.size() || 0 == spliter.size()) {
        PLOG(WARNING) << "string split error, either origini string or spliter is null";
        return -1;
    }

    std::string::size_type p, q;
    p = 0;
    q = origin.find(spliter);

    while (std::string::npos != q) {
        words.push_back(origin.substr(p, q - p));
        p = q + spliter.size();
        q = origin.find(spliter, p);
    }

    if (origin.size() != p) {
        words.push_back(origin.substr(p));
    }


    return 0;
}

double dot_product(DBvec a, DBvec b, double c) {
    if (a->size() != b->size()) {
        PLOG(ERROR) << "dotproduct error ,size not equal!";
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < a->size(); ++i) {
        sum += (a->at(i) * c * b->at(i));
    }
    return sum;
}

int vec_minus_to(DBvec a, DBvec b, DBvec c) {
    if (a->size() != b->size() || a->size() != c->size()) {
        PLOG(ERROR) << "vec_minus_to error, size not equal!";
        return -1;
    }
    
    for (int i = 0; i < a->size(); ++i) {
        c->at(i) = a->at(i) - b->at(i);
        //if(c->at(i) != 0){
            //logout.info() << " _ylist is 0 , a is " << a->at(i) << " b is " << b->at(i) << " c is " << c->at(i) << std::endl;
        //}
    }
    return 0;
}

int scale(DBvec a, double b) {
    for (int i = 0; i < a->size(); ++i) {
        a->at(i) *= b;
        //if(isnan(a->at(i))){
            //logout.info() << "_newX or _newGrad is nan , a is " << a->at(i) << " b is " << b << std::endl;
        //}
    }
    return 0;
}

int add_scale_vec(DBvec a,DBvec b, double c) {
    if (a->size() != b->size()) {
        PLOG(ERROR) << "add_scale_vec error ,size not equal!";
        return 0.0;
    }
    for (int i = 0; i < a->size(); ++i) {
        a->at(i) += c * b->at(i);
        //if(isnan(a->at(i))){
            //logout.info() << "_newX or _newGrad is nan , a is " << a->at(i) << " b is " << b->at(i) << " c is " << c << std::endl;
        //}
    }
    return 0;
}

int sum_two_scale_vec(DBvec a,DBvec b, DBvec c, double d) {
    if (a->size() != b->size() || a->size() != c->size()) {
        PLOG(ERROR) << "sum_two_scal_vec error ,size not equal!";
        return 0.0;
    }
    for (int i = 0; i < a->size(); ++i) {
        a->at(i) =  b->at(i) + d * c->at(i);
    }
    return 0;
}

int compute_signa(
         const Sparsematrix& sample,
         const int& start,
         const int& end,
         bool positive_sample,
         DBvec new_para,
         double& h_sita,
         double& loss) {
    if (start < 0 || start > end) {
        PLOG(ERROR) << "worker got error parameter size!";
        return -1;
    }

    double sum = 0.0;
    for (int i = start; i < end; ++i) {
        //logout.info() << "feat id is:" << sample.sparse_matrix.at(i)  << std::endl;
        sum += new_para->at(sample.sparse_matrix.at(i));
    }

    if (! positive_sample) { sum *= -1.0;}
    if (sum < -300) {
        h_sita = 0;
        loss += -sum;
    } else if (sum > 300) {
        h_sita = 1;
    } else {
        double temp = 1.0 + exp(-sum);
        h_sita = 1.0 / temp;
        loss += log(temp);
    }
    return 0;
}

int compute_fo_deriv(
           DBvec fo_deriv, 
           const double& h_sita,
           const Sparsematrix& sample,
           const int& start,
           const int& end,
           bool positive_sample){
    if (start < 0 || start > end) {
        PLOG(ERROR) << "worker got error fo_deriv size! start:["
                    << start << "],end[" << end << "], fo_deriv size ["
                    << fo_deriv->size() << "]";
        return -1;
    }

    for (int i = start; i < end; ++i) {
         //fo_deriv->at(i) = (h_sita - 1.0) * sample->at(i);
        if (positive_sample) {
            fo_deriv->at(sample.sparse_matrix.at(i)) += (h_sita - 1.0);
        } else {
            fo_deriv->at(sample.sparse_matrix.at(i)) -= (h_sita - 1.0);
        }
    }

    return 0;
}


void swap_ptr(void* a, void* b) {
    if (a == b) {
        PLOG(WARNING) << "swap same ptr!";
    }
    void* temp = a;
    a = b;
    b = temp;
}

bool abs_cmp(const double& a,const double& b) {
    return fabs(a) > fabs(b);
}

}
