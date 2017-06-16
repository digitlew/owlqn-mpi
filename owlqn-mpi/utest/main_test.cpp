#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <utility.h>
#include <string>
#include <mpi.h>
#include <stdio.h>
#include <server.h>
#include <worker.h>

int main(int argc, char** argv) {

    //google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging("test");
    FLAGS_log_dir = "./log";
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


namespace Optimizer
{
/*
    TEST(load_train_data, init) {
        const std::string path = "./data";
        const std::string file = "Blanc__Mel.txt";
        int feat_num = 685562;
        int sample_nums = 0;
        Sparsematrix dataset;
        std::vector<bool> labels;          
       
        logout.info() << "utest:[load_file], starting....." << std::endl;
        int ret = load_file(path, file, sample_nums, feat_num, dataset, &labels);
    
        EXPECT_EQ(ret, 0);

        int pos_labels = 0;
        std::vector<bool>::iterator iter = labels.begin();
        for (; iter != labels.end(); ++iter) {
            if (*iter) {
                ++pos_labels;
            }
        }

        EXPECT_EQ(pos_labels, 824);

    }

    TEST(utility, vec_comptue) {
        DBvec a = new std::vector<double>(10, 1.0);
        DBvec b = new std::vector<double>(10, 1.0);
        DBvec c = new std::vector<double>(10, 1.0);

        double sum = dot_product(a, b);
        EXPECT_EQ(sum, 10);

        scale(a, 5.0);
        sum = dot_product(a, b);
        EXPECT_EQ(sum, 50);

        vec_minus_to(a, b, c);
        sum = dot_product(b, c);
        EXPECT_EQ(sum, 40);

        Sparsematrix sample;
        for(int i =0; i < 10; ++i) {
            sample.sparse_matrix.push_back(i);
        }
        int start = 2;
        int end = 8;
        bool label = false;
        DBvec para = new std::vector<double>(20, 0.01);
        double h_sita = 0.0;
        compute_signa(sample, start, end, label, para, h_sita);
        //EXPECT_EQ(h_sita, 0.4655);
  
    }
*/

    Server* master = new Server(10, 80, 0.01, 0.0, "Blanc__Mel.txt", 685569, NULL);

    Worker* slave = new Worker(80, "Blanc__Mel.txt", 685569, NULL);

    TEST(server, run) {
        master->_init_interal_parameter();
        slave->_init_interal_parameter();
        master->_load_test_data();
        //EXPECT_EQ(master->_sample_nums, 186414);
        slave->_load_train_data();
        //EXPECT_EQ(slave->_sample_nums, 186414);
        //mpi_bcast x
        master->_check_test_loss();
        for (int i = 0; i < master->_feat_nums; ++ i) {
            slave->_x->at(i) = master->_newX->at(i);
        }
        slave->_compute_loss_and_grad();
        double train_loss = slave->_loss;
        master->_assemble_train_loss(train_loss);
        slave->_compute_f_deriv();
        for (int i = 0; i < master->_feat_nums; ++ i) {
            master->_newGrad->at(i) = slave->_fo_deriv->at(i);
            master->_grad->at(i) = slave->_fo_deriv->at(i);
        }

        int count = 0;
        do {
            master->_generate_newton_direction();
            master->_correction_direction();

            //back tracking line search
            double old_value;
            double orig_dir_deriv = dot_product(master->_newton_deriv, master->_newGrad);
            double alpha = 1.0 / sqrt(dot_product(master->_newton_deriv, master->_newton_deriv));
            logout.info() << "alpha is: " << alpha << " orig is" << orig_dir_deriv << std::endl;
            bool flag = true;

            old_value = train_loss;
            do {

                master->_back_tracking_line_search(alpha);

                for (int i = 0; i < master->_feat_nums; ++ i) {
                    slave->_x->at(i) = master->_newX->at(i);
                }
                slave->_compute_loss_and_grad();
                train_loss = slave->_loss;
                master->_assemble_train_loss(train_loss);
                slave->_compute_f_deriv();
                
                for (int i = 0; i < master->_feat_nums; ++ i) {
                    master->_newGrad->at(i) = slave->_fo_deriv->at(i);
                }
                // bts_condition
                double c = 1e-4;
                double backoff = 0.5;
                // logic for bts
                double cond = c * alpha * orig_dir_deriv;
                if (train_loss <= old_value + cond) {
                    flag = false;
                }

                alpha *= backoff;

            } while(flag);

            master->_check_test_loss();
            for(int i = 0; i < 100; ++i) {
                logout.info() << "newX" << master->_newX->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "X" << master->_x->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "newgrad" << master->_newGrad->at(i) << std::endl;
            }
            for(int i = 0; i < 100; ++i) {
                logout.info() << "grad" << master->_grad->at(i) << std::endl;
            }
            

            master->_update_internal_parameter();
           

            ++count;

        } while(count < 200);
                
    }



}

