#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include <server.h>
#include <worker.h>
#include <utility.h>


DEFINE_int32(bfgsm, 10, "iterator for L-BFGS");
DEFINE_int32(iters, 80, "iterator for gradient descent");
DEFINE_string(file, "", "train data or validation file");
DEFINE_double(c1, 0.01, "parameter for L1 reg-term");
DEFINE_double(c2, 0.0, "parameter for L2 reg-term");
DEFINE_int32(feats, 10000, "feat_num for train");

using namespace Optimizer;

int main(int argc, char **argv){

    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "./log";
    google::ParseCommandLineFlags(&argc, &argv, true);

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (0 == world_rank) {
        Server* master = new Server(FLAGS_bfgsm, FLAGS_iters, FLAGS_c1, FLAGS_c2, FLAGS_file, FLAGS_feats, MPI_COMM_WORLD);
        master->run(); 
    } else {
        Worker* slave = new Worker(FLAGS_iters, FLAGS_file, FLAGS_feats, MPI_COMM_WORLD);
        slave->run();
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
