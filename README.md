# owlqn-mpi
 OWLQN（L-BFGS add L1 reg-term） on mpi

start command:
  
    -f :host-file, record hostname of nodes in our mpi-cluster;
    -bfgsm : parameter for m in L-BFGS;
    -file : train or validtion file (must in put under ./data dir);
    -c1 : parameter for L1 reg-term;
    -c2 : parameter for L2 reg-term;
    -iters: num of iterator; 
    -feats: num of feature; 

  mpirun -n 3 -f slaves.txt ./owlqn -bfgsm=10 -file="xxxx" -c1=0.01 -c2=0.0 -iters=80 -feats=685569

result:

   using imdb dataset in [http://http://komarix.org/ac/ds/](http://http://komarix.org/ac/ds/ "dataset in this link")
 
   auc is :0.946087
