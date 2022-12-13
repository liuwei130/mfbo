# Python Code for ***Multi-fidelity Max-value Entropy Search (MF-MES)***
This page provides a python implementation of MF-MES (Takeno et al., ICML2020) and its extensions. 
The code can reproduce the result of our paper that extends the conference version of MF-MES (Takeno et al., Neural Computation, 2022).


# Environment
* Linux (CentOS 7.8.2003 (Core) /  Ubuntu 20.04.1 LTS)
* Python 3.6.10 ([Anaconda3-5.2.0](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh))
* Additional packages are as follows. (All packages are in requirements.txt)
    * GPy==1.9.6
    * nlopt==2.6.1
    * torch==1.10.1
    * torchvision==0.11.2
* Fortran (gfortran)

# Organization
* experiments
    * Files for trace-aware and multi-fidelity BO experiments exist
* scripts
    * `vanillaBO' includes usual BO scripts
    * `MultiFidelityBO' include sequential, parallel, and trace-aware multi-fidelity BO
    * `myutils' include util scripts
    * `test_functions' include several test functions used in the paper

# Instruction

## Installation
1. Download or clone this code
2. (check the environment and required packages)
3. Install our code as the package: ``pip install -e <path>``

## run experiments

### In `experiments_multifidelity':
* Methods
    * Sequential method names
        * MFMES_RFM
        * MFPES
        * BOCA
        * MFSKO
        * MES_RFM
    * Asynchronous method names
        * Parallel_MFMES_RFM
        * Parallel_MES_RFM
        * AsyncTS
        * GP_UCB_PE
        * MES_LP
    * Synchronous method names
        * Sync_MFMES_RFM
        * Sync_MES_RFM

* For the Benchmark function experiments (Benchmark_name: Styblinski_tang, Branin, HartMann3, HartMann4, HartMann6, and SVM)
    * For 10 parallel experiments of different seeds (0 ~ 9):
        * python bayesopt_exp.py method_name Benchmark_name -1 0 parallel_num
    * If you run an experiment with one specific seed:
        * python bayesopt_exp.py method_name Benchmark_name seed 0 parallel_num
    * For the plots of the experimental results:
        * python plot_results_benchmarks_wctime.py

* For the Synthetic function experiments
    * For 10 times 10 parallel experiments (10 generated functions and 10 initial samplings) of different seeds (0 ~ 9):
        * python bayesopt_exp.py method_name SynFun -1 -1 parallel_num
    * If you run the experiment with one specific function-generation seed, and initialization seed:
        * python bayesopt_exp.py method_name SynFun initial_seed function_seed parallel_num
    * For the plots of the experimental results:
        * python plot_results_synthetic_wctime.py

* The material experiments can't be executed in this code because the data is private.

### In `experiments_ta':
* Methods
    * Sequential method name
        * MFMES_RFM
    * Trace-aware method names
        * TAMFMES_RFM
        * TA_KG

* For the Benchmark function experiments (Benchmark_name: cnn_mnist and cnn_cifar10)
    * For 10 parallel experiments of different seeds (0 ~ 9):
        * python ta_bayesopt_exp.py method_name Benchmark_name -1 0 1
    * If you run an experiment with one specific seed:
        * python ta_bayesopt_exp.py method_name Benchmark_name seed 0 1
    * For the plots of the experimental results:
        * python plot_results_benchmarks_wctime.py

* For the Synthetic function experiments
    * For 10 times 10 parallel experiments (10 generated functions and 10 initial samplings) of different seeds (0 ~ 9):
        * python ta_bayesopt_exp.py method_name SynFun_for_ta -1 -1 1
    * If you run the experiment with one specific function-generation seed, and initialization seed:
        * python ta_bayesopt_exp.py method_name SynFun_for_ta initial_seed function_seed 1
    * For the plots of the experimental results:
        * python plot_results_synthetic_wctime.py

## Reference 
Multi-fidelity Bayesian optimization with max-value entropy search and its parallelization.  
Takeno, S., Fukuoka, H., Tsukada, Y., Koyama, T., Shiga, M., Takeuchi, I., and Karasuyama, M.  
In International Conference on Machine Learning (ICML2020), pages 9334–9345. PMLR. 

A Generalized Framework of Multi-fidelity Max-value Entropy Search through Joint Entropy.  
Takeno, S., Fukuoka, H., Tsukada, Y., Koyama, T., Shiga, M., Takeuchi, I., and Karasuyama, M.  
Neural Computation, 2022, 34 (10): 2145–2203.
