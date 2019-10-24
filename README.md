# CS7641 - Machine Learning Github Repository 
https://github.com/ezerilli/Machine_Learning

### SETTING UP THE ENVIRONMENT

The following steps lead to setup the working environment for [CS7641 - Machine Learning](https://www.omscs.gatech.edu/cs-7641-machine-learning) 
in the [OMSCS](http://www.omscs.gatech.edu) program. 👨🏻‍💻📚

Installing the conda environment is a ready-to-use solution to be able to run python scripts without having to worry 
about the packages and versions used. Alternatively, you can install each of the packages in `requirements.yml` on your 
own independently with pip or conda.

1. Start by installing Conda for your operating system following the instructions [here](https://conda.io/docs/user-guide/install/index.html).

2. Now install the environment described in `requirements.yaml`:
```bash
conda env create -f requirements.yml
```

4. To activate the environment run:
```bash
conda activate CS7641
```

5. Once inside the environment, if you want to run a python file, run:
```bash
python my_file.py
```

6. To deactivate the environment run:
```bash
conda deactivate
```

7. During the semester I may need to add some new packages to the environment. So, in the case you have previously 
installed this environment, you just need to update it:
```bash
conda env update -f requirements.yml
```

### ASSIGNMENT1 - SUPERVISED LEARNING

Assignment 1 aims to explore some algorithms in **Supervised Learning**, perform model complexity analysis and learning 
curves while comparing their performances on two interesting datasets: the **Wisconsin Diagnostic Breast Cancer (WDBC)** 
and the **Handwritten Digits Image Classification (the famous MNIST)**.

The assignment consists of two parts: 

- _experiment 1_, producing validation curves, learning curves and performances on the test set, for each of the 
algorithms, on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

- _experiment 2_, producing validation curves, learning curves and performances on the test set, for each of the 
algorithms, on the **Handwritten Digits Image Classification (MNIST)** dataset.

In order to run the assignment, you just need to move to the supervised learning directory, then execute `run_experiments.py`:
```bash
cd Supervised\ Learning/
python run_experiments.py
```
Figures will show up progressively. It takes a while to perform all the experiments and hyperparameter optimizations. 
However, they have already been saved into the images directory. Theory, results and experiments are discussed in the 
report (not provided here because of Georgia Tech's Honor Code).


### ASSIGNMENT2 - RANDOMIZED OPTIMIZATION

Assignment 2 aims to explore some algorithms in **Randomized Optimization**, namely _Random-Hill Climbing_ (RHC), _Simulated 
Annealing_ (SA), _Genetic Algorithms_ (GA) and _Mutual-Information Maximizing Input Clustering* (MIMIC), while comparing 
their performances on 3 interesting discrete optimisation problems: the _Travel Salesman Problem_, _Flip Flop_ and _4-Peaks_. 
Moreover, RHC, SA and GA will be compared to _Gradient Descent_ and _Backpropagation_ on a (nowadays) fundamental optimization 
problem: training complex _Neural Networks_.

The assignment consists of four parts: 

- _experiment 1_, producing complexity and performances curves for the _Travel Salesman_ problem.
- _experiment 2_, producing complexity and performances curves for _Flip Flop_.
- _experiment 3_, producing complexity and performances curves for _4-Peak_.
- _experiment 4_, producing complexity and performances curves for _Neural Networks_ training.

In order to run the assignment, you just need to move to randomised optimization directory and then execute `run_experiments.py`:
```bash
cd Randomized\ Optimization/
python run_experiments.py
```
Figures will show up progressively. It takes a while to perform all the experiments and parameters optimizations. 
However, they have already been saved into the images directory. Theory, results and experiments are discussed in the 
report (not provided here because of Georgia Tech's Honor Code). 

### ASSIGNMENT3 - UNSUPERVISED LEARNING

Work in progress ! 🔥🔥🔥


### ASSIGNMENT4 - MARKOV DECISION PROCESSES

Work in progress ! 🔥🔥🔥
