CS7641 - Machine Learning Github Repository (https://github.com/ezerilli/Machine_Learning)

SETTING UP THE ENVIRONMENT

The following steps leads to setup the working environment for CS7641 - Machine Learning 
in the OMSCS program.

Installing the conda environment is a ready-to-use solution to be able to run python 
scripts without having to worry about the packages and versions used. Alternatively you 
can install each of the packages in requirements.yaml on your own independently with pip.

1. Start by installing Conda for your operating system following the 
instructions [here](https://conda.io/docs/user-guide/install/index.html).

2. Now install the environment described in requirements.yaml:
```bash
conda env create -f requirements.yaml
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

7. During the semester I may need to add some new packages to the environment. So, in
the case you have previously installed this environment, you just need to update it:
```bash
conda env update -f requirements.yaml
```

ASSIGNMENT1 - SUPERVISED LEARNING

Assignment 1 aims to explore some algorithms in Supervised Learning, perform model 
complexity analysis and learning curves while comparing their performances on two 
interesting datasets: the Wisconsin Diagnostic Breast Cancer (WDBC) and the Handwritten 
Digits Image Classification (the famous MNIST).

The assignment consists of two parts: 

- experiment 1, producing validation and learning curves, with performances on the test 
set, for each of the algorithms in the the Wisconsin Diagnostic Breast Cancer (WDBC).

- experiment 2, producing validation and learning curves, with performances on the test 
set, for each of the algorithms in the he Handwritten Digits Image Classification (MNIST).

In order to run the assignment, you just need to move to supervised learning directory and 
then execute the experiments.py script:
```bash
cd cd Supervised\ Learning/
python experiments.py
```
Figures will show up progressively. It takes a while to perform all the experiments 
and hyperparameter optimizations. However, they have already been saved into the 
images directory. Theory, results and experiments are discussed in the report.


ASSIGNMENT2 - RANDOMIZED OPTIMIZATION

Assignment 2 aims to explore some algorithms in Randomized Optimization, namely Random-Hill 
Climbing, Simulated Annealing, Genetic Algorithms, MIMIC and Gradient Descent, while comparing 
their performances on 3 interesting discrete and 1 continuous optimisation problems: 
the Travel Salesman Problem, Flip Flop, 4-Peaks and Neural Network training.

The assignment consists of four parts: 

- experiment 1, producing complexity and performances curves for the Travel Salesman problem.
- experiment 2, producing complexity and performances curves for the Flip Flop problem.
- experiment 2, producing complexity and performances curves for the 4-Peak problem.
- experiment 2, producing complexity and performances curves for the Neural Network problem.

In order to run the assignment, you just need to move to randomised optimization directory and 
then execute the experiments.py script:
```bash
cd Randomized\ Optimization/
python experiments.py
```
Figures will show up progressively. It takes a while to perform all the experiments 
and parameters optimizations. However, they have already been saved into the 
images directory. Theory, results and experiments are discussed in the report.

