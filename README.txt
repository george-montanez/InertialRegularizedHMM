This repository contains Python (2.7.X) code for an inertial regularized Gaussian hidden
Markov model, useful for segmentation of multivariate time series data.

The code depends on the following libraries for Python: scipy, numpy and scikit-learn.

The repository contains the following files:
=============================================
1. GaussianHMM.py - This is the file for the GaussianHMM class. It can be run
from the command-line by itself, and contains a minimal example for
segmentation of a multivariate time series. Contains parameter settings at the
top of the file, such as maximum number of EM iterations (MAX_ITER),
parameters for controlling the minimum probabilities for transitions, initial
states and emission covariance (MIN_START_PROB, MIN_TRANS_PROB, MIN_COV_VAL --
assists with numerical instability issues), and a parameter for controlling
the desired minimum Gini coefficient (MIN_GINI).

2. main.py - Contains code for creating instances of GaussianHMM and
segmenting the data files included in the Datasets directory. Uses
evaluation.py for computing the metrics on how well each time series is
segmented compared to the ground truth labels. Results are output to folders
created in the current working directory. 

3. evaluation.py - Contains functions for evaluating the agreement between the
true and predicted label sets, for several different metrics.

4. Dataset.zip - Zipped collection of 100 multivariate (45D) time series of
human activity accelerometer data. Unzip to create a "Dataset" folder in the
current working directory.

5. gini.py - Helper code for estimating th Gini coefficient of a set.

To run:
Make sure you have write permission to the InertialRegularizedHMM directory
and python, scipy, numpy and scikit-learn installed. 

From the command line, change directory to InertialRegularizedHMM and unzip
the Datasets.zip file to the current directory. Then type:
  python main.py
This command will run an inertialized Gaussian HMM segmentation process on the
dataset and write the results to standard input and to created subdirectories.

