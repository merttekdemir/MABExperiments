# MAB_Experiments
This repository implements a framework for efficient simulations of different online learning algorithms for the Multi Armed Bandit Problems in the stochastic setting. 
Majority of the algorithms included in the experiments can be found in:

> Orabona, Francesco. A Modern Introduction to Online Learning. vol. 5, arXiv:1912.13213v5, https://doi.org/10.48550/arXiv.1912.13213.

## Requirements
The packages and additional resources needed to run the project are contained in the `Project.toml` file. It is possible to create a virtual environment from its specifications using `Pkg` and `instantiate`.

## Problem Statement
Experiments in Multi Armed Bandit problems are prone to combinatorically explode. 
Since algorithms are designed to minimize some regret benchmark in the long run, simulations of the algorithms require many iterations.
Moreover, since the bounds on the regrets are calculated in expectation, to accurately assess the results of each simulation it must be repeated many times.
Considering also the plethora of algorithms and corresponding hyperparameters one may wish to test, the need for efficient simulation of MAB games becomes clear.
As a result the purpose of this repository is to create an efficient framework for testing such online learning algorthims.
Moreover, as much as possible, we have tried to design the repository such that it can easily be extended to new algorithms, and new game settings (eg. Bandits with context).

## Repository Overview

### Multi-Armed Bandit Game
The MAB game is introduced as a `mutable struct` within `"src/MABStruct.jl"`. 
This struct defines a single Multi Armed Bandit Game. 
Currently the Struct is designed to support bandit and full-information stochastic games, however it can be easily extended by including the necessary fields.

The games work by running serially one iteration of the game at a time. At each iteration a sample is drawn from each of the arms according to the defined probability distributions of each arm, representing the reward from that arm in that iteration. This realization is used to update the relevant parameters of the game, which in turn is used by the chosen online learning function to determine the policy vector for the next iteration. This is repeated until the game terminates at a predetermined number of iterations.

### Online Learning Algorithms
The collection of online learning algorithms considered are as a `module` within `"src/OnlineLearningAlgorithms.jl"`. The algorithms are implemented such that given an iteration of the game, they receive the necessary inputs to update the policy vector for the next time step and return said policy vector. In this way they can directly interface with the `MABStruct`, as all the information needed to determine the new policy can be found within its fields. 

### Experiments
The script to launch an experiment can be found in `"src/MABExperiments.jl"`. Given the users configuration, which can be modfied through `"src/configuration.yml"`, the script is a self-contained way of running the games on the user defined learning algorithms for a given number of iterations per game and game-algorithm pair. The script also handles plotting the user specified fields of the `MABStruct`, once again through the configuration file.

### Plotting
The plotting functionality is handled by `"src/MABPlots.jl"`. Given the results of an Experiment the script handles plotting time series and histograms of some of the game's attributes. The outputs of the plots may be saved in a path specified in the configuration file or displayed directly.
