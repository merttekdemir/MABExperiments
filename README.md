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

## Brief Introduction to Multi Armed Bandits
The traditional introduction to the multi-arm bandit problem follows the analogy of slot machines in a casino. Imagine you have 100 dollars and walk into a casino with 10 slot machines. Your goal is to maximize the amount of money you walk out from the casino with, however all the slot machines are rigged in different ways unknown to you. Do you choose to spend all you money playing on one machine? Or perhaps you prefer to reserve a portion of you money for trying out all the machines first, then continuing to play on only the one that provided the lowest loss (best reward). 

For a more modern and practical applications consider serving advertisements on social media. The platform must choose an advertisement to display to the user given a set of possible advertisements and past experiences of displaying those advertisements to the user. The advertiser will observe a binary loss depending on if the user clicked on the advertisement or not, however will not be able to observe the counterfactual losses had the advertiser selected a different advertisement.

In general, the multi-arm bandit problem is part of a broader learning model called Online Convex Optimization (OCO). OCO can be viewed as a game between a learner and an adversary. The learner can take any action from an action space $K$, which is a fixed compact convex set determined before the game starts. The game then proceeds for $T$ rounds for some integer $T$.

In each round $τ=1,\ldots,T$:

    1. Learner selects $i_τ \in K$
    2. The adversary picks a convex loss function $f_τ: K \rightarrow R$
    3. The player suffers loss $f_t(i_t)$ and observes some information about $f_t$


Depending on the power of the adversary there are several possible setting:

Stochastic setting: $f_1,\ldots,f_T$ are i.i.d samples of a fixed distribution

Oblivious adversary setting: $f_1,\ldots,f_T$ are arbritary but decided before the game starts (i.e. independent of the player's actions)

Adaptive adversary setting: For each $t$, $f_t$ depends on $i_1,\ldots,i_τ$

Depending on the feedback given to the learner there are also several possible settings:

Full information setting: player observes $f_τ$

 Bandit setting: player only observes $f_τ(i_τ)$


In studying such OCO problems the goal is to design an online decision making algorithm to help the learner choose good actions. The quality of the learns actions is measured through the notion of regret. For a time step $T$ this measures the difference between the cumulative losses the learner has observed from realized actions and the loss the learner could have observed by playing the best fixed action. The goal of the learner is to minimize regret:

```math
\R_T = \sum_{τ=1}^T f_τ(i_τ)- \min_{i^*\in[K]} \sum_{τ=1}^T f_τ(i^*)
```

The OCO settings listed above study the trade-off between exploration and exploitation. In particular, at each time step $t$ actions can be thought of as associating to two key outputs; the loss suffered (reward gained) and feedback received about the action by having taken that action. Thus, to minimize regret the learner must devise a strategy balancing gaining information for future exploitation and exploiting his learning's.
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
