# Theory of Markov State Modeling

## Introduction
Markov state models (MSMs) are used to analyze and understand the kinetics of protein conformational dynamics based on molecular dynamics (MD) simulations. In this context, MSMs are a type of statistical model that describe the transitions between different conformations (often referred to as states) over time by means of memoryless jumps. MD simulations provide atomistic details about the movement of the protein over time, but the resulting data can be difficult to analyze and interpret, especially when the protein undergoes complex conformational changes. Therefore, MSMs can be used to reduce the complex MD data to a simpler, more manageable representation of the protein's conformational dynamics, which allows understanding of the biological process of interest.

In an MSM, the states are defined as discrete clusters of protein conformations and the transitions between states are modeled as a Markov process. This means that the probability of transition from one state to another depends only on the current state and not on the history of previous states. The resulting MSM can be used to calculate various quantities, such as the rate constants for transitions between states and the equilibrium populations of each state.

Overall, MSMs are a useful tool for studying the kinetics of protein conformational changes in MD simulations, providing a simplified representation of complex protein dynamics that can be used to gain insight into the underlying mechanisms of protein behavior.

## Theoretical Background
Markov state models are mathematical models that describe the transition probabilities between discrete states in a system of discrete time steps. The basic equation for a Markov state model is the following

$$P(x_t = j | x_{t-1} = i) = T_{ij}$$

where $T_{ij}$ is the transition probability from state $i$ to state $j$ at time $t$, and $P(x_t = j | x_{t-1} = i)$ is the probability of the system being in state $j$ at time $t$, given that it was in state $i$ at time $t-1$.

The transition probabilities are usually estimated from time-series data, such as molecular dynamics simulations, and can be organized into a transition matrix $T$

$$T = \{T_{ij}\}$$

where $T_{ij}$ is the $(i, j)$-th element of the matrix. The transition matrix defines the probability of moving from any one state to any other state in a single time step.

The stationary distribution, $\pi$, is a probability distribution over the states that satisfies the following equation

$$\pi T = \pi$$

where the left-hand side is the distribution after one time step and the right-hand side is the distribution at the current time step. The stationary distribution represents the long-term behavior of the system and can be used to calculate various quantities, such as the equilibrium populations of each state.

Further information can be found (freely available) on the following website, which provides a good introduction to the topic [docs.markovmodel.org](http://docs.markovmodel.org/) or the following article ["Markov models of molecular kinetics: Generation and validation" by Prinz et al.](https://doi.org/10.1063/1.3565032).
