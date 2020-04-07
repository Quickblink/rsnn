Naming conventions:
environments, scenarios, experiments


Thoughts about POMDP:
1. Environments can be made a POMDP by artificially hiding information. This way,
one environment can serve to create different scenarios.
2. Data Creation

Types of Challenge for an environment (POMDP):
1. Supervised Prediction of hidden information along when following some fixed policy/teacher
2. Supervised Prediction of action from a given policy/teacher (trajectories are sampled from learner)
3. Reinforcement Learning using hidden state estimation from type 1 learner
4. Full Reinforcement Learning without teacher

Types of Neuron:
1. Static LIF Neuron
2. Adaptive Neuron: Threshold gets increased by fix amount everytime it fires
3. Magic Neuron: Threshold is a sum over time of dedicated input, similar to the membrane potential
4. FlipFlop Neuron: Has positive and negative threshold. Firing brings the neuron into one of two states: Constantly firing or not firing.
5. NoReset Neuron: 

Types of Networks:
1. Feed Forward: Has no recurrent connection but can carry hidden state through potential and the explicit methods of the respective neurons.
Notably, this gives the network the possibility to remember the range of inputs it has already seen. However, it cannot edit the memory based on its content
(because of missing recurrent connections) and thus fails to remember temporal information about inputs e.g. in what order past inputs have been observed.
2. Recurrent: At least one layer takes its own or spikes from a following layer as input. Theoretically this allows forming of a complicated memory, but
suffers from instabilities, for example because of vanishing or exploding gradients.

Simulation Details:
1. Synapses: Ignored for now. What's the insight here?
2. Sequential vs. pipelined: The different layers of an SNN (including synapses and membranes) can be
simulated sequentially (meaning the output of each layer is immediately available for the next layer) or pipelined
(meaning all layers compute "at once" using the output of the previous layer from the previous timestep). While
the latter theoretically allows efficient implementation on neuromorphic/parallel hardware, information traverses
the network with latency. This is especially important, if the agent aims to make quick decisions.


Stability and longterm memory usage

