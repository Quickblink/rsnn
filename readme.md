TODO (in no order):
1. Display Architecture(s)
2. Train to forget: Reset during processing
3. Setup Thesis latex document
4. make bibtex list of referenced papers


"Spiking Neuron with Discontinous Integration" from "Algorithm and Hardware Design of Discrete-Time Spiking Neural Networks Based on Back Propagation with Binary Activations"

Naming conventions:
environments, scenarios, experiments


Thoughts about POMDP:
1. Environments can be made a POMDP by artificially hiding information. This way,
one environment can serve to create different scenarios.
2. Data Creation: Environments can serve to cheaply create (unlimited) data for supervised tasks (see below). While in
    reinforcement learning, data is usually sampled by invoking the learner, this doesn't have to be the case.

Types of Challenge for an environment (POMDP):
1. Supervised Prediction of hidden information when following some fixed policy/teacher
2. Supervised Prediction of action from a given policy/teacher (data is sampled from learner)
3. Reinforcement Learning using hidden state estimation from type 1 learner
4. Full Reinforcement Learning without teacher

Types of Neuron:
1. Static LIF Neuron
2. Adaptive Neuron: Threshold gets increased by fix amount everytime it fires
3. Magic Neuron: Threshold is a sum over time of dedicated input, similar to the membrane potential
4. FlipFlop Neuron: Has positive and negative threshold. Firing brings the neuron into one of two states: Constantly firing or not firing.
5. NoReset Neuron: Like LIF Neuron, but doesn't reset membrane potential when firing. It just keeps firing until membrane
potential is lowered by negative inputs
6. Cooldown Neuron: Similar to previous, however: Changes to membrane potential can only be positive, needs to be leaky
 to stop firing.

Types of Networks:
1. Feed Forward: Has no recurrent connection but can carry hidden state through potential and the explicit methods of the respective neurons.
Notably, this gives the network the possibility to remember the range of inputs it has already seen. However, it cannot edit the memory based on its content
(because of missing recurrent connections) and thus fails to remember temporal information about inputs e.g. in what order past inputs have been observed.
2. Recurrent: At least one layer takes its own or spikes from a following layer as input. Theoretically this allows forming of a complicated memory, but
suffers from instabilities, for example because of vanishing or exploding gradients.

Simulation Details:
1. Synapses: Following Chris, I disabled Synapses at the beginning. However, they seem to have stabilizing effects on the network.
2. Sequential vs. pipelined: The different layers of an SNN (including synapses and membranes) can be
simulated sequentially (meaning the output of each layer is immediately available for the next layer) or pipelined
(meaning all layers compute "at once" using the output of the previous layer from the previous timestep). While
the latter theoretically allows efficient implementation on neuromorphic/parallel hardware, information traverses
the network with latency. This is especially important, if the agent aims to make quick decisions.


Stability and longterm memory usage

