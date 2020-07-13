

options = {
    'spkfn' : ['bellec', 'superspike'],
    'spkconfig' : [0, 1],
    'architecture' : ['1L', '2L'],
    'beta' : [0.95, 0.9],
    'control_neuron' : ['Disc', 'LIF', 'NoReset'],
    'mem_neuron' : ['Adaptive', 'Cooldown', 'NoReset'],
    'lr' : [1e-2, 1e-3],
    '1-beta' : [True, False],
    'decay_out': [True, False]
}

# 1 - beta implementation???

# do spkconfig

# do all options as literal types

# convert config to runtime (maybe different file?)


#deactivate certain combinations: decay_out: False and 1-beta: True


#experiments:
#- most original lsnn without rewiring on cloud
#- my duplicate of first experiment in pytorch (on cloud???)


like_bellec = {
    'spkfn' : 'bellec',
    'spkconfig' : 0,
    'architecture': '1L',
    'beta': 0.95,
    'control_neuron': 'LIF',
    'mem_neuron' : 'Adaptive',
    'lr' : 1e-2,
    '1-beta': True,
    'decay_out': True
}