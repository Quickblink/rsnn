The code offers two modes of execution:
1. Executing configurations. (Preferred)
Configurations can be created with the help of `notebooks/make_configs.ipynb`. They are saved in the configs folder and hold both parameters and results for each experiment. A config named `myconfig.json` is executed in the main directory with
   `sh run_config.sh myconfig`. It requires only a working nvidia-docker installation to run and installs required container and dataset data on demand. Results can be plotted using `notebooks/plot_results.ipynb`.
2. Notebooks.
For both experiments there is a notebook that can be used directly. It should work with any standard PyTorch installation.