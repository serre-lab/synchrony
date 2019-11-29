## Kura-Net
This directory contains material for backpropagating through the flow of a Kuramoto model for the purpose of image segmentation. 

## Prerequisites
All nets are coded in torch==1.1.0. Eventually, you should be able to run `pip install -r requirements.txt` in this repo's top directory in order to install everything you need. I hope I will be able add that soon.

## Running demo
Calling `python run.py` will train a Unet to segment images of 3 MNIST digits using Kuramoto dynamics. 
Calling `python run.py --name D_search` will run one version of the above experiment for each setting of the `time_weight` parameter as it is varied from 0 to 5.

## Running a simple experiments
By "simple experiment", we mean one in which a single model is trained for one set of hyperparameters. To design an experiment, add experiment details to the config file `experiments.cfg`. The expexperiment name should be the section header. Then, you can run your experiment by calling `python run.py --name <my_experiment_name>`. Here is a comprehensive list of experimental parameters. Please note that this project is early in development and not all parameter combinations have been tested or debugged: 

* data_name- The name of the folder containing your data in data_cifs/yuwei/osci_save/data
* segments- The number of segments in the data. For now, this is constant across the dataset
* exp_name- The name you choose for your experiment, which will also be the name of the directory storing results
* model_name- The name of the coupling network holding learned parameters
* device- 'cuda' for GPU, 'cpu' for cpu
* interactive- Set to True for interactive plotting
* show_every- Integer indicating period of visualization
* img_side- Side length of data. Make sure it matches the data controlled by `data_name`
* batch_size - Batch size
* train_epochs- Number of full passes through the training set
* time_weight: Degree of polynomial weighting for time averaging loss
* time_steps- Number of timesteps in Kuramoto dynamics
* record_teps: How many steps from the end of the dynamics to include in the loss
* anneal- Float between 0 and 1. Controls the slope of linear annealing on Kuramoto update rate. 0 for no annealing.
* phase_initialization: Distribution from which to draw initial phases for the dynamics. Choices are `random` for uniform, `fixed` for random but constant across training, `gaussian` for normal about 0, `categorical` for categorical per oscillator with 4 choices
* intrinsic_frequencies- Distribution from which to draw intrinsic frequencies. Choices are 'zero' for zero intrinsic frequencies and `gaussian` for normal about 0
* update_rate- Update step for Kuramoto dynamics
* learning_rate- Learning rate for coupling net
* sparsity_weight- L1 sparsity penalty on coupling matrix
* small_world- Makes couplings a small world graph
* num_cn- Number of connected neighbors in the lattice from which the small world graph is generated.
* critical_distance - Neighborhood for local coupling
* in_channels- Number of coupling net input channels
* start_filts- Number of first layer features (?)
* out_channels- Number of output features(?)
* split- Number of sections into which to divide the conv output to repeatedly apply the fully connected layers. Increasing this integer over 1 should save memory.
* depth- Number of layers in the net
* kernel_size- Size of the kernels in a layer. Possibly a tuple for multi-scale models, but this has not been implemented.

## Running a search experiment
By "search experiment", we mean one in which several models sent to difference devices are trained as one hyperparameter, called the `search_parameter`, is varied. The section header for such an experiment in the config file must end in '_search'. You must also specify a `sub_experiment`, which should be the section header for a simple experiment, and a `search_parameter`, which should be the name of one of the parameters listed above. The values this parameter will assume during the search should be provided under the key `search_values` in the form of a list as should the device indices corresponding to the machine you'd like each sub-experiment to run on.  
