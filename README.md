# Efficient and Scalable Graph Generation through Iterative Local Expansion

This repository contains the reference implementation of the paper [Efficient and Scalable Graph Generation through Iterative Local Expansion](https://arxiv.org/pdf/2312.11529).


## Setup

To get started, follow these steps:

+ Clone this repository

+ Create the specified [conda](https://docs.conda.io/en/latest/) environment named `graph-generation` by running the following command on Linux or macOS:

    ```bash
    conda env create -f environment.yaml
    ```

    and on Windows:

    ```bash
    conda env create -f environment_win.yaml
    ```

    Note that the `graph-tool` library is not available on Windows. This library is used for SBM graph evaluation, which will consequently not work on Windows. Everything else will work regardless of the operating system.


+ Activate the newly created environment:

    ```bash
    conda activate graph-generation
    ```

+ Navigate to `util/orca` and compile the `Orca` library:

    ```bash
    g++ -O2 -std=c++11 -o orca orca.cpp
    ```


## Usage

The main entry point is `main.py` with parameters managed by the [Hydra](https://hydra.cc/) framework.
To reproduce the results from the paper, run:

```bash
python main.py +experiment=XXX
```

where `XXX` is one of the following experiments:
`planar`, `sbm`, `tree`, `protein`, `point_cloud`, `planar_extrapolation`, `tree_extrapolation`, `planar_interpolation`, `tree_interpolation`.

To reproduce the results for our one-shot baseline, add the prefix `oneshot_` to the experiment name when running the same experiments, for example, `oneshot_planar`.

New experiments can be added by creating a new config file in `config/experiment/` or by passing the parameters directly through the command line. Please refer to the [Hydra documentation](https://hydra.cc/docs/intro/) for more information.


### Checkpoints

When `training.save_checkpoint` in the configuration is set to `True`, checkpoints are saved. To resume training from a checkpoint, set `training.resume` to the step number of the checkpoint, or to `True` to resume from the latest checkpoint.


### Wandb

To log the results to [Wandb](https://wandb.ai/), set `wandb.logging` to `True` in the configuration.


### Multirun

The `config/experiment/multirun_sample.yaml` file contains a sample configuration for launching multiple runs with hyperparameter sweeps. 


### Slurm

The `slurm/cluster/my_cluster.yaml` file contains a sample configuration for launching experiments using the [Slurm](https://slurm.schedmd.com/) workload manager. To launch experiment `XXX` on the cluster, adapt the `my_cluster.yaml` for your specific cluster and run:


```bash
python main.py +experiment=XXX +slurm=my_cluster
```


## Citation
When using this code, please cite our paper:
```
@misc{bergmeister2023efficient,
      title={Efficient and Scalable Graph Generation through Iterative Local Expansion}, 
      author={Andreas Bergmeister and Karolis Martinkus and Nathanaël Perraudin and Roger Wattenhofer},
      year={2023},
      eprint={2312.11529},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```
