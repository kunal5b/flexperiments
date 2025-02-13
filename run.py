import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from custom_strategy import get_custom_strategy
import time


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml

def run(config_path="conf", config_name="mlp"):
    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name=config_name)
    start_time = time.time()
    # 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    # save_path = cfg['hydra']['run']['dir']

    # 2. Prepare your dataset
    # When simulating FL workloads we have a lot of freedom on how the FL clients behave,
    # what data they have, how much data, etc. This is not possible in real FL settings.
    # In simulation you'd often encounter two types of dataset:
    #       * naturally partitioned, that come pre-partitioned by user id (e.g. FEMNIST,
    #         Shakespeare, SpeechCommands) and as a result these dataset have a fixed number
    #         of clients and a fixed amount/distribution of data for each client.
    #       * and others that are not partitioned in any way but are very popular in ML
    #         (e.g. MNIST, CIFAR-10/100). We can _synthetically_ partition these datasets
    #         into an arbitrary number of partitions and assign one to a different client.
    #         Synthetically partitioned dataset allow for simulating different data distribution
    #         scenarios to tests your ideas. The down side is that these might not reflect well
    #         the type of distributions encounter in the Wild.
    #
    # In this tutorial we are going to partition the MNIST dataset into 100 clients (the default
    # in our config -- but you can change this!) following a independent and identically distributed (IID)
    # sampling mechanism. This is arguably the simples way of partitioning data but it's a good fit
    # for this introductory tutorial.
    traindatasets, valdatasets, testdataset = prepare_dataset(cfg.num_clients, attack_type=cfg.attack_type)

    # 3. Define your clients
    # Unlike in standard FL (e.g. see the quickstart-pytorch or quickstart-tensorflow examples in the Flower repo),
    # in simulation we don't want to manually launch clients. We delegate that to the VirtualClientEngine.
    # What we need to provide to start_simulation() with is a function that can be called at any point in time to
    # create a client. This is what the line below exactly returns.
    client_fn = generate_client_fn(
        traindatasets, valdatasets, cfg.num_classes, cfg.model, cfg= cfg)

    # 4. Define your strategy
    # A flower strategy orchestrates your FL pipeline. Although it is present in all stages of the FL process
    # each strategy often differs from others depending on how the model _aggregation_ is performed. This happens
    # in the strategy's `aggregate_fit()` method. In this tutorial we choose FedAvg, which simply takes the average
    # of the models received from the clients that participated in a FL round doing fit().
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    # where `{}` can be either `fit` or `evaluate`, depending on the FL stage. The final number of clients sampled is given by the formula
    # ``` # an equivalent bit of code is used by the strategies' num_fit_clients() and num_evaluate_clients() built-in methods.
    #         num_clients = int(num_available_clients * self.fraction_fit)
    #         clients_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```
    strategy = get_custom_strategy(model=cfg.model, cfg=cfg, testdataset=testdataset)

    # 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.5,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your workload)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
    )

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.

    # 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    # results_path = str(Path(save_path)) + "/" + getOutputFileName(cfg.model, cfg.attack_round, cfg.max_attack_ratio)

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    config = {"model": cfg.model, "attack_mode": cfg.attack_round, "attack_ratio": cfg.max_attack_ratio}
    results = {"history": history, "config": config}

    # save the results as a python pickle
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = time.time()
    print("Time taken:", end_time - start_time)
    return results

def getOutputFileName(model: str, attack_round: str, attack_ratio: float) -> str:
    return model + "_" + attack_round + "_" + str(int(attack_ratio * 100)) + ".pkl"
