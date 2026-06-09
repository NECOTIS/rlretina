#TODO
# Handle seed for both research algo and agent. (https://github.com/NECOTIS/torch-reproducible-block/blob/master/test.py)
# Try TuneBOHB and HyperbanforBOHB (https://docs.ray.io/en/master/tune/api_docs/suggestion.html#ax-tune-suggest-ax-axsearch)
import os, sys, copy, argparse
#import GPUtil
from env_without_user_feedback import p2penv
# from resnet_model import ResnetVisionNetwork
from visionnet import CustomVisionNetwork
from DistanceCallback import DistanceCallback

import ray
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining


tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()
from torchvision import models



# Preloading pretrained model on ssh node that have an internet access
models.resnet18(pretrained=True)

parser = argparse.ArgumentParser(description="Training RL agent with p2penv.")
parser.add_argument("--jobid", help="Saving job id in env_config for further debuging.", default=0)
parser.add_argument("--commitid", help="Saving commit id in env_config for further debuging.", type=str, default="0")
parser.add_argument("--trainingtime", help="Training time in hours.", type=float, default=20.0)
parser.add_argument("--datapath", help="Path to find the data.", type=str, default="/home/jlavoie/master/data")
parser.add_argument("--resultspath", help="Path to find the results.", type=str, default="/home/ray/ray_results")
parser.add_argument("--test", help="Training time in hours.", type=int, default=0)
parser.add_argument("--ncpu", help="CPU available.", type=int, default=24)
parser.add_argument("--gpu", help="GPU available.", type=float, default=0.0)
parser.add_argument("--name", help="Name of the experience.", type=str, default="hpalgo_agent_commit")
args = parser.parse_args()

# Experiment config
n_gpus_available = args.gpu

if args.test:
    n_cpus_available = 6
else:
    n_cpus_available = args.ncpu
config = copy.deepcopy(DEFAULT_CONFIG)


ray.init(num_cpus=n_cpus_available, num_gpus=n_gpus_available, include_dashboard=False)

print(ray.available_resources())

config["framework"] = "torch"
# config["input"] = {
#                     "sampler":0.5,
#                     args.datapath + "/sota_policy/output-2022-03-24_21-23-35_worker-1_0.json":0.5 # Very small data with good weight
                    # args.datapath + "/sota_policy/output-2022-03-24_21-00-02_worker-1_0.json":0.2 # Smaller data with good weight
                    # args.datapath + "/sota_policy/output-2022-03-07_21-18-37_worker-3_0.json":0.2 # Older suppose to work
                    # args.datapath + "/sota_policy/output-2022-03-17_15-57-20_worker-2_0.json":0.2 # Big new
                    # }
config["input"] = ["/home/jlavoie/projects/def-eplourde/jlavoie/rlretina/data/sota_policy/output-2022-03-22_15-25-56_worker-1_0.json",
                        "/home/jlavoie/projects/def-eplourde/jlavoie/rlretina/data/sota_policy/output-2022-03-22_15-25-57_worker-2_0.json",
                        "/home/jlavoie/projects/def-eplourde/jlavoie/rlretina/data/sota_policy/output-2022-03-22_20-11-10_worker-1_0.json",
                        "/home/jlavoie/projects/def-eplourde/jlavoie/rlretina/data/sota_policy/output-2022-03-22_20-11-10_worker-2_0.json"]
# Use env actor to evaluate current policy
config["input_evaluation"] = ["simulation"]
config["evaluation_config"] = {"input": "sampler"}

# config["seed"] = 42 #ray.tune.grid_search([42,63,123])
config["observation_filter"] = "NoFilter"
config["initial_alpha"] =  1.0 # ray.tune.choice([0.2,0.9,1.0])
config["n_step"] = 1 #ray.tune.grid_search([1,3])
config["target_network_update_freq"] = 1 #ray.tune.grid_search([0,1,100])
config["horizon"] = 16
# Source of actor crashing sooner ? 
config["train_batch_size"] = 32 #ray.tune.grid_search([32, 64, 256])
config["timesteps_per_iteration"] = 100
config["optimization"] = {
        "actor_learning_rate": 3e-4, #5e-3,
        "critic_learning_rate": 3e-4, #3e-4, #5e-3,
        "entropy_learning_rate": 3e-4, #1e-4
    }
config["grad_clip"] = 'norm'
config["target_entropy"] = 'auto'
config["tau"] =  0.005 #ray.tune.grid_search([0.01, 1.0])
config["prioritized_replay"] = True
config["rollout_fragment_length"] = 100 #ray.tune.grid_search([1, 16])
config["buffer_size"] = int(1e6)
config["learning_starts"] = 10000 #ray.tune.grid_search([100, 1000, 10000])
config["evaluation_interval"] = 100
config["evaluation_num_episodes"] = 5
#config["compress_observations"] = True
config["callbacks"] = DistanceCallback
config["log_level"] = "DEBUG" # "INFO"
config["ignore_worker_failures"] = True

# Environment config
env_config = {}
if args.test :
    env_config["data_path"] = "/workspace/git_repo/master/data"
    env_config["results_path"] = args.resultspath +"/"+ args.commitid[0:6] + args.name
    env_config["prosthesis_path"] = "/workspace/git_repo/rlretina/data/electrodesPositions.mat"
    # config["input"] = "/workspace/git_repo/rlretina/data/sota_policy/output-2022-03-17_14-51-56_worker-2_0.json"
    config["input"] = ["/workspace/git_repo/rlretina/data/sota_policy/output-2022-03-24_21-00-02_worker-1_0.json",
                        "/workspace/git_repo/rlretina/data/sota_policy/output-2022-03-24_21-00-02_worker-2_0.json"]
    config["evaluation_config"] = {"input": "sampler"}
    # config["input"] = {
    #                     "sampler":0.8,
    #                     "/workspace/git_repo/rlretina/data/sota_policy/output-2022-03-24_21-00-02_worker-1_0.json":0.2
    #                     # "/workspace/git_repo/rlretina/data/sota_policy/output-2022-03-17_15-57-20_worker-2_0.json":0.7
    #                     }
else:
    env_config["data_path"] = args.datapath
    env_config["results_path"] = args.resultspath +"/"+ args.commitid[0:6] + args.name
    env_config["prosthesis_path"] = args.datapath + "/electrodesPositions.mat"

env_config["render"] = False
env_config["seed"] =  ray.tune.grid_search([42, 123]) #config["seed"]
env_config["data_aug"] = True
env_config["episode_horizon"] = config["horizon"] #32 #ray.tune.grid_search([16, 32])
env_config["dataset"] = "mnist" #ray.tune.grid_search(["mnist", "cifar10"])
env_config["model"] = "axonmap"
env_config["implant_type"] = "argus"
env_config["rho"] = 200 #ray.tune.grid_search([150, 250, 350])
env_config["rotation"] = 0 #ray.tune.grid_search([-65, -45])
env_config["axlambda"] = 500 #ray.tune.grid_search([150, 250, 350])
env_config["decay"] = False
env_config["obs_with_step_num"] = True #ray.tune.grid_search([True, False])
env_config["coordconv"] = True #ray.tune.grid_search([True, False])
env_config["job_id"] = args.jobid
env_config["commit_id"] = args.commitid
env_config["electrode_activation_level"] = 0.5 #ray.tune.uniform(0.0,1.0)
env_config["overused_electrode_penalty"] = "" #ray.tune.grid_search(["", "ponderate", "negative"])
env_config["action_type"] = "continuous"
env_config["n_action_bundle"] = 1 #ray.tune.choice([1,2,15,60])
# ASHA scheduler stop trial based on absolute value of the reward
env_config["reward_scale"] = 200 #ray.tune.uniform(1,1000)
env_config["reward_distance"] = "patch" #ray.tune.grid_search(["euclidean","wasserstein"])
# env_config["n_patch"] = ray.tune.grid_search([21,])
env_config["sigmoid_reward"] = "" #"sigmoid" #ray.tune.grid_search(["", "sigmoid", "centered_sigmoid"])
env_config["agent_model"] = "mlp" # Just to keep track of experiments

# Model config
if env_config["agent_model"] == "resnet18":
    # Pretrained model
    ModelCatalog.register_custom_model("resnet18", ResnetVisionNetwork if config["framework"] == "torch" else VisionNetwork)
    config["Q_model"]["custom_model"] = "resnet18"
    config["policy_model"]["custom_model"] = "resnet18"
elif env_config["agent_model"] == "batchnormvisionnet":
    ModelCatalog.register_custom_model("batchnormvisionnet", CustomVisionNetwork if config["framework"] == "torch" else VisionNetwork)
    # config["_disable_preprocessor_api"] = True Not Converting to appropriate type if true
    # config["observation_space"] = [42,42,6]
    # config["action_space"] = [60]

    config["Q_model"]["custom_model"] = "batchnormvisionnet"
    config["policy_model"]["custom_model"] = "batchnormvisionnet"
    filters_42x42 = [[16, [4, 4], 2],[32, [4, 4], 2],[256, [11, 11], 1],]
    config["Q_model"]["custom_model_config"] = {"conv_filters":filters_42x42,
                                                "conv_activation":"relu",
                                                "no_final_linear":False,
                                                "vf_share_layers":True,
                                                "fcnet_hiddens":[256,256],
                                                "fcnet_activation":"relu",
                                                "post_fcnet_hiddens":[],
                                                "post_fcnet_activation": "tanh"
                                                    }
    config["policy_model"]["custom_model_config"] = {"conv_filters":filters_42x42,
                                                "conv_activation":"relu",
                                                "no_final_linear":False,
                                                "vf_share_layers":True,
                                                "fcnet_hiddens":[256,256],
                                                "fcnet_activation":"relu",
                                                "post_fcnet_hiddens":[],
                                                "post_fcnet_activation": "tanh"
                                                    }

    # config["Q_model"]["conv_filters"] = [[16, [4, 4], 2],[32, [4, 4], 2],[256, [11, 11], 1],]
    # config["policy_model"]["custom_model"] = "batchnormvisionnet"
    # config["policy_model"]["fcnet_activation"] = "relu"
    # config["policy_model"]["post_fcnet_activation"] = "tanh"
elif env_config["agent_model"] == "mlp":
    #MLP untrained model
    # Model options for the Q network(s). These will override MODEL_DEFAULTS.
    # The `Q_model` dict is treated just as the top-level `model` dict in
    # setting up the Q-network(s) (2 if twin_q=True).
    # That means, you can do for different observation spaces:
    # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
    # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
    #   -> post_fcnet
    # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
    #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
    # You can also have SAC use your custom_model as Q-model(s), by simply
    # specifying the `custom_model` sub-key in below dict (just like you would
    # do in the top-level `model` dict.
    config["Q_model"]["fcnet_hiddens"] = [512, 512, 512] #ray.tune.grid_search([[512, 512, 512], [1024, 1024, 1024], [2048, 2048, 2048]])
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    config["policy_model"]["fcnet_hiddens"] = config["Q_model"]["fcnet_hiddens"]
    # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
    # These are used if no custom model is specified and the input space is 2D.
    # Filter config: List of [out_channels, kernel, stride] for each filter.
    # Example:
    # Use None for making RLlib try to find a default filter setup given the
    # observation space.    
    # filters_42x42 = [[16, [8, 4], 2],[32, [8, 4], 2],[256, [11, 11], 1],]
    # config["Q_model"]["conv_filters"] = filters_42x42
    # config["policy_model"]["conv_filters"] = filters_42x42
    # config["policy_model"]["conv_filters"] = [[64, 3, 2], [128, 1, 3]]

    config["Q_model"]["fcnet_activation"] = "relu"
    config["Q_model"]["post_fcnet_hiddens"] = [128]
    config["Q_model"]["post_fcnet_activation"] = "tanh"

    config["policy_model"]["fcnet_activation"] = "relu"
    config["policy_model"]["post_fcnet_hiddens"] = [128]
    config["policy_model"]["post_fcnet_activation"] = "tanh"

def env_creator(env_config):
    return p2penv(env_config)
# env = env_creator(env_config)
register_env("p2penv", lambda env_config: p2penv(env_config))

config["env"] = p2penv
config["env_config"] = env_config
eval_env_config = copy.deepcopy(env_config)
eval_env_config["render"] = False
eval_env_config["env_render_itself"] = False
config["evaluation_config"]["env_config"] = eval_env_config



if  args.test:
    n_trial = 1.0
    config["num_workers"] = 1
    config["num_cpus_per_worker"] = int(((n_cpus_available-2)/n_trial)/config["num_workers"])
    config["num_envs_per_worker"] = 4
    config["remote_worker_envs"] = True
    # (1.0/2) * 0.3 + (3 * (1.0/2) * 0.6)/3 = 0.15 + 0.3 = 0.45
    # config["num_gpus"] = (n_gpus_available/n_trial) * 0.30 #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    # config["num_gpus_per_worker"] = ((n_gpus_available/n_trial)*0.60)/float(config["num_workers"])
    config["buffer_size"] = int(1e1)
    config["learning_starts"] = 100
    config["train_batch_size"] = 16
    perturbation_interval = 2
    num_samples = 1
elif not args.test and env_config["model"] == "scoreboard":
    config["num_workers"] = 4
    config["num_cpus_per_worker"] = int(((n_cpus_available-2)/n_trial)/config["num_workers"]) 
    config["num_envs_per_worker"] = config["num_cpus_per_worker"] - 1
    config["num_gpus"] = (n_gpus_available/n_trial) * 0.1 #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["num_gpus_per_worker"] = ((n_gpus_available/n_trial)-config["num_gpus"])/float(config["num_workers"])
    config["remote_worker_envs"] = True
elif not args.test and env_config["model"] == "axonmap":
    perturbation_interval = 10
    num_samples = 1
    n_trial = 2.0
    config["num_workers"] = 4
    config["num_cpus_per_worker"] = int(((n_cpus_available-2)/n_trial)/config["num_workers"]) 
    config["num_envs_per_worker"] = 8
    config["remote_worker_envs"] = True
    # (1.0/2) * 0.3 + (3 * (1.0/2) * 0.6)/3 = 0.15 + 0.3 = 0.45
    if n_gpus_available:
        config["num_gpus"] = (n_gpus_available/n_trial) * 0.30 #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        config["num_gpus_per_worker"] = ((n_gpus_available/n_trial)*0.60)/float(config["num_workers"])

stop = {}
stop["training_iteration"] = 50000
safe_training_time = int((args.trainingtime-0.25)*3600.0)

asha_scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="custom_metrics/distance_norm_mean",
    mode="min",
    max_t=3, # Number of epoch
    grace_period=2,
    reduction_factor=3
    )
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="custom_metrics/mse_end_mean",
    perturbation_interval=perturbation_interval,
    mode="max"
    )

def trainable(config):
    trainer = SACTrainer(env="p2penv", config=config)
    trainer._allow_unknown_configs = True
    # start = 0
    # if checkpoint_dir:
    #     with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
    #         state = json.loads(f.read())
    #         start = state["step"] + 1

    #     with ray.tune.checkpoint_dir(step=step) as checkpoint_dir:
    #         path = os.path.join(checkpoint_dir, "checkpoint")
    #         with open(path, "w") as f:
    #             f.write(json.dumps({"step": start}))

    #     ray.tune.report(hello="world", ray="tune")
    # trainer._allow_unknown_subkeys = trainer._allow_unknown_subkeys.append("data_path")
    # trainer=trainer.with_common_config(config)
    # return trainer
agent = "SAC"
results = ray.tune.run("SAC",
                        scheduler=pbt_scheduler,
                        name = args.commitid[0:8] + str(agent) + args.name,
                        config=config,
                        stop=stop,
                        local_dir=args.resultspath,
                        checkpoint_freq = 100,
                        checkpoint_at_end=True,
                        # log_to_file=True,
                        num_samples = num_samples,
                        max_failures=0,
                        # resume=True,
                        # resources_per_trial={"cpu": 6, "gpu": 1},
                        time_budget_s=safe_training_time)#, metric="episode_reward_mean", mode="max")


