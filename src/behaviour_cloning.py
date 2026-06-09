import os, argparse
import numpy as np
import gym
import ray
from env_without_user_feedback import p2penv
from sota_policy import SOTAPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.agents.ddpg import DEFAULT_CONFIG

# tf1, tf, tfv = try_import_tf(error=True)
torch, _ = try_import_torch(error=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="Training RL agent with p2penv.")
parser.add_argument("--datapath", help="Path to find the data.", type=str, default="/home/jlavoie/master/data")
parser.add_argument("--gpu", help="Include GPU in data generation.", type=float, default=0.0)
args = parser.parse_args()


# Setup policy and rollout workers
config = DEFAULT_CONFIG.copy()

config["framework"] = "torch"
# config["timesteps_total"] = 96
config["rollout_fragment_length"] = 32
# config["train_batch"] = 8
# config["train_batch_size"] = 16
config["output"] =  args.datapath + "/sota_policy/"
config["output_compress_columns"] = ["obs", "new_obs"]
# config["output_max_file_size"] = 64 * 1024 * 1024
config["horizon"] = 16
# config["seed"] = 123
config["batch_mode"] = "truncate_episodes"
config["input_evaluation"] = ["is"]
# config["stop"] = 96
num_cpus = 6
num_worker = 2
cpus_per_worker = int(num_cpus/num_worker)
config["num_cpus_per_worker"] = cpus_per_worker

if args.gpu:
	config["num_gpus"] = 0.1 if args.gpu else 0
	config["num_gpus_per_worker"] = (args.gpu - 0.2)/float(num_worker)

env_config = {}
env_config["data_path"] = args.datapath
env_config["framework"] = "torch"
env_config["render"] = False
env_config['state_shape'] = [128,128]
# env_config["seed"] = config["seed"]
env_config["data_aug"] = True
env_config["episode_horizon"] = config["horizon"]
env_config["dataset"] = "mnist" #ray.tune.grid_search(["mnist", "cifar10"])
env_config["model"] = "axonmap"
env_config["implant_type"] = "argus"
env_config["rho"] = 200 #ray.tune.grid_search([150, 250, 350])
env_config["rotation"] = 0 #ray.tune.grid_search([-65, -45])
env_config["axlambda"] = 500 #ray.tune.grid_search([150, 250, 350])
env_config["decay"] = False
env_config["obs_with_step_num"] = True #ray.tune.grid_search([True, False])
env_config["coordconv"] = True #ray.tune.grid_search([True, False])
env_config["electrode_activation_level"] = 0.0 #ray.tune.choice([0.0, 0.2 ,0.5, 1.0])
env_config["agent_model"] = "iccv" # Just to keep track of experiments

env_config["overused_electrode_penalty"] = "" #ray.tune.grid_search(["", "ponderate", "negative"])
env_config["action_type"] = "continuous"
# env_config["n_action_bundle"] = ray.tune.choice([1,5,60])
# ASHA scheduler stop trial based on absolute value of the reward
# env_config["reward_scale"] = ray.tune.randint(1,1000)
env_config["reward_distance"] = "patch" #ray.tune.grid_search(["euclidean","wasserstein"])
# env_config["n_patch"] = ray.tune.grid_search([21,])
env_config["sigmoid_reward"] = "" #"sigmoid" #ray.tune.grid_search(["", "sigmoid", "centered_sigmoid"])

# env = gym.make("CartPole-v0")
env = p2penv(env_config)


policy = SOTAPolicy(env.observation_space, env.action_space, env_config) # 
workers = WorkerSet(
	trainer_config=config,
	policy_class=SOTAPolicy,
	# env_creator=lambda c: gym.make("CartPole-v0"),
	env_creator=lambda c: p2penv(env_config),
	num_workers=num_worker)


if __name__ == '__main__':
	ray.init(num_cpus=num_cpus, num_gpus=args.gpu, include_dashboard=False, ignore_reinit_error=True)
	
	# Broadcast weights to the policy evaluation workers
	# weights = ray.put({"default_policy": policy.get_weights()})
	# for w in workers.remote_workers():
	# 	w.set_weights.remote(weights)
	# weights = ray.put({DEFAULT_POLICY_ID: policy.get_weights()})
	# for w in workers:
	# 	w.set_weights.remote(weights)

	# Gather a batch of samples
	samples = ray.get([w.sample.remote() for w in workers.remote_workers()])
	[print(s) for s in samples]

	# [s.__setitem__("weights",np.ones(config["rollout_fragment_length"])) for s in samples]
	# [print(s.columns(["weights"])) for s in samples]

	T1 = SampleBatch.concat_samples(samples)

	# Improve the policy using the T1 batch
	#policy.learn_on_batch(T1)


	ray.shutdown()
