import ray
import os, sys, pickle5, pickle, argparse, time, copy
from PIL import Image
import numpy as np
from env_without_user_feedback import p2penv
from resnet_model import ResnetVisionNetwork
# from iccv_resnet import ICCVResnetVisionNetwork

from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
#from ray.rllib.models.tf.visionnet import VisionNetwork SHOUD IT BE FROM TENSORFLOW ?
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()
from torchvision import models
models.resnet18(pretrained=True)

def experiment_or_result(unspecified_path):
	if [i for i in os.listdir(unspecified_path) if "checkpoint" in i]:
		result_path,result_basename = os.path.split(os.path.normpath(unspecified_path))
		result_path += "/"
		return result_path,result_basename
	else:
		return unspecified_path,[]

def get_experiment_dir(root_path, agent_type, overwrite = False):
	# Search for experiment
	exp_directories = [i for i in os.listdir(root_path) if (agent_type.upper() + "_" in i) and os.path.isdir(root_path + i)]
	if not exp_directories:
		root_path, result_basename = experiment_or_result(root_path)
		exp_directories.append(result_basename)
	
	exp_directories.sort()
	print(root_path)
	print(exp_directories)
	completed_exp_path = {}
	# Check all directory with checkpoint and no .png
	for i in exp_directories:
		# check if checkpoint files exist
		exp_files = os.listdir(root_path + i)
		checkpoint_path = [i for i in exp_files if "checkpoint" in i]
		checkpoint_path.sort()
		# Check if image sample already generate
		if overwrite:
			image_sample_path = []
		else:
			image_sample_path = [i for i in exp_files if ".png" in i]

		if checkpoint_path and not image_sample_path:
			# generate complete path for checkpoint
			checkpoint_content = os.listdir(root_path + i + "/" + checkpoint_path[-1])
			complete_checkpoint = checkpoint_path[-1] + "/checkpoint-" + str(int("".join(filter(str.isdigit,checkpoint_path[-1]))))
			completed_exp_path[i] = complete_checkpoint
	return completed_exp_path


def generate_result(completed_exp_path, root_path, n_sample, agent_type):
	
	def generate_sample(ref_img, output_img_path, env_config, sota_sample=False):
		sample_env = p2penv(env_config)
		sample_env.reset()
		if sota_sample:		
			trans = torchvision.transforms.Compose([
				torchvision.transforms.ToPILImage(),
				torchvision.transforms.Resize(sample_env.electrode_matrice_shape),
				torchvision.transforms.ToTensor()])
			sota_action = trans(ref_img)
			action = sota_action.numpy().squeeze().ravel()
		else:
			action = np.random.rand(*sample_env.electrode_matrice_shape).squeeze().ravel()
		print(action)
		sample_step = []

		for i in range(int(sample_env.episode_horizon)):
			obs, reward, status, info = sample_env.step(action)
			sample_step.append([obs,reward,status, info, action])
			action[np.argmax(action)] = 0.0
		sample_name = output_img_path[:-4] + "_sota" + ".png" if sota_sample else output_img_path[:-4] + "_rand" + ".png"
		sample_env.save_episode_sample(file = sample_name, canvas_only = True)
		sample_distance_l2,sample_distance_mse,sample_distance_ssim,sample_distance_kldiv,_,_ = sample_env.get_metrics()
		return info["distance_norm"],sample_distance_l2,sample_distance_mse,sample_distance_ssim,sample_distance_kldiv
	
	root_path,_ = experiment_or_result(root_path)
	# iterate trought all directory loading params
	for exp,cp in completed_exp_path.items():
		exp_path = root_path + exp + "/"
		print(exp_path)
		print("Generating samples for : {}".format(exp_path))
		file = open(exp_path + "params.pkl", "rb")
		content = pickle5.load(file)
		file.close()
		[print("{}:{}".format(i,j)) for i,j in content.items()]

		# Config depend on ray version. Solution is to check with which version experiment used and use it.
		config = copy.deepcopy(content) #DEFAULT_CONFIG.copy()

		config["env_config"]["render"] = True
		config["env_config"]["data_aug"] = False
		# config["env_config"]["dataset"] = "cifar10"
		config["env_config"]["data_path"] = exp_path
		# config["env_config"]["n_action_bundle"] = 
		# Retrocompatibility
		# if agent_type:
		# 	config["explore"] =  False
		config["env_config"]["agent_model"] = config["env_config"]["agent_model"] if "agent_model" in config["env_config"] else config["Q_model"]["custom_model"]
		config["num_workers"] = 1
		config["num_gpus"] = 0
		config["framework"] = "torch"

		config["observation_space"] = [42,42,6]
		config["action_space"] = [60]
		config.pop("observation_space")
		config.pop("action_space")
		# config.pop("input_config")
		def env_creator(env_config):
		    return p2penv(env_config)
		register_env("p2penv", env_creator)
		if config["env_config"]["agent_model"] == "resnet18":
			# Pretrained model
			# ModelCatalog.register_custom_model("resnet18", ResnetVisionNetwork if config["framework"] == "torch" else VisionNetwork)
			print("Agent with resnet not implemented")
		else:
			# try:
			# 	agent
			# except:
			# ModelCatalog.register_custom_model("iccvresnet18", ICCVResnetVisionNetwork)
			if agent_type == "ddpg":
				print("Instantiating {} agent...".format(agent_type.upper()))
				agent = DDPGTrainer(env=p2penv, config=config)
			elif agent_type == "sac":
				print("Instantiating {} agent...".format(agent_type.upper()))
				agent = SACTrainer(env="p2penv", config=config)


			print("Restoring agent...")
			agent.restore(exp_path + cp)
			[print("{}: {}".format(k,v)) for k,v in config["env_config"].items()]
			
			print("Instantiating environment...")
			
			env = p2penv(config["env_config"])

		print("Generating samples...")
		
		distance_norm = []
		date_tag = time.strftime("%Y_%m_%d-%H-%M-%S")
		for i in range(n_sample):
			print("Reseting environment...")
			obs = env.reset()
			print("Generating sample {}".format(i))
			done = False
			while not done:
				action = agent.compute_single_action(obs)
				print("SHAPE: {}, {}".format(action.shape, np.unique(action)))
				obs, reward, done, info = env.step(action)
			output_path = exp_path + "sota_animate_eps_{}/".format(date_tag)
			print("Results can be found at: {}".format(output_path))
			if not os.path.exists(output_path):
				os.mkdir(output_path)
			output_img_path = output_path + "eps_n{}_y_{}_i{}".format(env.episode_horizon,env.ref_y,i)+".png"
			env.save_episode_sample(output_img_path)
			sample_distance_wass,sample_distance_l2,sample_distance_mse,sample_distance_ssim,sample_distance_kldiv = generate_sample(env.canvas[:,:,0], output_img_path, config["env_config"])
			sota_distance_wass,sota_distance_l2,sota_distance_mse,sota_distance_ssim,sota_distance_kldiv = generate_sample(env.canvas[:,:,0], output_img_path, config["env_config"], sota_sample=True)
			distance_l2,distance_mse,distance_ssim,distance_kldiv,_,_ = env.get_metrics()
			distance_norm.append([i,env.ref_y.numpy(),
								info["distance_norm"],
								distance_l2, 
								distance_mse, 
								distance_ssim, 
								distance_kldiv, 
								sota_distance_wass, 
								sota_distance_l2, 
								sota_distance_mse, 
								sota_distance_ssim, 
								sota_distance_kldiv,
								sample_distance_wass, 
								sample_distance_l2,
								sample_distance_mse,
								sample_distance_ssim,
								sample_distance_kldiv])
		distance_norm = np.asarray(distance_norm)
		np.savetxt(output_path+"distance_{}_{}".format(env.reward_distance, date_tag)+".csv", distance_norm, delimiter=",", header="iter,label,wass_distance,distance_l2,distance_mse,distance_ssim,distance_kldiv,sota_distance_wass,sota_distance_l2,sota_distance_mse,sota_distance_ssim,sota_distance_kldiv,sample_distance_wass,sample_distance_l2,sample_distance_mse,sample_distance_ssim,sample_distance_kldiv")
		print("Cleaning training variables...")
		agent.cleanup()
		del env

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Generating episode samples.")
	parser.add_argument("--path", help="Path to experiments.", default="/home/jlavoie/ray_results/SAC/")
	parser.add_argument("--nsample", help="Number of samples to generate.", type=int, default=5)
	parser.add_argument("--overwrite", help="Rerun animate even if img already exist. Can results in overwrite of sample.", type=int, default=1)
	parser.add_argument("--agent", help="Agent algorithm name  found in experiment name.", default="sac")
	args = parser.parse_args()

	print("Using Ray version {}".format(ray.__version__))

	completed_exp_path = get_experiment_dir(root_path = args.path, overwrite = args.overwrite, agent_type=args.agent)
	if completed_exp_path:
		print(completed_exp_path)
		ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False)
		generate_result(completed_exp_path=completed_exp_path, root_path=args.path, n_sample=args.nsample, agent_type=args.agent)

		ray.shutdown()
	else:
		print("No valid experiment can be found.")
