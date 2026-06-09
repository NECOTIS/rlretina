#!/bin/python
import sys, time, argparse, pickle
from env_without_user_feedback import p2penv
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=180)
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
torch.set_printoptions(profile="full")
import torchvision
from PIL import Image


parser = argparse.ArgumentParser(description="Training RL agent with p2penv.")
parser.add_argument("--test", help="Training time in hours.", type=int, default=0)
parser.add_argument("--heavy", help="Test env for heavy use.", type=int, default=0)
args = parser.parse_args()


def summary_stats(step, file="patch_episode_trajectory_.png"):
    obs_mean = np.mean([i[0] for i in step])
    obs_std = np.std([i[0] for i in step])
    obs_max = np.max([i[0] for i in step])
    obs_min = np.min([i[0] for i in step])
    reward = [i[1] for i in step]
    distance = [i[3]["distance_norm"] for i in step]
    mse = [i[3]["mse_end"] for i in step]
    kl = [i[3]["kl_end"] for i in step]
    ssim_metric = [i[3]["ssim_end"] for i in step]
    wass_metric = [i[3]["wass_end"] for i in step]
    patch_metric = [i[3]["patch_end"] for i in step]
    action = [i[4] for i in step]
    print(reward)
    frq_reward,edges_reward = np.histogram(reward)
    frq_distance,edges_distance = np.histogram(distance)
    frq_action,edges_action = np.histogram(action)
    print("Distance: {}".format(edges_distance))
    print("Occurence: {}".format(frq_distance))
    print("Action: {}".format(edges_action))
    print("Occurence: {}".format(frq_action))
    print("Reward: {}".format(edges_reward))
    print("Occurence: {}".format(frq_reward))
    print("Reward mean: {}".format(np.mean(reward)))
    fig = plt.figure()
    plt.plot(range(len(reward)), reward)
    plt.show()
    plt.savefig(file[:-4] + "_reward" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(distance)), distance)
    plt.show()
    plt.savefig(file[:-4] + "_distance" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(mse)), mse)
    plt.show()
    plt.savefig(file[:-4] + "_mse" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(kl)), kl)
    plt.show()
    plt.savefig(file[:-4] + "_kl" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(ssim_metric)), ssim_metric)
    plt.show()
    plt.savefig(file[:-4] + "_ssim_metric" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(wass_metric)), wass_metric)
    plt.show()
    plt.savefig(file[:-4] + "_wass_metric" + file[-4:])
    fig = plt.figure()
    plt.plot(range(len(patch_metric)), patch_metric)
    plt.show()
    plt.savefig(file[:-4] + "_patch_metric" + file[-4:])
    return obs_mean,obs_std,obs_max,obs_min

env_config = {}
env_config["render"] = True
env_config["env_render_itself"] = True
env_config["rho"] = 200
# env_config["rotation"] = -65
env_config["axlambda"] = 500
env_config["data_aug"] = True
env_config["all_data"] = True

if args.test :
    env_config["data_path"] = "/workspace/git_repo/rlretina/data"
    env_config["results_path"] = "/workspace/git_repo/rlretina/data/results"
    env_config["prosthesis_path"] = "/workspace/git_repo/rlretina/data/electrodesPositions.mat"
else:
    env_config["data_path"] = "/home/jlavoie/master/data"
    env_config["results_path"] = "/home/jlavoie/rlretina/data/results"
    env_config["prosthesis_path"] = "/home/jlavoie/rlretina/data/electrodesPositions.mat"

env_config["episode_horizon"] = 16
env_config["model"] = "axonmap"
env_config["implant_type"] = "argus"
env_config["non_linear_reward"] = ""
env_config["dataset"] = "mnist" # "cifar10"
env_config["action_type"] = "continuous"
env_config["n_action_bundle"] = 1
env_config["obs_with_step_num"] = True
env_config["coordconv"] = True
env_config["reward_distance"] = "patch"
env_config["reward_input_scale_factor"] = 1.0
env_config["overused_electrode_penalty"] = ""
env_config["reward_scale"] = 200 

t0 = time.time()
print("########## Random action sample ##########")
env = p2penv(env_config)
obs = env.reset()
print("Observation shape: {},Action shape: {}".format(env.observation_space, env.action_space))
t1 = time.time()

step = []

for i in range(int(env.episode_horizon)):
	action = env.action_space.sample()
	obs, reward, status, info = env.step(action)
	step.append([obs,reward,status, info, action])
t2 = time.time()

print("Time elapsed for instanciation: {}".format(t1-t0))
print("Time elapsed generating data with {} steps: {}".format(env_config["episode_horizon"],t2-t1))

obs_mean,obs_std,obs_max,obs_min = summary_stats(step, file="../data/test_env/episode_patch_trajectory_"+env_config["implant_type"]+".png")
# loss = env.loss(torch.Tensor(obs),torch.Tensor(obs))

print("Obs mean should be near 0 not {}".format(obs_mean))
print("Obs std should be near 1 not {}".format(obs_std))
assert obs_max <= 1, "Obs should be less or equal to 1 not {}".format(obs_max)
assert obs_min >= 0, "Obs should be greater or equal to 0 not {}".format(obs_min)
# print(loss)
# assert loss == torch.Tensor([0.0]), "Loss should be greater or equal to 0 not {}".format(loss)


env.save_episode_sample("../data/test_env/test_env_sample.png")

print(len(env.animation))

print("########## SOTA sample ##########")

env.reset()
env.save_episode_sample("../data/test_env/sota_init_patch_env_sample_.png")

action_grid = torch.arange(np.prod(env.electrode_matrice_shape))
action_grid = torch.reshape(action_grid, tuple(env.electrode_matrice_shape))

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(env.electrode_matrice_shape),
    torchvision.transforms.ToTensor()])
ref = env.canvas[:,:,0].cpu()
sota_action = trans(ref)


# sota_action = sota_action > 0.85
# print(sota_action*action_grid)
# Image.fromarray(np.uint8(sota_action.numpy().squeeze()*255), 'L').save("sota_resize_orig.png")
# sota_action = torch.unique(action_grid * sota_action)[1:]
# print(sota_action)
# sota_action = sota_action[torch.randperm(sota_action.size(dim=0))[:int(env.episode_horizon)]]
# Precise electrode
# sota_action = torch.ones(int(env.episode_horizon), dtype=int)
# Diagonal
# sota_action = torch.diagonal(action_grid)
# Horizontal line
# sota_action = torch.zeros(env.electrode_matrice_shape)
# sota_action[0,:] = 1
# Vertical line
# sota_action = torch.zeros(env.electrode_matrice_shape)
# sota_action[:,0] = 1
# sota_action = torch.arange(0, env_config["episode_horizon"])

# print(sota_action.numpy())
index_test = 0
step = []

# assert len(sota_action) >= int(env.episode_horizon), "Not enough pixel for episode horizon"
action = sota_action.numpy().squeeze().ravel()
print(action.shape)

for i in range(int(env.episode_horizon)):
    obs, reward, status, info = env.step(action)
    step.append([obs,reward,status, info, action])
    action[env.action_bundle] = 0
    if index_test:
        env.save_episode_sample("../data/test_ibionics_index/sota_env_sample_" + str(i) + ".png")
        #env.reset()
# env.save_episode_sample(file = "../data/test_env/sota_env_patch_sample_.png")
env.save_animation("../data/test_env/sota_stim_pattern.mp4")

obs_mean,obs_std,obs_max,obs_min = summary_stats(step, "../data/test_env/sota_patch_episode_trajectory_"+env_config["implant_type"]+".png")

if args.heavy:
    print("########## Heavy use ##########")
    step = []
    for i in range(30):
        env.reset()
        for i in range(int(env.episode_horizon)):
            action = env.action_space.sample()
            obs, reward, status, info = env.step(action)
            step.append([obs,reward,status, info, action])
            # action[env.action_bundle] = 0
        # if index_test:
        #     env.save_episode_sample("../data/test_ibionics_index/ones_env_sample_" + str(i) + ".png")
            #env.reset()
    # env.save_episode_sample(file = "../data/test_env/ones_env_patch_sample_.png")
    # env.save_animation("../data/test_env/ones_stim_pattern.mp4")
    obs_mean,obs_std,obs_max,obs_min = summary_stats(step, "../data/test_env/brute_test_episode_trajectory_"+env_config["implant_type"]+".png")

print("########## All action sample ##########")
env_config["all_data"] = False
env_config["reward_input_scale_factor"] = 1.0
env = p2penv(env_config)
obs = env.reset()
reward_mapping_step = []
action = np.zeros(env.electrode_matrice_shape, dtype=float).ravel()
index_test = 1

for i in range(len(action)):
    action[i] = 1.0
    obs, reward, status, info = env.step(action)
    _, _, _, info = env.step(action)
    reward_mapping_step.append([obs,reward,status, info, action.astype(int)])
    if index_test:
        env.save_episode_sample("../data/test_ibionics_index/reward_mapping_sample_" + str(i) + ".png")
    action[i] = 0.0
    obs = env.reset()

print("Saving all action sample...")

reward_mapping_results_file = open("../data/test_env/reward_mapping_results.txt", "wb")
pickle.dump(reward_mapping_step, reward_mapping_results_file)
reward_mapping_results_file.close()
# env.save_episode_sample(file = "../data/test_env/ones_env_patch_sample_.png")
# env.save_animation("../data/test_env/ones_stim_pattern.mp4")
obs_mean,obs_std,obs_max,obs_min = summary_stats(reward_mapping_step, "../data/test_env/reward_mapping_episode_trajectory_"+env_config["implant_type"]+".png")
