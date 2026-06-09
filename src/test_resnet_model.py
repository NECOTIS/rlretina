from resnet_model import ResnetVisionNetwork
from visionnet import CustomVisionNetwork
from env_without_user_feedback import p2penv
import torch, argparse

parser = argparse.ArgumentParser(description="Training RL agent with p2penv.")

parser.add_argument("--test", help="Training time in hours.", type=int, default=0)

args = parser.parse_args()

env_config = {}
env_config["data_aug"] = True
env_config["render"] = False
env_config["model"] = "scoreboard"
if args.test :
    env_config["data_path"] = "/workspace/master/data"
    env_config["prosthesis_path"] = "/workspace/rlretina/data/electrodesPositions.mat"
else:
    env_config["data_path"] = "/home/jlavoie/master/data"
    env_config["prosthesis_path"] = "/home/jlavoie/rlretina/data/electrodesPositions.mat"

env_config["episode_horizon"] = 50
env_config["dataset"] = "mnist"
env_config["decay"] = False
env_config["job_id"] = 0
env_config["obs_with_step_num"] = True #ray.tune.grid_search([True, False])
env_config["coordconv"] = True #ray.tune.grid_search([True, False])

env = p2penv(env_config)

print(env.observation_space)

model = CustomVisionNetwork(env.observation_space, env.action_space, 288, {}, "resnet18")

# inputs = torch.randn(64,3,84,84)
inputs = torch.from_numpy(env.observation_space.sample())
print("INPUT SHAPE:{}".format(inputs.shape))
inputs = torch.stack((inputs,inputs),dim=0)
out = model.forward({"obs":inputs},0,0)
print(model)
print(out)
print(out[0].size())
assert out[0].shape == (1,288)
