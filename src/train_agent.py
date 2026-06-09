import os, sys, copy, argparse
from env_without_user_feedback import p2penv
import ray

from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

parser = argparse.ArgumentParser(description="Training RL agent with p2penv.")
parser.add_argument("-jobid", help="Saving job id in env_config for further debuging.")

args = parser.parse_args()


env_config = {}
env_config["render"] = False
env_config["data_path"] = "/home/jlavoie/master/data"
env_config["episode_horizon"] = 100
env_config["job_id"] = args.jobid

ray.init()
# "evaluation_num_workers": 2, "evaluation_interval": 1,"num_framestacks": 0,
config = copy.deepcopy(DEFAULT_CONFIG)
config["env_config"] = env_config
config["num_gpus"] = 1 #int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["framework"] = "torch"
#config["log_level"] = "DEBUG"
config["remote_worker_envs"] = True
config["num_envs_per_worker"] = 8
config["num_workers"] = 8
config["num_gpus_per_worker"] = 1.0/8.0
config["Q_model"]["fcnet_hiddens"] = [512, 512]
config["policy_model"]["fcnet_hiddens"] = [512, 512]
config["n_step"] = 4
config["target_network_update_freq"] = 32
config["train_batch_size"] = 64
config["target_entropy"] = 'auto'
config["tau"] = 1.0
config["target_network_update_freq"] = 8000
config["prioritized_replay"] = true
config["learning_starts"] = 100000
#config["initial_alpha"] = 0.8

# stop = {}
# stop["training_iteration"] = 100

agent = SACTrainer(env=p2penv, config=config)
agent_path_checkpoint = agent.save()
agent_logdir = agent.logdir

print(agent_path_checkpoint)
print(agent_logdir)

train_iter = 5000

for i in range(train_iter):
        sys.stdout.write("////////// In iterations number {} ////////// \n".format(i))
        result = agent.train()
        print(pretty_print(result))
        if not ((i+1) % int(train_iter/2)):
            agent_path_checkpoint = agent.save()


# results = tune.run("SAC", config=config,stop=stop, checkpoint_at_end=True)

agent_logdir = agent.logdir
print(agent_path_checkpoint)
print(agent_logdir)
print("Realized {} iterations".format(i))

env_config["render"] = True
env = p2penv(env_config)

for i in range(3):
    state = env.reset()
    done = False
    while not done:
        action = agent.compute_action(state)  # key line; get the next action
        state, reward, done, _ = env.step(action)
    exp_tag = os.path.dirname(agent_path_checkpoint)
    env.save_animation(exp_tag+"_episode_n{}_y{}".format(env.episode_horizon,env.ref_y)+".mp4")

print(agent.get_policy().model.base_model.summary())
