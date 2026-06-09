from bz2 import compress
import os, time
import random
import pulse2percept as p2p
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from PIL import Image
import scipy.io as spio
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from layers import SinkhornDistance
from ibionicsElectrodeArray import iBionicsElectrodeArray
from pulse2percept.implants import ProsthesisSystem

from filelock import FileLock
## TODO

class p2penv(gym.Env):
	def __init__(self, env_config):
		# Defaults env parameters
		self.manual_seed = env_config["seed"] if "seed" in env_config else 42
		self.retina_model = env_config["model"] if "model" in env_config else "axonmap"
		self.implant_type = env_config["implant_type"] if "implant_type" in env_config else "argus"
		self.dataset = env_config['dataset'] if 'dataset' in env_config else "mnist"
		self.all_data = env_config['all_data'] if 'all_data' in env_config else True
		self.data_aug = env_config['data_aug'] if 'data_aug' in env_config else True
		self.non_linear_reward = env_config['non_linear_reward'] if 'non_linear_reward' in env_config else ""
		self.electrode_activation_level = env_config['electrode_activation_level'] if 'electrode_activation_level' in env_config else 1.0
		self.overused_electrode_penalty = env_config['overused_electrode_penalty'] if 'overused_electrode_penalty' in env_config else ""
		self.obs_with_step_num = env_config['obs_with_step_num'] if 'obs_with_step_num' in env_config else False
		self.coordconv = env_config['coordconv'] if 'coordconv' in env_config else False
		self.decay = env_config['decay'] if 'decay' in env_config else False
		self.action_type = env_config['action_type'] if 'action_type' in env_config else "continuous"
		self.n_action_bundle = int(env_config['n_action_bundle']) if 'n_action_bundle' in env_config else 1
		self.render_animation = env_config['render'] if 'render' in env_config else False
		self.env_render_itself = env_config['env_render_itself'] if 'env_render_itself' in env_config else False
		self.reward_distance = env_config['reward_distance'] if 'reward_distance' in env_config else "patch"
		self.reward_scale = env_config['reward_scale'] if 'reward_scale' in env_config else 1
		self.reward_input_scale_factor = env_config['reward_input_scale_factor'] if 'reward_input_scale_factor' in env_config else 1.0
		self.n_patch = env_config['n_patch'] if 'n_patch' in env_config else 6
		self.episode_horizon = float(env_config['episode_horizon']) if 'episode_horizon' in env_config else 16
		self.state_dim = env_config['state_dim'] if 'state_dim' in env_config else [42,42]
		self.rho = env_config["rho"] if "rho" in env_config else 200
		self.rotation = env_config["rotation"] if "rotation" in env_config else 0
		self.axlambda = env_config["axlambda"] if "axlambda" in env_config else 200
		self.resultspath = env_config["results_path"] if 'results_path' in env_config else ""
		# Seed
		if self.manual_seed:
			# https://stackoverflow.com/questions/63498865/how-do-i-make-ray-tune-run-reproducible
			if isinstance(self.manual_seed, dict):
				print("SEED:{}".format(self.manual_seed))
			random.seed(self.manual_seed)
			np.random.seed(self.manual_seed)
			torch.manual_seed(self.manual_seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed(self.manual_seed)

		# Implant and retinal models
		if self.implant_type == "custom":
			self.electrodes_positions = spio.loadmat(env_config["prosthesis_path"])
			self.electrodes_positions = self.electrodes_positions['electrodesdPosition']
			self.ibionics_implant = iBionicsElectrodeArray(self.electrodes_positions)
			self.implant = ProsthesisSystem(self.ibionics_implant)
			self.electrode_matrice_shape = self.ibionics_implant.shape
			#self.implant.plot()
		elif self.implant_type == "argus":
			self.implant = p2p.implants.ArgusII(rot=self.rotation)
			self.electrode_matrice_shape = [6,10]
		elif self.implant_type == "alphaams":
			self.implant = p2p.implants.AlphaAMS()
			self.electrode_matrice_shape = [42,42]
		# Sharper phosphene (decrease distance to the retina) is difficult for the agent
		if self.retina_model == "axonmap":
			axons_map_name = "axons{}_{}.pickle".format(self.rho, self.axlambda)
			with FileLock(axons_map_name + ".lock"):
				if self.implant_type == "custom":
					self.model = p2p.models.AxonMapModel(rho=self.rho, axlambda=self.axlambda, xrange=(-7, 7), yrange=(-7, 7))
				elif self.implant_type == "argus":
					self.model = p2p.models.AxonMapModel(rho=self.rho,
														axlambda=self.axlambda,
														xrange=(-12, 10),
														axon_pickle=axons_map_name,
														yrange=(-6, 7))
				self.model.engine = 'joblib'
				self.model.build()
		elif self.retina_model == "scoreboard":
				if self.implant_type == "custom":
					self.model = p2p.models.ScoreboardModel(rho=150, xrange=(-7, 7), yrange=(-7, 7))
				elif self.implant_type == "argus":
					self.model = p2p.models.ScoreboardModel(rho=150, xrange=(-2, 8), yrange=(-1, 7.5))                    
				self.model.engine = 'joblib'
				self.model.build()
		self.percept_t = transforms.Compose([transforms.Resize(size=self.state_dim),
											transforms.ToTensor()])
		# Data
		self.t = transforms.Compose([transforms.Grayscale(),
									transforms.Resize(size=self.state_dim),
									transforms.ToTensor()])
		self.data_t = transforms.Compose([transforms.Grayscale(),
									transforms.RandomAffine(degrees=(0,45), translate=(0.1,0.1), scale=(0.8, 1.2)),
									transforms.ColorJitter(brightness=0.2, contrast=0.2),
									transforms.Resize(size=self.state_dim),
									transforms.ToTensor() #,
									#transforms.Normalize(mean=0.13101, std=0.30854)
									])
		with FileLock(env_config["data_path"] + "torch_data.lock"):
			if self.dataset == "mnist":
				trainset = datasets.MNIST(root=env_config["data_path"], 
											train=True,
											download=True,
											transform= self.data_t if self.data_aug == True else self.t)
			elif self.dataset == "cifar10":
				trainset = datasets.CIFAR10(root=env_config["data_path"], 
											train=True,
											download=True,
											transform=self.data_t if self.data_aug == True else self.t)
		self.dl_train = enumerate(iter(DataLoader(trainset, shuffle=self.all_data,pin_memory=True)))

		if not self.all_data:
			self.episode_test, (self.reference_test, self.ref_y_test) = next(self.dl_train)
			self.episode_test, (self.reference_test, self.ref_y_test) = next(self.dl_train)
			self.episode_test, (self.reference_test, self.ref_y_test) = next(self.dl_train)

		# Reward
		wasserstein_distance = SinkhornDistance(eps=0.1, max_iter=125, reduction='mean')
		reduced_shape = (float(self.state_dim[0])*self.reward_input_scale_factor,float(self.state_dim[1])*self.reward_input_scale_factor)
		self.loss_template = torch.arange(np.prod(reduced_shape))
		if torch.cuda.is_available():
			self.loss_template = self.loss_template.to("cuda")
		# self.loss_template = torch.arange(100)
		self.wass_loss = lambda x,y: wasserstein_distance(torch.stack((self.loss_template ,
																torch.histc(nn.functional.interpolate(
																						x.unsqueeze(0).unsqueeze(0),
																						scale_factor=self.reward_input_scale_factor,
																						mode="bilinear").view(-1),
																						bins=len(self.loss_template
																			),
																)), 
														dim=1),
													torch.stack((self.loss_template ,
																torch.histc(nn.functional.interpolate(
																						y.unsqueeze(0).unsqueeze(0),
																						scale_factor=self.reward_input_scale_factor,
																						mode="bilinear").view(-1),
																						bins=len(self.loss_template
																			),
																)), 
														dim=1))
		self.wass_loss = lambda x,y: wasserstein_distance(torch.stack((self.loss_template ,x.view(-1)), dim=1),torch.stack((self.loss_template ,y.view(-1)), dim=1))
		if self.reward_distance == "euclidean":
			self.loss = lambda x,y: torch.dist(x, y, 2)
		elif self.reward_distance == "patch":
			self.n_patch = 6
			self.loss = lambda x,y: ((x-y).pow(2)).unfold(0,self.n_patch,self.n_patch).unfold(1,self.n_patch,self.n_patch).sum((2,3)).mean()
		elif self.reward_distance == "kl":
			self.loss = lambda x,y: torch.nn.functional.kl_div(torch.log(x+1),torch.log(y+1), reduction="batchmean", log_target=True)

		self.ssim_loss = lambda x,y: ssim(x.detach().cpu().clone().numpy(), y.detach().cpu().clone().numpy(), data_range=1.0)
		self.patch_loss = lambda x,y: ((x-y).pow(2)).unfold(0,self.n_patch,self.n_patch).unfold(1,self.n_patch,self.n_patch).sum((2,3)).mean()
		if self.non_linear_reward == "sigmoid":
			self.reward_post = lambda x: self.reward_scale*np.asscalar(torch.sigmoid(x).detach().cpu().clone().numpy().astype(float))
		elif self.non_linear_reward == "tanh":
			self.reward_post = lambda x: self.reward_scale*np.asscalar(torch.tanh(x).detach().cpu().clone().numpy().astype(float))
		elif self.non_linear_reward == "":
			self.reward_post = lambda x: self.reward_scale*np.asscalar(x.detach().cpu().clone().numpy().astype(float))
		
		# Observation space
		self.agent_model_channel = 2
		if self.coordconv:
			self.agent_model_channel += 3
			self.coord = torch.zeros([2, self.state_dim[0], self.state_dim[1]])
			for i in range(self.state_dim[0]):
				for j in range(self.state_dim[1]):
					self.coord[0, i, j] = i / float(self.state_dim[0])
					self.coord[1, i, j] = j / float(self.state_dim[1])
		if self.obs_with_step_num:
			self.agent_model_channel += 1
		
		self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(*self.state_dim, self.agent_model_channel), dtype=np.float32) # image dimension
		# Action space
		if self.action_type == "discrete":
			self.action_space = spaces.Discrete(np.prod(self.electrode_matrice_shape))
		elif self.action_type == "continuous":
			# self.action_space = spaces.Box(low=-1.0, high=1.0, shape=[int(np.prod(self.electrode_matrice_shape))], dtype=np.float32)
			# For batchnormvision second dim is needed
			self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(int(np.prod(self.electrode_matrice_shape)),), dtype=np.float32)
		elif self.action_type == "multidiscrete":
			self.action_space = spaces.Tuple([spaces.Tuple([spaces.Discrete(self.electrode_matrice_shape[0]), spaces.Discrete(self.electrode_matrice_shape[1])]) for i in range(self.n_action_bundle)])

		self.action_array = np.zeros(self.electrode_matrice_shape)
		self.reset()

	def reset(self):
		self.step_num = 0
		self.done = 0
		if self.all_data:
			self.episode, (self.reference, self.ref_y) = next(self.dl_train)
		elif not self.all_data:
			self.episode, (self.reference, self.ref_y) = self.episode_test, (self.reference_test, self.ref_y_test)
		if self.obs_with_step_num:
			# Reference image, step, canvas
			self.canvas = torch.cat([self.reference.squeeze().unsqueeze(dim=2),
										torch.zeros((*self.state_dim,1), dtype=torch.float32),
										torch.zeros((*self.state_dim,1), dtype=torch.float32),
										torch.zeros((*self.state_dim,1), dtype=torch.float32),
										self.coord[0,:,:].squeeze().unsqueeze(dim=2), 
										self.coord[1,:,:].squeeze().unsqueeze(dim=2)], dim=2)
		else:
			self.canvas = torch.cat([self.reference.squeeze().unsqueeze(dim=2), torch.zeros((*self.state_dim,1))], dim=2)
		
		self.canvas = self.canvas.to(torch.float32)
		if torch.cuda.is_available():
			self.canvas = self.canvas.to("cuda")
		loss = self.loss(self.canvas[:,:,1].squeeze(), self.canvas[:,:,0].squeeze())
		if self.reward_distance == "wasserstein":
			self.init_loss,_,_ = loss
			self.prev_loss,_,_ = loss
		else:
			self.init_loss = loss
			self.prev_loss = loss

		if self.render_animation:
			try:
				plt.close(self.fig_animation)
				del self.fig_animation
				del self.ax_animation
				del self.animation
			except:
				pass
			self.animation = []
			self.fig_animation, self.ax_animation = plt.subplots()
			canvas_img = self.ax_animation.imshow(self.canvas[:,:,1].cpu().numpy(), animated=True)
			self.animation.append([canvas_img])
		
		if self.overused_electrode_penalty:
			self.electrode_used = torch.ones(np.prod(self.electrode_matrice_shape))

		assert self.canvas.shape == self.observation_space.shape, "Observation tensor size {} does not match observation space size {}".format(self.canvas.shape, self.observation_space.shape)
		return self.canvas.detach().cpu().clone().numpy().astype(float)

	def step(self, action):
		self.step_num += 1

		l2_value, mse_value, ssim_value, kl_div_value, wass_value, patch_value = self.get_metrics()

		if isinstance(action, int):
			# Discrete
			action = [action]
			# self.action_bundle = 0
		else:
			# Continuous
			action = np.asarray(action)

		# Update percept
		if self.action_type == "continuous":
			if self.n_action_bundle:
				self.action_bundle = np.argsort(action)[int(-1*self.n_action_bundle)::]
				self.action_array = self.unravel_action(action)
				self.action2percept(self.action_array)
			else:
				self.action2percept(np.clip(action,0.0,1.0).reshape(self.electrode_matrice_shape))
		elif self.action_type == "discrete":
			print("Not implemented")
		elif self.action_type == "multidiscrete":
			print("Not implemented")

		if self.decay:
			self.canvas[:,:,1] = torch.clamp(self.percept[0,:,:],min=0,max=1)
		# Normalized canvas
		if self.coordconv:
			self.canvas[:,:,2] = self.canvas[:,:,1].squeeze()
		self.canvas[:,:,1] = torch.clamp(self.canvas[:,:,1].squeeze() + self.percept[0,:,:].squeeze(),min=0.0,max=1.0)
		if self.obs_with_step_num:
			self.canvas[:,:,3] = float(self.step_num)/float(self.episode_horizon)

		# Next reward
		if self.reward_distance == "wasserstein":
			curr_loss,_,_ = self.loss(self.canvas[:,:,1],self.canvas[:,:,0])
		else:
			curr_loss = self.loss(self.canvas[:,:,1],self.canvas[:,:,0])
		
		if self.overused_electrode_penalty == "":
			# reward = curr_loss / self.init_loss
			reward = (self.prev_loss - curr_loss) / self.init_loss
		elif self.overused_electrode_penalty == "ponderate":
			reward = self.electrode_used[self.action_bundle] * (self.prev_loss - curr_loss) / self.init_loss
		elif self.overused_electrode_penalty == "negative":
			reward = (self.prev_loss - (self.electrode_used[self.action_bundle] * curr_loss)) / self.init_loss
		elif self.overused_electrode_penalty == "kl":
			reward =  curr_loss
		self.prev_loss = curr_loss
		reward = self.reward_post(reward)

		if self.overused_electrode_penalty:
			self.electrode_used[self.action_bundle] += 1

		if self.step_num >= self.episode_horizon:
			self.step_num = 0
			self.done = 1

		info = {"distance_norm":curr_loss.item(),"mse_end":mse_value.item(),"kl_end":kl_div_value.item(),"ssim_end":ssim_value,"wass_end":wass_value,"patch_end":patch_value, "step":self.step_num}

		if self.env_render_itself:
			self.render()
		return self.canvas.detach().cpu().clone().numpy().astype(float), reward, self.done, info
		
	def action2percept(self, action):
		stim = p2p.stimuli.ImageStimulus(action)#.encode()
		self.implant.stim = stim # TODO: Use encode method to generate biphasic pulse https://pulse2percept.readthedocs.io/en/latest/examples/stimuli/plot_image_stim.html#converting-the-image-to-a-series-of-electrical-pulses
		self.percept = self.model.predict_percept(self.implant)
		data = self.percept.data.squeeze()
		self.percept = self.percept_t(Image.fromarray(cm.gray(data, bytes=True)).resize(self.state_dim))
		if torch.cuda.is_available():
			self.percept = self.percept.to("cuda")

	def unravel_action(self, action):
		action_array = np.zeros(self.action_array.shape, dtype=np.float32)
		self.electrodes_idx = np.unravel_index(self.action_bundle, self.electrode_matrice_shape)
		self.electrodes_values = action[self.action_bundle]
		if self.electrode_activation_level:
			action_array[self.electrodes_idx[0], self.electrodes_idx[1]] = self.electrode_activation_level
		else:
			action_array[self.electrodes_idx[0], self.electrodes_idx[1]] = self.electrodes_values

		return np.clip(action_array, 0.0, 1.0)

	def render(self):
		filename = self.resultspath + "/eps_{}_step_{}_done_{}_{}".format(self.ref_y, self.step_num, self.done, time.strftime("%Y_%m_%d-%H_%M_%S"))

		if self.done and self.render_animation:
			# Save ref, final canvas, ani
			self.save_animation(file=filename + ".mp4")
		elif self.render_animation:
			# Save canvas with steps
			# self.save_episode_sample(file=filename+".png", canvas_only=True)  
			canvas_img = self.ax_animation.imshow(self.canvas[:,:,1].cpu().numpy(), animated=True)
			self.animation.append([canvas_img])
		return True

	def get_metrics(self):
		l2 = torch.dist(self.canvas[:,:,1],self.canvas[:,:,0], 2).detach().cpu().clone().numpy()
		mse = torch.nn.functional.mse_loss(self.canvas[:,:,1].squeeze(),self.canvas[:,:,0].squeeze()).detach().cpu().clone().numpy()
		ssim = self.ssim_loss(self.canvas[:,:,1],self.canvas[:,:,0])
		kl_div = torch.nn.functional.kl_div(torch.log(self.canvas[:,:,1].squeeze()+1),torch.log(self.canvas[:,:,0].squeeze()+1), reduction="batchmean", log_target=True).detach().cpu().clone().numpy()
		wass,_,_ = (torch.zeros(1),0,0) #self.wass_loss(self.canvas[:,:,1],self.canvas[:,:,0])
		patch = self.patch_loss(self.canvas[:,:,1],self.canvas[:,:,0])
		return l2, mse, ssim, kl_div, wass.detach().cpu().clone().numpy(), patch.detach().cpu().clone().numpy()

	def save_animation(self, file='stimulation_pattern.mp4'):
		Image.fromarray(np.uint8(self.canvas[:,:,0].cpu().numpy()*255), 'L').save("{}_orig.png".format(file[:-4]))
		Image.fromarray(np.uint8(self.canvas[:,:,1].cpu().numpy()*255), 'L').save("{}_canvas.png".format(file[:-4]))
		ani = animation.ArtistAnimation(self.fig_animation, self.animation, interval=50, blit=True, repeat_delay=1000)
		ani.save(file)

	def save_episode_sample(self, file='sample.png', canvas_only = False):
		if canvas_only:
			Image.fromarray(np.uint8(self.canvas[:,:,1].cpu().numpy()*255), 'L').save("{}_canvas.png".format(file[:-4]))
		else:
			merged = self.canvas[:,:,0] + self.canvas[:,:,1]
			Image.fromarray(np.uint8(self.canvas[:,:,0].cpu().numpy()*255), 'L').save("{}_orig.png".format(file[:-4]))
			Image.fromarray(np.uint8(self.canvas[:,:,1].cpu().numpy()*255), 'L').save("{}_canvas.png".format(file[:-4]))
			Image.fromarray(np.uint8(merged.cpu().numpy()*255), 'L').save("{}_merged.png".format(file[:-4]))

