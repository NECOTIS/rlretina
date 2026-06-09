import ray
import torch
import torchvision
import numpy as np
from ray.rllib.policy.policy import Policy


class SOTAPolicy(Policy):
	"""Example of a custom policy written from scratch.

	You might find it more convenient to use the `build_tf_policy` and
	`build_torch_policy` helpers instead for a real policy, which are
	described in the next sections.
	"""

	def __init__(self, observation_space, action_space, config):
		Policy.__init__(self, observation_space, action_space, config)
		# example parameter
		self.weights = 1.0
		self.config = config
		self.observation_space = observation_space
		self.action_space = action_space


	def compute_actions(self,
						obs_batch,
						state_batches,
						prev_action_batch=None,
						prev_reward_batch=None,
						info_batch=None,
						episodes=None,
						explore=False,
						**kwargs):

		def get_sota_action(obs):
			action_array = np.zeros(60, dtype=np.float32)
			trans = torchvision.transforms.Compose([
			torchvision.transforms.ToPILImage(),
			torchvision.transforms.Resize([6,10]),
			torchvision.transforms.ToTensor()])

			step_num = int(np.unique(obs[:,:,3])*16.0) + 1

			flat_ref = torch.from_numpy(obs[:,:,0].astype(np.float32)).float()
			flat_ref = trans(flat_ref).numpy().squeeze().ravel()

			flat_ref_index = flat_ref.argsort()
			
			flat_ref = np.sort(flat_ref)
			electrode_value = flat_ref[-step_num]
			
			# print("Step {}: {} at {}".format(step_num, flat_ref_index[-step_num], electrode_value))
			
			action_array[flat_ref_index[-step_num]] = electrode_value

			return action_array.astype(np.float32)
		

		# return action batch, RNN states, extra values to include in batch
		return [get_sota_action(obs = b) for b in obs_batch], [], {}


	def learn_on_batch(self, samples):
		# implement your learning code here
		return {}  # return stats

	def get_weights(self):
		return {"weights": self.weights}
	
	def set_weights(self, weights):
		self.weights = weights["weights"]
