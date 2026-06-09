import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class ResnetVisionNetwork(TorchModelV2, nn.Module):
	"""Pretrained Vision network for ResnetVisionNetwork"TorchModelV2, nn.Module"""

	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
		nn.Module.__init__(self)
		self.num_outputs = np.product(num_outputs)
		self.model = models.resnet18(pretrained=True)
		
		# for param in self.model.parameters():
		# 	param.requires_grad = False

		self.num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(self.num_ftrs, self.num_outputs)

	def forward(self, input_dict, state, seq_lens):
		# [print(k) for k in input_dict.keys()]
		# [print(t.shape) for t in input_dict["obs"]]
		#[print("Size: {} Type: {}".format(o.shape, type(o))) for o in input_dict["obs"]]
		# print(type(input_dict["obs"]))
		if isinstance(input_dict["obs"], list):
			action = input_dict["obs"].pop()
			self._features = input_dict["obs"].pop()
			# self._features = torch.stack(input_dict["obs"],dim=0)
		else:
			self._features = input_dict["obs"]

		conv_out = self.model(self._features)

		return conv_out, state


# Check on reward function in 07-07
# Check if resnet is frozen all the way or only last layer
