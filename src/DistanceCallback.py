import ray
from typing import Dict
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class DistanceCallback(DefaultCallbacks):
	def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, env_index: int, **kwargs):
		# assert episode.length == 0, \
		# 	"ERROR: `on_episode_start()` callback should be called right after env reset!"
		# print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
		episode.user_data["distance_norms"] = []
		episode.hist_data["distance_norms"] = []

	# def on_episode_step(self, * worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, env_index: int, **kwargs):
	# 	assert episode.length > 0, \
	# 	"ERROR: `on_episode_step()` callback should be called right after env reset!"

	def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, env_index: int, **kwargs):
		# assert episode.batch_builder.policy_collectors["default_policy"].batches[-1]["dones"][-1], \
		# "ERROR: `on_episode_end()` sould only be called " \
		# "after episode is done!"
		distance_norm = episode.last_info_for()["distance_norm"]
		ssim_end = episode.last_info_for()["ssim_end"]
		mse_end = episode.last_info_for()["mse_end"]
		kl_end = episode.last_info_for()["kl_end"]
		# assert 1 if "distance_norm" in episode.last_info_for().keys() else 0, "ERROR: Distance norm not found in info dict keys: {}".format(episode.last_info_for().keys())
		#distance_norm = np.mean(episode.user_data["distance_norms"])
		# print("episode {} (env-idx={}) ended with length {} and distance "
		# 	  " (2-norm) {}".format(episode.episode_id, env_index, episode.length, distance_norm))
		episode.user_data["distance_norm"] = distance_norm
		episode.custom_metrics["distance_norm"] = distance_norm
		episode.user_data["mse_end"] = mse_end
		episode.custom_metrics["mse_end"] = mse_end
		episode.user_data["ssim_end"] = ssim_end
		episode.custom_metrics["ssim_end"] = ssim_end
		episode.user_data["kl_end"] = kl_end
		episode.custom_metrics["kl_end"] = kl_end
		#episode.hist_data["distance_norms"] = episode.user_data["distance_norms"]

	#def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
		# print("returned sample batch of size {}".format(samples.count))
	
	def on_train_result(self, *, trainer, result: dict, **kwargs):
		result["callback_ok"] = True
		# print("trainer.train() result: {} -> {} episodes".format(
		# 	trainer, result["episodes_this_iter"]))
		# you can mutate the result dict to add new fields to return

	# def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
	# 	result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
	# 	print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
	# 		policy, result["sum_actions_in_train_batch"]))

	# def on_postprocess_trajectory(
	# 		self, *, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str,
	# 		policy_id: str, policies: Dict[str, Policy],
	# 		postprocessed_batch: SampleBatch,
	# 		original_batches: Dict[str, SampleBatch], **kwargs):
		# print("postprocessed {} steps".format(postprocessed_batch.count))
		# if "num_batches" not in episode.custom_metrics:
		# 	episode.custom_metrics["num_batches"] = 0
		# episode.custom_metrics["num_batches"] += 1
