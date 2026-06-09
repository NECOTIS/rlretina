import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle5
import base64
import lz4.frame

def unpack(jsonLine):
	encoded = json.loads(jsonLine)
	data = base64.b64decode(encoded["new_obs"])
	data = lz4.frame.decompress(data)
	data = pickle5.loads(data)
	return data
	
listfiles = os.listdir("../data/sota_policy/")
listfiles = sorted(listfiles)
print("Opening file {}".format(listfiles[-1]))
f = open("../data/sota_policy/" + listfiles[-1])
lines = f.readlines()
obs_tensor = [unpack(i) for i in lines]

print("{} batch with size {}".format(len(obs_tensor), obs_tensor[0].shape))
print(obs_tensor[0].shape)
# f = open("../data/output-onesample.json")
print("Plot of size: {},{}".format(obs_tensor[0].shape[-1], obs_tensor[0].shape[0]))

fig, axs = plt.subplots(obs_tensor[0].shape[-1], obs_tensor[0].shape[0])

for batch in range(obs_tensor[0].shape[0]):
	for i in range(obs_tensor[0].shape[-1]):
		d = obs_tensor[0][batch,:,:,i]
		axs[i,batch].imshow(d,vmin=0.0,vmax=1.0)
		axs[i, batch].set_title('batch {}'.format(batch))

plt.show()
