# Before modifying this to fix bug, verify that Permission denied error are solved.
FROM rayproject/ray:1.5.0-py37-gpu

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && sudo dpkg -i cuda-keyring_1.0-1_all.deb

RUN sudo apt-key del 7fa2af80 && \
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
	wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | sudo apt-key add - && \
	sudo apt-get update && \
	sudo apt-get -y install libglib2.0-0 ffmpeg

# RUN sudo apt-get update && sudo apt-get -y install curl iputils-ping git ffmpeg rsync libglib2.0-0

# libopenmpi-dev libgl1 libgtk2.0-dev 

# RUN conda install python=3.9.6

# RUN pip install jupyter \
# 				numpy==1.21.2 \
# 				torch==1.10.0 \
# 				gym \
# 				pandas \
# 				lz4 \
# 				aiohttp \
# 				opencv-python \
# 				pytorch_lightning \
# 				dm_tree \
# 				ipyexperiments \
# 				ray[default] \
# 				ray[rllib] \
# 				"aioredis<2.0" \
# 				gym \
# 				Cython \
				# pickle5 \
# 				matplotlib

RUN pip install pickle5 \
				torch \
				# opencv \
				torchvision

RUN cd && \
	git clone https://github.com/ascientist/pulse2percept.git && \
	cd pulse2percept && \
	# git checkout tags/v0.7.1 && \
	git checkout double_requirement_bug_fix && \
	pip install -r requirements.txt && \
	pip install -r requirements-dev.txt && \
	pip install pulse2percept
	# pip install -e .


CMD ["bash"]
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
