#!/bin/bash

COMPUTE_HOST=$1
COMPUTE_TIME=${2:-24}

git commit -am 'Launching job on cluster...'

if [[ "$CC_CLUSTER" == "cedar" ]]; then
	SOURCE_PATH="$PROJECT/$USER"
	PERSISTENT_DATA_PATH="$SOURCE_PATH"
	CPU_N=24
elif [[ "$CC_CLUSTER" == "narval" ]]; then
	SOURCE_PATH="$HOME"
	PERSISTENT_DATA_PATH="$HOME/projects/def-eplourde/$USER"
	CPU_N=48
elif [[ "$CC_CLUSTER" == "beluga" ]]; then
	SOURCE_PATH="$HOME"
	PERSISTENT_DATA_PATH="$HOME/projects/def-eplourde/$USER"
	CPU_N=40
fi

echo "Creating archive to $SOURCE_PATH/rlretina/"
cd "$SOURCE_PATH/rlretina/"
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_HASH_TAR="$COMMIT_HASH.tar"
mkdir -p "$SOURCE_PATH/rlretina/exp/"
git archive --format=tar --output="$SOURCE_PATH/rlretina/exp/$COMMIT_HASH.tar" HEAD

echo "Copying source files from $COMMIT_HASH_TAR over scratch..."
cp -R "$SOURCE_PATH/rlretina/exp/$COMMIT_HASH_TAR" $SCRATCH/

cd $SCRATCH

if [ ! -f $COMMIT_HASH ]; then
	echo "Unarchive commit archive..."
	mkdir $COMMIT_HASH && tar -xvf $COMMIT_HASH_TAR -C $COMMIT_HASH
fi

# Move data
echo "Moving datasets..."
mkdir -p $PERSISTENT_DATA_PATH/ray_results/
mkdir -p $SCRATCH/$COMMIT_HASH/sota_policy/
mkdir -p $SCRATCH/$COMMIT_HASH/agent_policy/
touch $SCRATCH/$COMMIT_HASH/torch_data.lock
cp -R $SOURCE_PATH/rlretina/data/MNIST $SCRATCH/$COMMIT_HASH/
cp $SOURCE_PATH/rlretina/data/electrodesPositions.mat $SCRATCH/$COMMIT_HASH/
cp $PERSISTENT_DATA_PATH/rlretina/data/sota_policy/output-2022-03-24_21-23-35_worker-1_0.json $SCRATCH/$COMMIT_HASH/sota_policy

# Add local for salloc experiment

if [[ "$COMPUTE_HOST" == "local" ]]; then
	echo "Running training locally ($SCRATCH/$COMMIT_HASH)"
	cd "$SCRATCH/$COMMIT_HASH/src" 
	bash run_tune_agent.sh "$COMMIT_HASH" $COMPUTE_TIME "$PERSISTENT_DATA_PATH/ray_results" $CPU_N

elif [[ "$COMPUTE_HOST" == "node" ]]; then
	echo "Running training on ${CC_CLUSTER} compute canada cluster"
	cd "$SCRATCH/$COMMIT_HASH/src" && sbatch --job-name="${COMMIT_HASH:0:5}_rlretina" --cpus-per-task=$CPU_N --time=$COMPUTE_TIME:00:00 run_tune_agent.sh $COMMIT_HASH_TAR $COMPUTE_TIME "$PERSISTENT_DATA_PATH/ray_results" $CPU_N
fi
