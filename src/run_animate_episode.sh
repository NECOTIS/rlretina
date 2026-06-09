#!/bin/bash
#SBATCH --account=def-eplourde
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=animate_episode
#SBATCH --mem=128G
#SBATCH --mail-user=jacob.lavoie@usherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

EXP_FOLDER=$1
NSAMPLE=$2
module load python/3.7
module load ffmpeg
source /home/jlavoie/rlp2p/bin/activate

if [[ -z "$(pgrep tensorboard)" ]]; then
	tensorboard --logdir=~/ray_results/ --host 0.0.0.0 --load_fast false &
fi

ray stop --force
if [[ EXP_FOLDER == "" ]]; then
	echo "Missing experiment folder."
else
	python animate_episode.py --path "/home/jlavoie/ray_results/$EXP_FOLDER/" --nsample $NSAMPLE
fi

