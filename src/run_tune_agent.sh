#!/bin/bash
#SBATCH --account=def-eplourde
##HH:MM:SS
#SBATCH --mem=128000M
##SBATCH --cpus-per-task=48
##SBATCH --gres=gpu:1
#SBATCH --mail-user=jacob.lavoie@usherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

COMMIT_HASH=$1
COMPUTE_TIME=$2
RESULT_PATH=$3
N_CPU=$4
module load ffmpeg
module load python/3.9
module load cuda

echo "Loading python env..."
source "$HOME/rlp2p/bin/activate"

cd $SCRATCH

echo "Launching tmux session with commit $COMMIT_HASH for $COMPUTE_TIME hours..."

ray stop --force

unset TMUX

cd $COMMIT_HASH && tmux new-session -d -s raystdout
tmux send-keys -t raystdout.0 "cd $SCRATCH/$COMMIT_HASH/src && python tune_agent.py --jobid $SLURM_JOB_ID --commitid $COMMIT_HASH --trainingtime $COMPUTE_TIME --ncpu $N_CPU --gpu 0.0  --resultspath $RESULT_PATH --datapath $SCRATCH/$COMMIT_HASH" ENTER

echo "Tensorboard launching..."

if [[ -z "$(pgrep tensorboard)" ]]; then
	tensorboard --logdir="$RESULT_PATH" --host 0.0.0.0 --load_fast false &
fi

echo "Tensorboard launched ($RESULT_PATH). Tailing bash session..."

tail -f /dev/null
