#!/bin/bash
#SBATCH --account=def-eplourde
##HH:MM:SS
#SBATCH --time=06:00:00
#SBATCH --job-name=sac_model_based
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jacob.lavoie@usherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load ffmpeg
module load python/3.6
##virtualenv $SLURM_TMPDIR/env
##source $SLURM_TMPDIR/env/bin/activate
##pip install --no-index --upgrade pip

##pip install --no-index -r requirements.txt
source /home/jlavoie/p2platest/bin/activate
tensorboard --logdir=/home/jlavoie/ray_results/ --host 0.0.0.0 &
python train_agent.py -jobid $SLURM_JOB_ID

