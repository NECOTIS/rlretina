# rlretina

Reinforcement-learning agents that learn to **drive an epiretinal implant** so that
a simulated retina "sees" a target image. The environment wraps
[pulse2percept](https://github.com/pulse2percept/pulse2percept)'s biophysical
percept models (AxonMap) as the forward model of a retinal prosthesis: the agent
chooses electrode stimulation, pulse2percept renders the elicited percept, and the
agent is rewarded for matching the intended image.

This is the research code behind the model-based deep-RL study of *in silico*
epiretinal stimulation carried out in the **iBionics** project within the
**NECOTIS / uSENS** research group (Université de Sherbrooke).

> 📄 **Paper:** *Learning to See via Epiretinal Implant Stimulation in silico with
> Model-Based Deep Reinforcement Learning* — *Biomedical Physics & Engineering
> Express* (2024). See [Citation](#citation).

---

## What's here

| Path | Purpose |
|------|---------|
| `src/env_without_user_feedback.py` | The Gym-style retinal-stimulation environment (pulse2percept forward model). |
| `src/ibionicsElectrodeArray.py` | iBionics electrode-array geometry. |
| `src/train_agent.py` / `src/tune_agent.py` | Train / hyper-parameter-tune the RL agent (Ray RLlib / SAC). |
| `src/sota_policy.py`, `src/behaviour_cloning.py` | Baseline & behaviour-cloning policies. |
| `src/resnet_model.py`, `src/visionnet.py`, `src/layers.py` | Vision / model networks. |
| `src/animate_episode.py` | Render an episode (percept vs. target) to video. |
| `src/test_*.py` | Environment / model / CUDA smoke tests. |
| `src/run_*.sh`, `src/submit_git_commit.sh` | SLURM launch scripts (Digital Research Alliance of Canada clusters). |
| `docker/torch/Dockerfile` | GPU image (Ray + Torch + pulse2percept). |
| `data/electrodesPositions.mat` | Electrode positions used by the environment. |

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a **reference snapshot** of the environment used for the
experiments. The key dependency is a **recent pulse2percept (> 0.7)** — the
released stable version is not suited to this work, so install from source:

```bash
pip install "git+https://github.com/pulse2percept/pulse2percept.git"
```

## Usage

```bash
cd src
# Train / tune (locally or via SLURM — see the run_*.sh headers)
python tune_agent.py --help
python train_agent.py --help
# Render an episode to video
python animate_episode.py --path <ray_results_run> --nsample <N>
```

The `run_*.sh` and `submit_git_commit.sh` scripts document how the experiments
were launched on SLURM clusters and are kept for reproducibility; adapt the
account, module, and path lines to your environment.

## Roadmap

> 📐 Detailed engineering plan for the GPU-native refactor: [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

- [ ] **GPU-native environment.** Re-implement the AxonMap forward model so the
      *same* percept computation runs on **JAX or PyTorch on GPU** (the current
      pulse2percept path is CPU-bound and dominates wall-clock). This is the main
      planned refactor — it should make the env differentiable-friendly and
      dramatically faster to train against.
- [ ] Train on subject-specific electrode coordinates to test per-patient adaptation.
- [ ] CIFAR-10 / data-augmentation training regimes.
- [ ] Surface actor/critic architectures in `animate_episode`.

## Citation

If you use this code, please cite:

```bibtex
@article{lavoie2024learningtosee,
  title   = {Learning to See via Epiretinal Implant Stimulation in silico
             with Model-Based Deep Reinforcement Learning},
  author  = {Lavoie, Jacob and others},
  journal = {Biomedical Physics \& Engineering Express},
  year    = {2024},
  doi     = {10.1088/2057-1976/acf1a5}
}
```

## License

BSD 3-Clause — see [LICENSE](LICENSE). © NECOTIS (NEuro COmputational &
Intelligent Signal Processing Research Group), Université de Sherbrooke.
