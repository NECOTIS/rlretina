# Refactor Plan: GPU-Native, Differentiable AxonMap Forward Model

**Status:** Proposed roadmap (engineering plan only — does not change model code yet)
**Owner:** Jacob Lavoie (NECOTIS / uSENS, Université de Sherbrooke)
**Goal:** Re-implement the pulse2percept *AxonMap* retinal-percept forward model as a
vectorized, GPU-resident tensor kernel (PyTorch, with a JAX option discussed), so the
RL environment stops being CPU-bound and becomes batchable and differentiable. The
existing pulse2percept path is kept as the **reference oracle** for numerical-parity
validation.

---

## 0. TL;DR

- **Verdict: PyTorch.** The stack is already Torch + Ray RLlib (SAC); the env, reward,
  and observation tensors already live on CUDA. A Torch percept kernel drops in with
  zero framework-boundary copies and zero new runtime deps. JAX would buy cleaner
  `vmap`/`jit` batching and arguably nicer differentiability, but the integration tax
  (a second array framework inside Ray workers, dtype/device bridging, RLlib is Torch)
  is not worth it here. (See §3.)
- **Core idea:** The AxonMap percept is, for fixed geometry and fixed `(rho, axlambda)`,
  a **static precomputed tensor contracted with the per-step electrode amplitude
  vector**. Everything expensive (Jansonius axon bundles, per-segment soma distances,
  electrode→segment radial Gaussian) is geometry- and parameter-dependent only →
  **precompute once at `build()`**. Per step, the percept is a masked
  `(amplitudes × radial_gauss × axon_sensitivity)`, summed over electrodes, then a
  **max over axon segments per pixel** — i.e. a batched segment-gather + reduction, a
  handful of GPU ops. (See §2.)
- **Expected win:** removes a per-step Python/Cython CPU call (`predict_percept`,
  ~tens of ms) from the hot loop and replaces it with a sub-millisecond GPU op that
  **batches across all parallel RLlib envs at once**. Order-of-magnitude-plus
  wall-clock improvement on training throughput is the target, with the exact factor
  set by `num_workers × num_envs_per_worker` and image size. (See §1.)
- **Bonus:** the kernel is differentiable in the electrode amplitudes, enabling
  gradient-based stimulus optimization and analytic world-model gradients. (See §4.)

---

## 1. Bottleneck analysis

### 1.1 Where the percept is computed

The percept is recomputed **on every environment step**, inside the RL hot loop.

- `src/env_without_user_feedback.py :: p2penv.step()` calls
  `self.action2percept(self.action_array)` once per step.
- `action2percept()` does the expensive work:

  ```python
  def action2percept(self, action):
      stim = p2p.stimuli.ImageStimulus(action)
      self.implant.stim = stim
      self.percept = self.model.predict_percept(self.implant)   # <-- CPU AxonMap forward pass
      data = self.percept.data.squeeze()
      self.percept = self.percept_t(Image.fromarray(cm.gray(data, bytes=True)).resize(self.state_dim))
      ...
  ```

- `self.model` is a `p2p.models.AxonMapModel(...)` built with `self.model.engine = 'joblib'`
  in `__init__`. `predict_percept` runs the compiled `fast_axon_map` kernel on CPU
  (joblib over CPU cores), returning a `Percept` object that is then:
  1. squeezed to a NumPy array,
  2. colormapped via `matplotlib.cm.gray(..., bytes=True)` to an RGBA uint8 image,
  3. wrapped in a PIL `Image`, resized to `state_dim`,
  4. run through `torchvision` transforms back to a tensor,
  5. moved back to CUDA.

That round-trip (Cython CPU compute → NumPy → matplotlib colormap → PIL → torchvision
→ CUDA) happens **once per step, single-stimulus, no batching**.

### 1.2 Step frequency

- `episode_horizon` is 16 (`test_env.py`) or 100 (`train_agent.py`). One
  `predict_percept` per step ⇒ 16–100 forward passes per episode per env.
- `train_agent.py` configures Ray RLlib SAC with
  `num_workers = 8`, `num_envs_per_worker = 8` ⇒ up to **64 parallel envs**, each
  independently calling `predict_percept` on CPU through its own pulse2percept model.
- Off-policy SAC with `learning_starts = 100000` and `train_iter = 5000` means
  **millions of env steps**, hence millions of CPU AxonMap forward passes. The
  `test_env.py` harness already prints "Time elapsed generating data with N steps",
  which is the right place to measure the baseline.

### 1.3 Why it dominates wall-clock

1. **It is the only CPU-bound, non-batched op in an otherwise GPU pipeline.** The
   policy/Q nets, reward (`patch`/Sinkhorn), and canvas all run on CUDA already
   (the env moves `canvas`, `loss_template`, `percept` to `"cuda"`). The percept is
   the lone CPU detour.
2. **No batching.** Each of the 64 envs computes a single 1-stimulus percept. AxonMap
   cost scales with `n_axons × n_ax_segments × n_electrodes` per pixel; doing this 64×
   serially on CPU wastes the GPU that is otherwise idle during the env step.
3. **Per-step Python overhead** (PIL/matplotlib/torchvision conversions, object
   allocation) adds fixed cost on top of the kernel itself, ~every step.
4. **Off-policy RL is sample-hungry:** the ratio of env steps to gradient steps is high,
   so env throughput, not the learner, caps training speed.

### 1.4 Estimated win from GPU + batching

The forward model is an affine-then-pointwise-then-reduce computation (see §2) whose
static part is a sparse/dense tensor of shape `(n_pixels, n_segments)` plus an
electrode→segment Gaussian. On GPU this is a single fused kernel.

- **Per-call latency:** CPU `predict_percept` + conversions is typically tens of ms;
  the GPU equivalent for one stimulus is sub-millisecond. **~10–100× per call.**
- **Batching across envs:** stacking all `B = num_workers × num_envs_per_worker`
  amplitude vectors into one `(B, n_electrodes)` matrix turns 64 serial CPU calls into
  **one** batched GPU matmul/reduction. This is the larger multiplier.
- **Net:** the percept stops being the bottleneck; throughput becomes gated by RLlib
  sampling/learner overhead. Concretely, target an **order-of-magnitude-plus**
  reduction in env wall-clock; the precise factor must be measured (Milestone M5)
  against the `test_env.py` baseline because it depends on image size, `n_axons`,
  `n_ax_segments`, batch size, and GPU.

> **Important caveat to validate early:** the current env wraps stimulation as
> `ImageStimulus(action)` (the action is rasterized over the electrode grid and the
> electrode amplitude per electrode is then the resampled image value). The GPU kernel
> must reproduce *that exact* electrode-amplitude assignment, not just the AxonMap math
> (see §6.1). Note the env also does `electrode_activation_level` / `n_action_bundle`
> gating in `unravel_action()` before the percept — the GPU path replaces only the
> `ImageStimulus → predict_percept → tensor` segment, keeping the amplitude-construction
> logic identical.

---

## 2. The AxonMap math, written as vectorized tensor ops

### 2.1 Reference formulation (pulse2percept, Beyeler et al. 2019)

The AxonMap (a.k.a. `AxonMapSpatial` inside `AxonMapModel`) model has two ingredients.

**(a) Axon trajectories — Jansonius (2009) model.** Nerve fiber bundles spiral out
from the optic disc. For each starting angle `phi0` at the optic nerve head, a bundle
is traced in polar coordinates `(r, phi)`:

```
b   = exp(beta_sup + 3.9 * tanh(-(phi0 - 121.0) / 14.0))      # superior retina
c   = 1.9 + 1.4 * tanh((phi0 - 121.0) / 14.0)
phi = phi0 + b * (r - r_min)**c
x'  = r * cos(deg2rad(phi)),   y' = r * sin(deg2rad(phi))
```

(analogous constants for inferior retina). This yields, per axon, an ordered list of
segment positions `(x_s, y_s)` walking from the periphery toward the soma. Controlled by
`n_axons` (default 1000), `axons_range` (−180°..180°), `n_ax_segments` (default 500),
`ax_segments_range` (0..50 dva). **Purely geometric — depends only on retina geometry,
not on stimulation.**

**(b) Per-pixel brightness.** Each pixel of the output grid is assigned to the nearest
axon bundle. Brightness along that axon is governed by two decays:

- **Radial (electrode → segment), governed by `rho`:** a Gaussian in the distance
  `r` from each electrode to the axon segment:

  ```
  radial = exp( -r^2 / (2 * rho^2) )
  ```

- **Axonal (segment → soma), governed by `axlambda`:** the segment's *sensitivity*
  decays with cumulative distance `d` walked along the axon from the soma:

  ```
  axon_sensitivity(segment) = exp( -d^2 / (2 * axlambda^2) )
  ```

**(c) Combination rule (exact, from the compiled `fast_axon_map` kernel).** For a given
pixel `p` whose axon has segments `s`:

```
segment_brightness(s) = axon_sensitivity(s) * Σ_e [ amp_e * exp( -r(e,s)^2 / (2*rho^2) ) ]
pixel_brightness(p)   = max_s | segment_brightness(s) |       # max over that axon's segments
if |pixel_brightness(p)| < thresh_percept: pixel_brightness(p) = 0
```

i.e. **sum over electrodes** (linear superposition of electrode contributions, scaled
by amplitude `amp_e`), **multiply by the per-segment axon sensitivity**, then take the
**max (by absolute value) over the axon's segments**, then threshold. Confirmed against
`pulse2percept/models/_beyeler2019.pyx`:

```cython
gauss      = c_exp(-r2 / (2.0 * rho * rho))
sgm_bright = sgm_bright + gauss * axon_segments[idx_ax, 2] * amp   # axon_segments[.,2] = sensitivity
if c_abs(sgm_bright) > c_abs(px_bright):
    px_bright = sgm_bright                                          # max over segments
if c_abs(px_bright) < thresh_percept:
    px_bright = 0.0
```

> Note the *radial* term is `exp(-r²/(2ρ²))` (distance electrode→segment) and the
> *axonal sensitivity* is `exp(-d²/(2λ²))` (cumulative arc-length segment→soma). Both
> are Gaussian-in-squared-distance. The published intensity form
> `I = exp(-(r²)/(2ρ²)) · exp(-(d²)/(2λ²))` is exactly this product, with the
> electrode sum inside and the max-over-segments outside.

### 2.2 Static vs dynamic decomposition

| Quantity | Depends on | Static? | Where |
|---|---|---|---|
| Axon bundle segment positions (Jansonius) | retina geometry | **static** | build once |
| Pixel→axon assignment | grid + geometry | **static** | build once |
| Per-segment axon sensitivity `exp(-d²/(2λ²))` | `axlambda`, geometry | **static** (per `axlambda`) | build once |
| Electrode positions | implant geometry | **static** | build once |
| Electrode→segment distances `r(e,s)` | geometry | **static** | build once |
| Radial Gaussian `exp(-r²/(2ρ²))` | `rho`, geometry | **static** (per `rho`) | build once |
| **Electrode amplitudes `amp_e`** | **RL action, per step** | **dynamic** | per step |

Since `(rho, axlambda)` are fixed for a training run (env_config), **everything except
`amp_e` is precomputable.** This is the crux of the speedup.

### 2.3 Precomputed tensors (built once)

Let:
- `E` = number of electrodes,
- `A` = number of axons (bundles), `S` = max segments per axon,
- `P` = number of output pixels (then resized to `state_dim`).

Build, on `init`/`build()`:

1. **Segment geometry** `seg_xy ∈ R^{A×S×2}` from the Jansonius trace (pad ragged axons
   to `S`, with a boolean `seg_mask ∈ {0,1}^{A×S}`).
2. **Axon sensitivity** `axon_sens ∈ R^{A×S}` = `exp(-d²/(2·axlambda²))` (cumulative
   arc length `d` per segment), zeroed where `seg_mask==0`.
3. **Electrode→segment radial weights**
   `W ∈ R^{A×S×E}`, `W[a,s,e] = exp(-‖seg_xy[a,s] − elec_xy[e]‖² / (2·rho²))`.
   (Optionally pruned: drop `(a,s,e)` entries below a tolerance to make `W` sparse —
   AxonMap radial support is local, so this is a big memory/compute saver.)
4. **Pixel→axon map** `pix2axon ∈ Z^{P}` (nearest bundle per pixel) plus a scatter/gather
   index so the final per-axon brightness can be written to pixels. Equivalently
   precompute, per pixel, the index of its axon and reduce over that axon's segments.

These are buffers (PyTorch `register_buffer`, moved to CUDA once). `W` is the dominant
memory term (`A·S·E`); pruning to local support keeps it tractable. As a fallback,
recompute the Gaussian on the fly from precomputed `r²` (store `r2 ∈ R^{A×S×E}` instead
of `W`, apply `exp(-r2/(2ρ²))` each step) if differentiating w.r.t. `rho` is ever wanted.

### 2.4 Per-step forward pass (dynamic)

Given a **batch** of electrode amplitudes `amp ∈ R^{B×E}` (B = number of envs in the
batch):

```
# 1. Sum over electrodes (linear superposition), per axon segment:
#    seg_drive[b,a,s] = Σ_e amp[b,e] * W[a,s,e]
seg_drive = einsum('be,ase->bas', amp, W)          # (B, A, S)

# 2. Multiply by static axon sensitivity:
seg_bright = seg_drive * axon_sens                 # (B, A, S)  (axon_sens broadcast)

# 3. Max over segments (by |.|), masked:
seg_bright = where(seg_mask, seg_bright, 0)
axon_bright = max_by_abs(seg_bright, dim=S)        # (B, A)

# 4. Scatter to pixels via pix2axon, reshape to image:
percept = axon_bright[:, pix2axon].reshape(B, H, W_img)   # (B, H, W)

# 5. Threshold + clamp/normalize to match env's [0,1] convention:
percept = where(abs(percept) < thresh, 0, percept)
percept = clamp(percept, 0, 1)                     # match canvas usage
# 6. Resize to state_dim with F.interpolate (replaces PIL/torchvision resize)
percept = F.interpolate(percept[:,None], size=state_dim, mode='bilinear')[:,0]
```

Step 1 is the only heavy op and is a single batched contraction; with `W` pruned to
local support it is sparse and cheap. Steps 2–6 are elementwise / reductions / a gather.
The whole thing stays on GPU; **no NumPy/PIL/matplotlib round-trip.**

`max_by_abs` mirrors the Cython `c_abs` comparison: take the segment whose signed value
has the largest magnitude (`idx = argmax(|x|, S); gather(x, idx)`); preserves sign for
biphasic/negative amplitudes. (For the current env, amplitudes are clamped ≥ 0, so a
plain `max` is equivalent — but keep the abs form for parity and generality.)

### 2.5 Reuse note

Rather than reimplement Jansonius tracing from scratch, **harvest the static tensors
directly from a built pulse2percept model**: `AxonMapSpatial` exposes the grown axon
bundles (the pickled `axon_contrib` / `grid` structures). Plan: build the p2p model once
(as today), extract segment positions, sensitivities, and the pixel grid, convert to
Torch buffers, and compute `W`. This guarantees geometric parity by construction and
removes the risk of re-deriving the Jansonius constants. (See `_build()` /
`calc_axon_sensitivity` / the pickled axon map in `beyeler2019.py`.)

---

## 3. JAX vs PyTorch — recommendation

**Recommendation: PyTorch.** Implement the kernel as an `nn.Module` (buffers + a
`forward(amp) -> percept`) living in the env.

| Dimension | PyTorch | JAX |
|---|---|---|
| Fits existing stack | **Yes** — env, reward, canvas, RLlib (SAC) are all Torch/CUDA already | No — adds a 2nd array framework inside Ray workers |
| Integration cost | Drop-in; replace `action2percept` body; no device/dtype bridging | dlpack/`jax.numpy`↔Torch bridging at the env boundary; extra deps |
| Batching | `einsum` + `torch.vmap` (or just batched einsum) | `vmap`/`jit` are cleaner, but not needed for a single einsum |
| JIT / fusion | `torch.compile` available for the kernel | `jit` is excellent, marginal benefit for this op |
| Differentiability | autograd through the einsum/reductions — sufficient for §4 | also autograd; `grad`/`jacfwd` ergonomics slightly nicer |
| RLlib compatibility | **Native Torch policy** | RLlib runs the Torch policy regardless; JAX env is friction |
| Team familiarity | Codebase is already Torch | New paradigm (functional, PRNG keys) |

**Tradeoff being consciously accepted:** JAX's `vmap`/`jit`/`grad` ergonomics and its
functional purity are genuinely nicer for "a batched differentiable simulator," and if
this were greenfield JAX would be defensible. But the deciding factors are (1) RLlib +
SAC + the entire env/reward stack are Torch on CUDA, (2) introducing JAX means a second
accelerator framework co-resident in every Ray worker (memory, init, version-pinning
pain), and (3) the hot kernel is one batched contraction — JAX's compilation edge is
marginal here. **Torch keeps the percept on the same device as everything else with zero
framework boundary.** Use `torch.compile` on the kernel for fusion if needed.

> Escape hatch: the kernel is ~50 lines of array ops. If a future, purely offline
> *stimulus-optimization / world-model* research track wants JAX, the same math ports in
> an afternoon. Keep the math in a framework-thin module to make that cheap.

---

## 4. Differentiability payoff

Because the GPU kernel is built from `einsum`, elementwise `exp`, masked `max`/gather,
and `interpolate`, **the percept is differentiable w.r.t. the electrode amplitudes**
`amp` (and, if `W` is recomputed from `r²`, w.r.t. `rho`/`axlambda` too). The only
non-smooth points are the `max`-over-segments (subgradient, like max-pool — fine) and the
hard threshold (use a straight-through or soft threshold if gradients through it matter).

What this unlocks:

1. **Gradient-based stimulus optimization (white-box baseline).** With a differentiable
   percept `f(amp)` and a differentiable image-match loss `L(f(amp), target)` (the env
   already uses differentiable `patch`/Sinkhorn losses), one can directly
   `argmin_amp L` by gradient descent. This is a strong non-RL oracle/upper-bound to
   benchmark the RL agent against, and a much better "SOTA" baseline than the current
   greedy `sota_policy.py`.
2. **Analytic world-model gradients (model-based RL).** The project is *model-based*
   deep RL. A differentiable, known transition (percept) model means the
   "imagination"/planning rollouts can backprop *through the true dynamics* rather than
   through a learned approximation — enabling analytic policy gradients
   (SVG/Dreamer-style value-gradient or short-horizon differentiable planning) instead of
   only score-function (SAC) gradients. The canvas update
   `canvas[...,1] = clamp(canvas + percept)` is already differentiable, so a whole
   episode is differentiable end-to-end in the actions.
3. **Better learned world models / auxiliary losses.** Even where full
   differentiable planning isn't adopted, the exact percept can supervise a learned
   dynamics model (distillation), or serve as a differentiable reconstruction
   auxiliary loss to shape representations.
4. **Sensitivity / parameter inference.** Differentiability w.r.t. `rho`/`axlambda`
   enables fitting model parameters to subject data by gradient descent (relevant to the
   roadmap's "subject-specific coordinates" goal).

**What changes for the RL formulation:** SAC still works unchanged on the faster batched
env (drop-in). The *new* capability is an optional **differentiable-env mode** where the
transition is exposed as a differentiable function, opening SVG/short-horizon
value-gradient methods and a gradient-descent stimulus-optimization baseline. This is an
extension (Milestone M4), not a prerequisite for the speedup (M1–M3).

**Prior art to reuse, not reinvent.** Differentiable phosphene simulation is established:
van der Grinten et al., *"Towards biologically plausible phosphene simulation for the
differentiable optimization of visual cortical prostheses,"* eLife 13:e85812 (2024), ships
the **dynaphos** PyTorch simulator built explicitly for end-to-end differentiable
optimization of prosthetic vision. pulse2percept itself is (as of current releases)
**CPU/Cython/joblib with no JAX/Torch/GPU differentiable backend** — confirming this
refactor is net-new for the AxonMap model and should borrow design patterns (Gaussian
phosphene fields, differentiable rendering, batched Torch) from dynaphos while keeping the
*AxonMap-specific* axon-streak math and numerical parity with pulse2percept.

---

## 5. Architecture / integration

### 5.1 New module

Add `src/axonmap_torch.py` exposing:

```python
class TorchAxonMap(nn.Module):
    def __init__(self, p2p_model, grid, rho, axlambda, thresh_percept, device): ...
        # extract Jansonius segments + sensitivities from the *built* p2p model,
        # precompute W (electrode->segment radial Gaussian), axon_sens, seg_mask,
        # pix2axon; register as buffers; move to device.
    def forward(self, amp):           # amp: (B, E) or (E,)  -> percept (B, H, W)
        ...
    @classmethod
    def from_env_config(cls, env_config): ...   # builds the p2p oracle once, harvests tensors
```

Keep the math framework-thin (pure tensor ops in one function) so a JAX port is cheap.

### 5.2 Env integration (`env_without_user_feedback.py`)

- In `__init__`: after building `self.model` (the p2p AxonMap — **kept as oracle**),
  also build `self.gpu_model = TorchAxonMap.from_env_config(env_config)` (gated by a new
  `env_config["percept_backend"] in {"p2p", "torch"}`, default `"torch"`).
- Replace the body of `action2percept(action)` for the torch backend:

  ```python
  amp = self._action_to_amplitudes(action)        # same electrode-amp construction as today
  self.percept = self.gpu_model(amp_tensor)        # (1,H,W) on CUDA, already resized
  # no PIL / matplotlib / numpy round-trip
  ```

  The `_action_to_amplitudes` step must reproduce *exactly* what
  `ImageStimulus(action)` + the implant currently feed to `predict_percept` (electrode
  ordering, the image→electrode resampling, `electrode_activation_level`,
  `n_action_bundle` gating in `unravel_action`). This is the parity-critical glue (§6.1).
- `step()`, reward, canvas, observation are unchanged — they already consume
  `self.percept` as a CUDA tensor.

### 5.3 RLlib vectorized rollout

Two integration levels, do the simple one first:

- **Level A (per-env, immediate, M3):** each RLlib env independently calls the Torch
  kernel with `B=1`. Already a big win (GPU kernel, no CPU round-trip) and requires no
  RLlib changes. Watch GPU-context cost per worker: with `num_gpus_per_worker = 1/8`,
  many envs share a GPU; keep buffers small (prune `W`) and reuse one module per worker.
- **Level B (true batched env, M2/M5):** implement a vectorized env (RLlib
  `VectorEnv` / `num_envs_per_worker` batched, or a Gym `VectorEnv`) so all envs in a
  worker submit one `(B, E)` amplitude batch and get `(B, H, W)` back in a single kernel
  launch. This realizes the full batching multiplier from §1.4. Cleanest if the canvas /
  reward are also batched (they already are tensor ops).

### 5.4 Oracle retention

`self.model` (pulse2percept) stays available behind `percept_backend="p2p"` and is used
by the parity test harness (§6). It is the ground truth; the Torch model must match it.

---

## 6. Numerical-parity validation

**Goal:** prove the Torch percept ≈ pulse2percept percept within tolerance, before any
training is trusted.

### 6.1 Parity targets (two layers)

1. **Amplitude-construction parity:** assert the Torch `_action_to_amplitudes(action)`
   produces the *same per-electrode amplitude vector* that pulse2percept derives from
   `ImageStimulus(action)` + implant. Compare `implant.stim`-derived amplitudes vs the
   Torch vector elementwise. (Most parity bugs will live here, not in the math.)
2. **Percept parity:** for the same amplitude vector, compare Torch percept vs
   `model.predict_percept(implant).data`, **before** the env's resize/colormap (compare
   on the model's native grid to isolate the math from the resampling), and again after
   the full env post-processing.

### 6.2 Metrics & tolerances

On the model-native grid, over a battery of stimuli:
- **Max abs per-pixel error** `max|P_torch − P_p2p|` ≤ `1e-3` (normalized [0,1]).
- **Mean abs error** ≤ `1e-4`.
- **Relative L2** `‖ΔP‖/‖P_p2p‖` ≤ `1e-3`.
- **SSIM** ≥ `0.999` (the env already imports `skimage ssim`).
- Differences should be attributable only to float32 vs float64 and the `thresh_percept`
  boundary; investigate anything larger (usually a geometry/indexing/electrode-order bug).

### 6.3 Test images / stimuli

- **Single-electrode impulses:** activate each electrode alone at unit amplitude →
  verifies per-electrode axon streak shape, position, `rho`/`axlambda` decay. This is the
  most diagnostic test (isolates `W`, `axon_sens`, `pix2axon`).
- **Pairs / superposition:** two electrodes → verifies the sum-over-electrodes and the
  max-over-segments interaction.
- **Random amplitude vectors** (uniform, sparse) — fuzz over many seeds.
- **Real action distribution:** sample actions from `action_space.sample()` and from the
  greedy `sota_policy` / a checkpointed agent, so parity is measured on the *actual*
  on-policy stimulus distribution, not just synthetic ones.
- Sweep `(rho, axlambda)` over the configs used in experiments (e.g. rho∈{150,200},
  axlambda∈{200,500}) and both implants (`argus`, custom iBionics array).

### 6.4 Harness

Extend `src/test_env.py` (already the env smoke-test and already times steps) with a
`--parity` mode, or add `src/test_axonmap_parity.py`:
- builds both backends from one `env_config`,
- runs the stimulus battery,
- asserts the §6.2 tolerances,
- prints/【saves】 error maps and a timing comparison (reuse the existing
  "Time elapsed generating data" instrumentation for the speedup number).
Wire it into CI so regressions are caught.

---

## 7. Milestones

| # | Milestone | Scope | Rough effort | Risks / unknowns |
|---|---|---|---|---|
| **M0** | **Baseline + harvest** | Instrument `test_env.py` for per-step percept timing on current CPU path; extract static tensors (segments, sensitivities, grid, electrode xy) from a built p2p model and dump shapes. | 0.5–1 day | p2p internal attribute names/structure for the pickled axon map; confirm grid↔pixel mapping. |
| **M1** | **Parity prototype** | Single-stimulus `TorchAxonMap.forward`; precompute `W`, `axon_sens`, `pix2axon`; pass §6 parity (single-electrode + pairs + random) on the native grid. | 2–4 days | `max`-by-abs vs sum semantics; thresholding; electrode ordering; float32 drift. |
| **M2** | **Batched / vectorized kernel** | `(B,E)→(B,H,W)` batched einsum; prune `W` to local support (sparse) for memory; `torch.compile`; micro-benchmark vs CPU at B∈{1,8,64}. | 2–3 days | `W` memory (`A·S·E`) without pruning; sparse vs dense tradeoff on GPU; pruning tolerance vs accuracy. |
| **M3** | **Integrated env** | `percept_backend` switch; replace `action2percept` body (Level A per-env); `_action_to_amplitudes` parity (§6.1); full end-to-end env parity (post-resize) and a short SAC smoke-run for stability. | 2–3 days | Amplitude-construction parity (`ImageStimulus` resampling, `n_action_bundle`); per-worker GPU sharing under `num_gpus_per_worker=1/8`. |
| **M4** | **Differentiable extension** | Expose differentiable-env mode; implement a gradient-descent stimulus-optimization baseline (`argmin_amp L`); optional soft-threshold; (stretch) wire into a value-gradient/world-model experiment. | 3–5 days | Subgradient through `max`/threshold; whether MBRL track adopts it; soft-threshold parity vs hard-threshold oracle. |
| **M5** | **Benchmark + Level-B vectorized env** | True batched `VectorEnv` so all envs in a worker share one kernel launch; end-to-end training-throughput benchmark (steps/s, episodes/h) vs CPU baseline; report the realized speedup; parity re-check. | 2–4 days | Batched env wiring into RLlib; reward/canvas batching; reproducing learning curves (parity must hold or curves diverge). |

**Sequencing:** M0→M1→M2→M3 delivers the speedup and is the critical path. M4 (differentiability) and the Level-B half of M5 are higher-value-but-optional follow-ons. **Total critical-path ~1.5–2.5 weeks of focused work; full plan ~3–4 weeks.**

### Cross-cutting risks / unknowns
- **Amplitude semantics** are the #1 parity risk: exactly reproducing how an `action`
  becomes per-electrode amplitudes (image rasterization over the electrode grid,
  `electrode_activation_level`, `n_action_bundle`). Validate first (§6.1).
- **p2p internals drift:** harvesting static tensors couples to pulse2percept's internal
  layout; pin the p2p commit (the repo already pins one in `requirements.txt`).
- **`W` memory:** `A·S·E` (e.g. 1000·500·60 ≈ 3·10⁷ float32 ≈ 120 MB dense) is fine for
  Argus but grows with the iBionics array (E≈288) and finer grids; pruning to local
  radial support is the mitigation.
- **GPU contention:** 64 envs sharing GPUs via `num_gpus_per_worker=1/8`; batching
  (Level B) is the real fix, Level A must stay lightweight.
- **Numerical:** float32 vs p2p float64 and the hard threshold cause small, expected
  diffs; keep tolerances at §6.2 and treat larger diffs as bugs.

---

## Sources

- AxonMap model overview & equations — pulse2percept docs, *Beyeler et al. (2019):
  Axonal streaks with the axon map model*:
  https://pulse2percept.readthedocs.io/en/latest/examples/models/plot_beyeler2019_axonmap.html
- AxonMap implementation (Jansonius trace, `calc_axon_sensitivity`, `_build`,
  `_predict_spatial`) — `pulse2percept/models/beyeler2019.py`:
  https://github.com/pulse2percept/pulse2percept/blob/master/pulse2percept/models/beyeler2019.py
- Combination rule (sum over electrodes, max over segments, threshold) — compiled kernel
  `pulse2percept/models/_beyeler2019.pyx` (`fast_axon_map`):
  https://github.com/pulse2percept/pulse2percept/blob/master/pulse2percept/models/_beyeler2019.pyx
- Jansonius nerve-fiber-bundle model — Jansonius et al. (2009), *Vision Research*
  (the trajectory equations pulse2percept implements).
- Beyeler et al. (2019), *A model of ganglion axon pathways accounts for distortions of
  the perceptual space of prosthetic vision*, bioRxiv/eLife (the AxonMap model).
- Differentiable phosphene simulation prior art — van der Grinten et al. (2024),
  *Towards biologically plausible phosphene simulation for the differentiable
  optimization of visual cortical prostheses*, eLife 13:e85812 (the **dynaphos** PyTorch
  simulator): https://elifesciences.org/articles/85812 ;
  code: https://github.com/neuralcodinglab/dynaphos
- pulse2percept project (engines: cython/joblib/serial — CPU only, no JAX/Torch GPU
  backend as of current releases): https://github.com/pulse2percept/pulse2percept
