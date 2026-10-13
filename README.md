# pe-models-behavioral-evaluation

**Behavioral benchmarking of pitch-estimation models against human pitch perception,**
with model-derived estimates of the perceptual equivalent rectangular bandwidth (ERB).

*Zhou Yifei, Institute of Automation, Chinese Academy of Sciences (CASIA)*

This repository accompanies our ICASSP 2027 paper on human-guided evaluation and
data augmentation for pitch-estimation models. It contains the methodology and
research code that turns off-the-shelf pitch-estimation models into *psychophysical
observers*: we present the same stimuli that were used in classic human pitch
experiments to each model, extract its "perceptual" frequency resolution (ERB),
and compare it against human behavior — then use the behavioral deficits we find
to steer data augmentation.

> **Scope.** This repository is a *methods and research-idea* release. Result
> data (model predictions, final metric tables) and the training datasets are
> **not** included. Large assets are documented in [`data/README.md`](data/README.md)
> with their expected names and locations; place your own copies there.

---

## What's inside

| Path | Content |
|---|---|
| `stimuli/` | Stimulus generation for the psychophysical paradigms (Bernstein & Oxenham 2005 F0DL, pitch doubling, frequency-shifted complexes, mistuned harmonics, transposed tones) — outputs HDF5 files described in `data/README.md` |
| `predictors/` | Wrapper evaluators for **11 pitch-estimation models** (CREPE, FCN-F0, FCPE, FTANet, HARMOF0, JDC, JoshCNN, PENN, PESTO, RMVPE, SwiftF0). Each wrapper runs the model over a stimulus file and stores 700-class pitch-probability vectors |
| `analysis/` | Behavioral analysis pipeline: F0DL computation, pitch-doubling ratio, pitch-shift slope, per-F0 reanalysis, perceptual-ERB extraction, F0–N heatmap construction, and behavior/engineering correlation |
| `run_strategy_A.py` | Zero-new-data strategy: reanalyze existing model predictions per F0 to derive perceptual-ERB scatter and cross-validate across paradigms |
| `run_strategy_B.py` | New-stimulus pipeline driver: generate stimuli → run model inference → compute behavioral metrics → extract perceptual ERB |
| `run_phase4_6.py` | Cross-model ERB consensus, behavior–engineering correlation, and architecture attribution |
| `generate_plots.py` | Figure driver for the F0–N behavioral heatmaps (per experiment and model) |
| `nervegram_example.py` / `nervegram_generate_B6.py` | Auditory-nerve (AN) model nervegram generation (BEZ2018 peripheral model; 100 CF × 1000 timesteps @ 20 kHz, 50 ms core) — used to feed deep models that operate on neural representations (e.g. PitchNet, *Saddler et al.*, Nat. Commun. 2021) |
| `data/` | **Placeholder directory only** — see `data/README.md` for where each data file goes and how it is named |

## Experiments at a glance

- **Strategy A** — reanalysis of predictions from the 2021 Josh et al. behavioral
  dataset, no new data required.
- **B1** — F0-discrimination limens (F0DL) across the F0–N plane
  (Bernstein & Oxenham 2005 two-interval paradigm).
- **B2** — pitch-doubling ratio (octave judgments).
- **B3** — pitch-shift slope for frequency-shifted complexes.
- **B4** — mistuned-harmonic sensitivity.
- **B5** — transposed-tone pitch perception.
- **B6** — gate experiment on the B1 paradigm with AN nervegrams as model input
  (deep pitch models trained on neural representations).
- **QC** — engineering benchmark under CHiME-Home noise (−10 … +30 dB SNR),
  correlated against the behavioral metrics.

## Quick start

```bash
# 1. (optional) generate stimuli into data/
python run_strategy_B.py --step generate

# 2. run one model over the stimuli (adapt paths inside the evaluators first)
python predictors/CREPE/PSYCHO_EVALUATOR_CREPE.py --data data/stim_ERB_B1.hdf5

# 3. compute behavioral metrics
python analysis/compute_F0DL.py --predictions data/predictions/...

# 4. plot the F0–N heatmaps
python generate_plots.py
```

Dependencies: `numpy`, `scipy`, `h5py`, `soundfile`, plus the deep-learning
framework required by each model wrapper (PyTorch or TensorFlow). Model
**checkpoints are not included** — download them from each model's official
repository and point the wrapper to the checkpoint path.

## Notes for re-use

- The scripts were written for our internal compute environment and contain
  hard-coded paths (e.g. `/data/zyf/...`). Search-and-replace them with your
  own layout before running.
- The AN-model nervegram conversion runs the BEZ2018 peripheral model
  (Bruce, Erfani & Zilany 2018); see the header comments of
  `nervegram_generate_B6.py` for the locked configuration.

## Citation

Paper in preparation (submitted to ICASSP 2027). Citation information will be
added upon publication.

## License

[MIT](LICENSE) © 2026 Zhou Yifei.
