# Data directory

This repository does **not** ship data files. This document defines where each
data file goes and how it is named, so the pipeline scripts and a fresh clone
agree on the layout.

## Layout

```
data/
├── stim_ERB_B1.hdf5            # B1 stimulus set (see table below)
├── stim_ERB_B1_fine.hdf5
├── stim_ERB_B2.hdf5
├── stim_ERB_B2_v2.hdf5
├── stim_ERB_B3.hdf5
├── stim_ERB_B4.hdf5
├── stim_ERB_B5.hdf5
├── stim_ERB_B6.hdf5
├── nervegrams/
│   ├── nervegram_stim_ERB_B6.hdf5        # B6 gate nervegrams
│   └── josh_dnns/
│       ├── B1/
│       │   └── nervegram_stim_ERB_B1_part{1..4}.hdf5   # B1 full-set nervegrams, batched
│       └── QC/
│           └── nervegram_qc_{Bach10,SpeechSynth}_{clean,-10,-5,0,5,10,15,20,25,30}.hdf5
├── predictions/                # per-model inference outputs (700-class pitch-probability vectors)
└── results/                    # aggregated metrics, perceptual-ERB scatter, F0–N heatmap source tables
```

## Stimulus files

| File | Experiment | Content |
|---|---|---|
| `stim_ERB_B1.hdf5` | B1 F0DL | 397,364 harmonic-complex trials across the F0–N plane; 150 ms @ 32 kHz (key `tone_in_noise`, shape `(397364, 4800)` float32) |
| `stim_ERB_B1_fine.hdf5` | B1 fine variant | denser F0 sampling of the same paradigm |
| `stim_ERB_B2.hdf5` / `stim_ERB_B2_v2.hdf5` | B2 pitch doubling | octave-judgment trials (v2 = revised design) |
| `stim_ERB_B3.hdf5` | B3 shift slope | frequency-shifted complex trials |
| `stim_ERB_B4.hdf5` | B4 mistuned harmonics | mistuned-harmonic trials |
| `stim_ERB_B5.hdf5` | B5 transposed tones | transposed-tone trials |
| `stim_ERB_B6.hdf5` | B6 gate (B1 paradigm) | 8,470-trial gate set used with auditory-nerve nervegrams |

## Nervegrams

Auditory-nerve model outputs (BEZ2018 peripheral model), shape
`(n_trials, 100, 1000)` float32 = 100 characteristic frequencies × 1000
timesteps @ 20 kHz (50 ms core), firing rates in Hz. Configuration is locked
to: `cf_list=None + num_cf=100` (human ERB / Greenwood map), IHC low-pass
3 kHz / 7th order, 70/10 ms buffers, 40 spike trains, `species=2`.

- `nervegrams/nervegram_stim_ERB_B6.hdf5` — B6 gate set (8,470 trials), the
  consistency anchor for all downstream nervegram generation.
- `nervegrams/josh_dnns/B1/` — B1 full set split into 4 parts of 99,341 trials
  (`part1`–`part4`), for evaluating deep pitch models trained on nervegrams.
- `nervegrams/josh_dnns/QC/` — CHiME-Home noise-robustness sets, one file per
  dataset (`Bach10`, `SpeechSynth`) × condition (`clean` and 9 SNR levels:
  −10, −5, 0, +5, +10, +15, +20, +25, +30 dB).

## Predictions and results

- `predictions/` — raw model outputs: per trial, a 700-class probability vector
  over the pitch grid (80–1000 Hz, 1/192-octave bins), one file per
  (experiment, model). Written by the wrappers in `predictors/`.
- `results/` — derived tables (F0DL values, doubling ratios, shift slopes,
  perceptual-ERB scatters, F0–N heatmap source data) written by the scripts in
  `analysis/` and the `run_*` drivers.

> These two directories are outputs of the pipeline; re-create them by running
> the code rather than copying them.
