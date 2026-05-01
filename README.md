# EEG-Quantum-Correlations

Reproduction repository ([GitHub](https://github.com/SATORU-SIEL/EEG-Quantum-Correlations/)) for the paper:

**Intersection-Defined Phase Coordinates Reveal Localized Selection and a Non-Closed Observational Structure**  
https://doi.org/10.5281/zenodo.19956935

This repository provides the notebooks and precomputed CSV artifacts needed to reproduce the public figures and to rerun the main analysis pipeline.

## Quick Demo (Browser)

Run the Binder demo:

https://mybinder.org/v2/gh/SATORU-SIEL/EEG-Quantum-Correlations/HEAD?urlpath=%2Fdoc%2Ftree%2Fbinder_demo_5figures_precomputed.ipynb

## Repository Structure

```text
EEG-Quantum-Correlations/
├── binder_demo_5figures_precomputed.ipynb      # Binder demo (precomputed CSV only)
├── public_repro_5figures_from_experiments.ipynb # Public reproduction notebook
├── EEG_Quantum_Corr.ipynb                       # Full analysis notebook
├── Quantum Mesurement.ipynb                     # Quantum measurement notebook
├── requirements.txt                             # Dependencies
├── Repro_CSV/                                   # Bundled CSV files for demo/reproduction
├── Repro_Figure/                                # Output directory for regenerated figures
```

## Full Reproduction

To regenerate the five public figures from the embedded experiment registry already bundled in this repository:

1. Open `public_repro_5figures_from_experiments.ipynb`
2. Run the notebook from top to bottom

This notebook supports the main public frame switches:

- correlation feature family: `60corr` or `20corr`
- task regime: `26task` or `30task`
- correlation frame: `4ch` or `14ch`

## Raw-Data Reproduction

Raw EEG dataset:

**Raw EEG Data (26 Sessions) for Reproduction of Neural–Quantum Structural Analysis in IDPC**  
https://doi.org/10.5281/zenodo.19624924

To rerun the broader analysis pipeline from the full session data:

1. Download the dataset from Zenodo
2. Open `EEG_Quantum_Corr.ipynb`
3. Run the notebook from top to bottom

## Notes

- CSV files included in this repository are sufficient for the Binder demo and public figure reproduction
- Full raw-data reproduction requires the EEG dataset described in the paper
- `Quantum Mesurement.ipynb` is required only when reproducing the quantum measurement side of the study
