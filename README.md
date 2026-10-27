# jvmc-2.0-data

Data, scripts and figures for the jVMC 2.0 paper.

## Layout

```
paper/
├── Scripts/          # produce the data
├── Data/             # results of the scripts
├── Figures/          # PDFs made by the plot_*.py scripts
├── plot_config.py    # style shared by all figures
└── plot_*.py         # one script per figure
```

The libraries and versions are in `requirements.txt` (`pip install -r requirements.txt`).

| Script | Data | Figures |
|---|---|---|
| `compare_netket.py` | `compare_netket/` | `netket_comparison.pdf` |
| `compare_netket_sr.py` | `compare_netket_sr/`, `compare_netket_sr_4gpus/` | `netket_comparison_sr_*.pdf` |
| `compare_netket_minsr.py` | `compare_netket_minsr/`, `compare_netket_minsr_4gpus/` | `netket_comparison_minsr_*.pdf` |
| `compare_netket_sr.py` + `compare_netket_minsr.py` | both of the above | `netket_comparison_sr_minsr_*.pdf` |
| `quantum_dot_2d.py` | `quantum_dot_2d.h5` | `quantum_dot_2d.pdf` |

- **compare_netket**: time of sampling and of the estimate of ⟨H⟩, jVMC_exp vs NetKet.
- **compare_netket_sr**: time and peak memory of one SR step for a growing number of samples.
- **compare_netket_minsr**: time and peak memory of one MinSR step for a growing number of parameters.
- **quantum_dot_2d**: VMC ground state of 6 electrons in a 2D harmonic trap, compared with FCI.

The NetKet comparisons were run on NVIDIA GH200 GPUs (1 GPU, and 4 GPUs for the `_4gpus` folders)