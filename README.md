# nucleus-sev-autoencoder
Unsupervised clean pulse selection for Standard Event (SEV) construction for cryogenic TES detectors using autoencoders.

nucleus-sev-autoencoder/
## Repository Structure

```text
.
├── README.md
├── find_linear_range.py
├── pulse_selection_autoencoder.py
└── figures/
    ├── sev_comparison_configA.png
    ├── sev_comparison_configB.png
    ├── sev_comparison_configC.png
    └── sev_comparison_configE.png
```

### Files

* `find_linear_range.py` — Determines the detector linear response range using rise-time and decay-time stability.
* `pulse_selection_autoencoder.py` — Autoencoder-based pulse quality selection and event classification.
* `report/report.pdf` — Full project report describing methodology, implementation, and results.
* `figures/` — Example SEV comparisons and diagnostic plots.

```
```
## Example Usage

```python
linear_range = find_linear_range(dh, ch=0)

errs = pulse_selection_with_autoencoder(
    dh,
    pulse_type="event",
    ch=0,
    linear_range=linear_range
)

dh.calc_sev(
    type="events",
    use_cuts=["autoencoder_cut_0"]
)
```
## Acknowledgements

Developed as part of research activities within the NUCLEUS experiment at the E15 Chair at the Technical University of Munich (TUM).
