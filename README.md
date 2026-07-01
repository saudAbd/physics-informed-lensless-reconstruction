[README (1).md](https://github.com/user-attachments/files/29531872/README.1.md)
# Physics-Informed Lensless Reconstruction: Unsupervised Fresnel Digital Twin and Inverse Imaging Prototype

This repository contains an ongoing research prototype for:

**Unsupervised Physics-Informed Neural Reconstruction for Lensless and Holographic Imaging**

The project investigates whether diffraction intensity measurements can be used to reconstruct object amplitude and phase without labelled training pairs, using differentiable optical physics, multi-distance measurements, and physically meaningful priors.

> **Project status:** This work is currently in progress. The present repository contains the Phase-1 prototype, ablation experiments, and early reconstruction pipeline. A full neural PINN/INR implementation is planned as the next stage.

---

## Project Overview

Lensless and holographic imaging systems replace conventional lenses with computational reconstruction. Instead of directly capturing a focused image, the sensor records a diffraction or holographic intensity pattern. The inverse problem is to recover the object field from this measured intensity.

This project follows a **physics-informed inverse-problem workflow**. The current prototype first builds a differentiable Fresnel propagation model in PyTorch, then uses it as a digital twin for simulation, misalignment analysis, and unsupervised reconstruction.

The central idea is:

```math
I_{pred} = |\mathcal{P}_z(u)|^2
```

where \(u\) is the estimated object field and \(\mathcal{P}_z\) is the Fresnel propagation operator. The object is optimised so that its simulated diffraction intensity matches the measured intensity.

---

## Current Focus

The project currently focuses on:

1. Building a **Fresnel diffraction digital twin**
2. Testing **misalignment sensitivity** using tilt and distance error
3. Studying **intensity-only inverse reconstruction**
4. Demonstrating ambiguity in phase retrieval
5. Improving reconstruction using physical priors:
   - multi-distance measurements
   - support constraint
   - amplitude-only prior
   - TV regularisation
   - bounded amplitude prior

---

## Main Figures

Add your saved figures here after uploading them to the repository.

Example:

```markdown
<img width="900" alt="Ablation comparison" src="docs/images/amplitude_comparison.png" />
```

Recommended figures to include:

- residual vs tilt misalignment
- residual vs distance error
- visual comparison of reconstructed amplitudes
- centre-row amplitude profile
- ablation loss bar chart

---

## Methodology

The current prototype has four main methodological components.

### 1. Fresnel Digital Twin

A differentiable FFT-based Fresnel propagation model was implemented in PyTorch:

```math
H_z(f_x,f_y)=\exp\left(-j\pi\lambda z(f_x^2+f_y^2)\right)
```

The forward model maps an object field to a sensor intensity pattern:

```math
I(x,y)=\left|\mathcal{F}^{-1}\left[\mathcal{F}\{u(x,y)\}H_z(f_x,f_y)\right]\right|^2
```

This digital twin forms the foundation of the current project.

---

### 2. Misalignment and Robustness Experiments

The simulator was extended with physically interpretable perturbations:

- **wavefront tilt** using a linear phase ramp;
- **distance error** by changing propagation distance \(z\);
- **intensity noise** to model measurement degradation and sim-to-real effects.

Residual curves were generated to show how intensity mismatch increases with tilt and distance error. These curves can be interpreted as simple optimisation-landscape diagnostics for alignment.

---

### 3. Unsupervised Inverse Reconstruction

The inverse problem was first tested by directly optimising a complex object field so that its propagated intensity matched the measured intensity.

The basic data-consistency loss was:

```math
\mathcal{L}_{data} = \|I_{pred}-I_{meas}\|^2
```

This confirmed that gradients can flow through the differentiable physics model. However, the recovered object was not unique because intensity-only measurements lose phase information.

---

### 4. Physics Priors and Ablation Study

To reduce ambiguity, several priors were added progressively:

1. **Two-distance measurements**  
   The object must explain intensity measurements at two propagation distances.

2. **Support constraint**  
   The object is restricted to a known central region.

3. **Amplitude-only prior**  
   The reconstruction is constrained to a non-negative amplitude object.

4. **TV regularisation**  
   Rapid pixel-to-pixel fluctuations are penalised.

5. **Bounded amplitude prior**  
   A sigmoid parameterisation constrains amplitude to the physical range \([0,1]\).

---

## Current Ablation Results

| Method | MSE z1 | MSE z2 | Total 2-plane loss | Amp mean | Amp max | Main Interpretation |
|---|---:|---:|---:|---:|---:|---|
| True object | 0.000000 | 0.000000 | 0.000000 | 0.026917 | 1.000 | Ground-truth reference |
| 2z + support + amp + TV | 0.000192 | 0.000336 | 0.000529 | 0.029336 | 1.711 | Best measurement fit, but unbounded peaks remain |
| 2z + support + amp | 0.000230 | 0.000413 | 0.000643 | 0.029537 | 1.596 | Strong improvement from amplitude-only prior |
| 2z + support + amp + TV + bound | 0.002490 | 0.002430 | 0.004920 | 0.026808 | 0.847 | Most physically realistic amplitude range |
| 2z + support | 0.003619 | 0.002958 | 0.006578 | 0.029278 | 1.718 | Localised but noisy and less stable |

The ablation shows a clear trade-off between **data fidelity** and **physical plausibility**. The softplus amplitude + TV version gives the lowest reconstruction loss, while the bounded sigmoid version better respects the physical aperture range.

---

## Key Findings So Far

- A differentiable Fresnel model can act as a practical digital twin for lensless imaging.
- Residual-vs-misalignment curves can be interpreted as slices through the optimisation landscape.
- Intensity-only reconstruction is ill-posed because phase is not directly measured.
- Multi-distance measurements improve identifiability but do not fully solve ambiguity.
- Physical priors significantly improve reconstruction quality.
- Bounded amplitude priors improve physical realism but may reduce perfect intensity fitting.

---

## Repository Structure

```text
physics-informed-lensless-reconstruction/
│
├── src/
│   └── prototype_core.py
│
├── figures/
│   └── saved reconstruction and ablation figures
│
├── tables/
│   └── ablation_results.csv
│
├── experiment_summaries/
│   └── experiment summary text files
│
├── notebooks_backup/
│   └── Colab prototype notebook and exported Python file
│
├── outputs/
│   └── additional saved outputs
│
├── README.md
└── requirements.txt
```

---

## How to Run

Install the main dependencies:

```bash
pip install torch matplotlib pandas numpy
```

Suggested workflow:

1. Open the prototype notebook in Google Colab.
2. Run the Fresnel digital twin cells.
3. Generate the square aperture object.
4. Run misalignment experiments for tilt and distance error.
5. Run inverse reconstruction experiments.
6. Add priors progressively.
7. Generate the ablation table and comparison figures.

A cleaner script-based workflow is currently being organised under `src/` and future `experiments/` folders.

---

## Current Limitations

This repository is a research prototype, not a completed reconstruction framework. Current limitations include:

- experiments currently use a synthetic square aperture;
- reconstruction is based on direct optimisation rather than a trained PINN/INR network;
- amplitude and phase recovery is not yet fully generalised;
- no real holographic or lensless dataset has been evaluated yet;
- no Gerchberg–Saxton or classical phase-retrieval baseline has been added yet;
- deployment/API packaging is not part of the current phase.

---

## Next Steps

Planned next steps include:

- cleaning the notebook into modular scripts;
- adding a Gerchberg–Saxton baseline;
- replacing pixel-wise optimisation with a neural parameterisation;
- testing on digit or simple shape datasets;
- extending to multi-wavelength and multi-distance reconstruction;
- adding PSNR and SSIM evaluation where ground truth is available;
- preparing the project for GitHub release and future deployment experiments.

Longer term, the reconstruction module may be wrapped as an API service or containerised inference component, but the current focus is on validating the physics-informed reconstruction pipeline first.

---

## Main Conclusion

The current prototype shows that physics-based differentiable simulation is a strong foundation for unsupervised lensless reconstruction, but physics alone is not enough for a stable inverse solution. Because intensity-only imaging loses phase, the reconstruction problem is ambiguous unless additional constraints are introduced.

The main lesson so far is that **physics consistency plus appropriate priors** improves identifiability. Multi-distance measurements, support constraints, amplitude priors, and TV regularisation progressively improve reconstruction quality and reveal the trade-off between matching the measured data and enforcing physically realistic object structure.

This repository should therefore be interpreted as an **in-progress physics-informed inverse imaging prototype**, forming the foundation for a future PINN/INR-based lensless reconstruction system.

---

## Author

**Saud Abdullah**  
Computational Imaging · Physics-Informed AI · Computer Vision
