Project: Unsupervised Physics-Informed Neural Reconstruction for Lensless/Holographic Imaging
==========================================================================================

Current completed components:
1. Fresnel diffraction digital twin using FFT propagation.
2. Misalignment modelling: wavefront tilt and distance error.
3. Noise robustness experiments for sim-to-real gap.
4. Residual-vs-misalignment diagnostics.
5. Unsupervised inverse reconstruction from intensity-only measurements.
6. Multi-distance reconstruction.
7. Physical priors: support mask, amplitude-only prior, TV regularisation, bounded amplitude.
8. Ablation comparison of reconstruction methods.

Important interpretation:
- Physics-only reconstruction is ambiguous because intensity loses phase.
- Multi-distance measurements improve identifiability.
- Support and amplitude priors improve physical plausibility.
- TV improves smoothness.
- Bounded amplitude improves realism but may increase measurement loss.

Last saved: 2026-07-01_02-48-38
