
"""
Project: Unsupervised Physics-Informed Neural Reconstruction for Lensless/Holographic Imaging

Core prototype code:
- Fresnel FFT propagation digital twin
- Misalignment modelling
- Noise model
- Object/support generation
- TV regularisation
- Evaluation helpers

Author: Saud Abdullah
"""

import math
import torch


# ============================================================
# 1. Frequency and coordinate helpers
# ============================================================

def make_f2(N, dx, device="cpu"):
    """
    Build spatial-frequency squared grid:
        f2 = fx^2 + fy^2

    N: grid size
    dx: pixel pitch in metres
    """
    fx = torch.fft.fftfreq(N, d=dx, device=device).reshape(1, -1)
    fy = torch.fft.fftfreq(N, d=dx, device=device).reshape(-1, 1)
    return fx**2 + fy**2


def make_xy(N, dx, device="cpu"):
    """
    Build real-space coordinate grids X, Y in metres.
    """
    x = (torch.arange(N, device=device) - N // 2) * dx
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return X, Y


# ============================================================
# 2. Fresnel propagation digital twin
# ============================================================

def fresnel_propagate(u0, wavelength, z, dx, sigma_f=None):
    """
    Fresnel propagation using the FFT transfer-function method.

    u0:
        Complex object field [N, N]

    wavelength:
        Optical wavelength in metres

    z:
        Propagation distance in metres

    dx:
        Pixel pitch in metres

    sigma_f:
        Optional Gaussian MTF width in cycles/metre.
        If None, ideal phase-only propagation is used.

    Returns:
        Uz: propagated complex field
        I: intensity = |Uz|^2
    """
    u0 = u0.to(torch.complex64)
    N = u0.shape[0]
    f2 = make_f2(N, dx, device=u0.device)

    # Fresnel transfer function:
    # H_z(fx, fy) = exp(-j*pi*lambda*z*(fx^2 + fy^2))
    Hz = torch.exp(-1j * math.pi * wavelength * z * f2)

    if sigma_f is not None:
        # Gaussian MTF / low-pass blur model
        M = torch.exp(-f2 / (2.0 * sigma_f**2))
        H = Hz * M
    else:
        H = Hz

    Uz = torch.fft.ifft2(torch.fft.fft2(u0) * H)
    I = Uz.real**2 + Uz.imag**2

    return Uz, I


# ============================================================
# 3. Misalignment and sensor realism models
# ============================================================

def add_tilt(u, alpha, beta, dx, wavelength):
    """
    Add wavefront tilt using a linear phase ramp.

    alpha, beta:
        Small tilt parameters.
        These simulate angular misalignment, analogous to mirror tilt.
    """
    u = u.to(torch.complex64)
    N = u.shape[0]
    X, Y = make_xy(N, dx, device=u.device)

    phase = (2 * math.pi / wavelength) * (alpha * X + beta * Y)
    ramp = torch.exp(1j * phase)

    return u * ramp


def add_intensity_noise(I, sigma):
    """
    Add simple Gaussian noise to intensity and clamp to non-negative values.
    """
    return (I + sigma * torch.randn_like(I)).clamp(min=0)


# ============================================================
# 4. Object and support generation
# ============================================================

def make_square_aperture(N=512, square_fraction=12, device="cpu"):
    """
    Create a centred square aperture as a complex amplitude-only object.

    square_fraction:
        Square half-width is N // square_fraction.
    """
    u = torch.zeros((N, N), dtype=torch.complex64, device=device)

    s = N // square_fraction
    u[N//2 - s:N//2 + s, N//2 - s:N//2 + s] = 1.0 + 0.0j

    return u


def make_support_mask(N=512, support_fraction=10, device="cpu"):
    """
    Create a centred binary support mask.

    support_fraction:
        Support half-width is N // support_fraction.
    """
    support = torch.zeros((N, N), dtype=torch.float32, device=device)

    s = N // support_fraction
    support[N//2 - s:N//2 + s, N//2 - s:N//2 + s] = 1.0

    return support


# ============================================================
# 5. Priors and regularisation
# ============================================================

def tv_loss(x):
    """
    Total Variation loss.
    Penalises rapid pixel-to-pixel changes.
    """
    dx_tv = torch.abs(x[:, 1:] - x[:, :-1]).mean()
    dy_tv = torch.abs(x[1:, :] - x[:-1, :]).mean()
    return dx_tv + dy_tv


# ============================================================
# 6. Evaluation helpers
# ============================================================

def evaluate_reconstruction(u, I_meas1, I_meas2, wavelength, z1, z2, dx):
    """
    Evaluate reconstruction against two measured intensity planes.
    """
    amp = u.abs()

    _, I_pred1 = fresnel_propagate(u, wavelength, z1, dx, sigma_f=None)
    _, I_pred2 = fresnel_propagate(u, wavelength, z2, dx, sigma_f=None)

    mse_z1 = ((I_pred1 - I_meas1)**2).mean().item()
    mse_z2 = ((I_pred2 - I_meas2)**2).mean().item()

    return {
        "MSE z1": mse_z1,
        "MSE z2": mse_z2,
        "Total 2-plane loss": mse_z1 + mse_z2,
        "Amp mean": amp.mean().item(),
        "Amp max": amp.max().item(),
        "Amp min": amp.min().item(),
        "TV": tv_loss(amp).item(),
    }
