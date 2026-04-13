#!/usr/bin/env python3
"""
Magnetron Design Calculator
============================
Physics-based design tool implementing:
  - Carter (2018) §15 equations (15.22, 15.23, 15.40, 15.49, 15.88, 15.93, 15.154)
  - Collins & Clogston, MIT Radiation Laboratory Series Vol.6 (1948) §16
  - McDowell, IEEE Trans. Plasma Sci. 26, 733-754 (1998)
  - Liu et al., MetalMat (2024) — cathode emission limits
  - Real datasheet values: Teledyne e2v MG5193/MG7095, CPI VMC3105/VMC3109, 4J50/4J52

Usage examples
--------------
  python3 magnetron_design.py                          # interactive prompts
    python3 magnetron_design.py --flask                 # start Flask API server
    curl "http://127.0.0.1:5000/calculate?freq=9.375&power=250&type=s_pls&cath=disp"
  python3 magnetron_design.py --preset linac13         # 1.3 GHz linac 50 kW
  python3 magnetron_design.py --preset linac_s         # S-band linac 2.6 MW
  python3 magnetron_design.py --preset radar_x         # X-band radar 250 kW
  python3 magnetron_design.py --preset oven            # CW industrial 6 kW
  python3 magnetron_design.py --preset ka_rs           # Ka-band rising sun 80 kW
  python3 magnetron_design.py -f 9.375 -p 250 -t s_pls --cath disp
  python3 magnetron_design.py -f 1.3 -p 50 -t coa --cath disp --etac 88
  python3 magnetron_design.py --list-db                # show reference magnetron database
  python3 magnetron_design.py --load-db mg5193         # load MG5193 specs
  python3 magnetron_design.py -f 1.3 -p 50 -t coa --compare-types
  python3 magnetron_design.py -f 9.375 -p 250 -t s_pls --no-color
"""

import argparse
import json
import math
import sys
import os
from functools import lru_cache
from datetime import datetime, timezone

try:
    from flask import Flask, jsonify, request, render_template
except ImportError:
    Flask = None
    jsonify = None
    request = None
    render_template = None

# ─── Physical constants ──────────────────────────────────────────────────────
EOM   = 1.75882e11   # e/m₀  (C/kg)
EPS0  = 8.854e-12    # vacuum permittivity (F/m)
MU0   = 4*math.pi*1e-7  # vacuum permeability (H/m)
C     = 3e8          # speed of light (m/s)
RCU   = 1.72e-8      # Cu resistivity at 20°C (Ω·m)

# ─── Thermal / vane-tip temperature (Carter §15.7, Eq. 15.155–15.156) ───────
COPPER_KAPPA = 401.0  # W/(m·K) thermal conductivity of OFHC copper
T_COOL_K = 293.0      # K (≈20°C) cooling-channel reference temperature
L_TH_M = 0.010        # m effective thermal path to cooling channel (Carter example)

# Carter notes: avoid recrystallization above ~700°C; use 500°C for margin.
T_VANE_WARN_K = 773.15   # 500°C
T_VANE_FATAL_K = 973.15  # 700°C

# ─── Collins Fig. 10-9: analytic fit for a1(r_a/r_c) ────────────────────────
# Used in Collins Eq. (26c) for the characteristic current scale.
#
# The original Fig. 10-9 curve was digitized and then least-squares fit with a
# degree-8 polynomial in a normalized log-domain variable:
#
#   u = 2 * (ln(x) - ln_min) / (ln_max - ln_min) - 1,  where x = r_a/r_c.
#   a1(x) ≈ sum_{k=0..8} c[k] * u^k
#
# This avoids per-call table lookups/interpolation while preserving the curve.
_A1_FIT_LN_MIN = 0.05158720119271373   # ln(1.052941)
_A1_FIT_LN_MAX = 1.7832002769594129    # ln(5.948864)
_A1_FIT_COEFFS_U = [
    0.9480964736150357,
    0.1343167961175846,
    0.3385982556573334,
    -0.02378311006723695,
    -0.06720129636763175,
    -0.09976163133958345,
    0.20298030313882626,
    0.09464202504274469,
    -0.11217668407614961,
]


def _poly_eval(coeffs, x):
    """Evaluate polynomial with coeffs in increasing order via Horner's rule."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _a1_ra_over_rc(ra_over_rc: float) -> float:
    """Return a1(r_a/r_c) from an analytic fit to Collins Fig. 10-9."""
    if ra_over_rc is None or not math.isfinite(ra_over_rc) or ra_over_rc <= 0:
        return float("nan")

    # Match the old interpolation behavior outside the digitized domain by
    # clamping to the fit range.
    ln_x = math.log(ra_over_rc)
    if ln_x <= _A1_FIT_LN_MIN:
        u = -1.0
    elif ln_x >= _A1_FIT_LN_MAX:
        u = 1.0
    else:
        u = 2.0 * (ln_x - _A1_FIT_LN_MIN) / (_A1_FIT_LN_MAX - _A1_FIT_LN_MIN) - 1.0

    return _poly_eval(_A1_FIT_COEFFS_U, u)


def _vane_tip_temperature_K(Pa_W: float, Nv: int, t_vane_m: float, La_m: float) -> float:
    """Return vane-tip temperature estimate (K) using Carter Eq. 15.156.

    Tv = Tr + (Pa * Lth) / (Av * kappa)

    Where:
      - Pa is anode dissipation power (W), approximated as Pdc - Prf (Eq. 15.155)
      - Lth is effective thermal path length to cooling channel
      - Av is total vane cross-sectional area normal to heat flow

    This is intended for high-power CW designs where vane-tip heating limits.
    """
    if Pa_W is None or not math.isfinite(Pa_W) or Pa_W <= 0:
        return float("nan")
    if Nv is None or Nv <= 0:
        return float("nan")
    if t_vane_m is None or La_m is None:
        return float("nan")
    if not (math.isfinite(t_vane_m) and math.isfinite(La_m)):
        return float("nan")
    if t_vane_m <= 0 or La_m <= 0:
        return float("nan")

    Av_m2 = Nv * t_vane_m * La_m
    if Av_m2 <= 0:
        return float("nan")

    return T_COOL_K + (Pa_W * L_TH_M) / (Av_m2 * COPPER_KAPPA)

# ─── ANSI colour helpers ──────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def cyan(t):    return _c("36", t)
def bold(t):    return _c("1",  t)
def blue(t):    return _c("34", t)
def magenta(t): return _c("35", t)
def dim(t):     return _c("2",  t)

def color_val(val, ok_low, ok_high, warn_low=None, warn_high=None, flip=False):
    """Colour a value green/yellow/red based on range thresholds."""
    if val is None or not math.isfinite(val):
        return dim("—")
    if flip:
        if val <= ok_high:   return green(f"{val:.3g}")
        if warn_high and val <= warn_high: return yellow(f"{val:.3g}")
        return red(f"{val:.3g}")
    else:
        if ok_low <= val <= ok_high:          return green(f"{val:.3g}")
        if warn_low is not None and warn_high is not None:
            if warn_low <= val <= warn_high:  return yellow(f"{val:.3g}")
        return yellow(f"{val:.3g}")

def hr(char="─", width=90):
    return dim(char * width)

# ─── Cathode material database ────────────────────────────────────────────────
CATHODES = {
    "oxide": {
        "label":  "Oxide (BaO)",
        "Jcw":    0.4,    # max CW current density (A/cm²)
        "Jpulse": 8.0,    # max pulse current density (A/cm²)
        "Tmax":   900,    # max operating temperature (°C)
        "note": (
            "Work function φ ≈ 1.0–1.8 eV. Excellent pulsed emission (to ~8 A/cm²). "
            "Sensitive to back-bombardment — oxide layer degrades over time. "
            "Requires vacuum ≤ 10⁻⁷ torr. Common in low-power/pulsed magnetrons."
        ),
    },
    "disp": {
        "label":  "Dispenser (Ba-W, B-type)",
        "Jcw":    5.0,
        "Jpulse": 30.0,
        "Tmax":   1100,
        "note": (
            "BaO-CaO-Al₂O₃ impregnated porous W matrix. J_c = 1–15 A/cm² CW, "
            "5–30 A/cm² pulsed. Secondary emission coefficient δ_m = 2–4. "
            "Back-bombardment consumes BaO reservoir — constrains duty cycle. "
            "Dominant in high-power pulsed radar/linac magnetrons. (Liu et al. 2024)"
        ),
    },
    "thw": {
        "label":  "Thoriated W (Th-W)",
        "Jcw":    1.0,
        "Jpulse": 5.0,
        "Tmax":   1900,
        "note": (
            "ThO₂ surface monolayer reduces φ to ~2.63 eV. T_op = 1600–1900°C. "
            "High back-bombardment heat tolerance. Preferred CW industrial/oven cathode "
            "(Carter §18.5.1). Rare-earth alternatives (La₂O₃, Y₂O₃) avoid radioactivity "
            "with comparable emission. (Liu et al. 2024)"
        ),
    },
    "matrix": {
        "label":  "Matrix / Scandate",
        "Jcw":    7.0,
        "Jpulse": 40.0,
        "Tmax":   1100,
        "note": (
            "Sc₂O₃ or rare-earth doped porous W matrix. Highest CW emission 5–8 A/cm²; "
            "δ_m = 3.5–5. Lowest startup noise due to high emission uniformity. "
            "Requires careful conditioning and ultra-high vacuum. (Liu et al. MetalMat 2024)"
        ),
    },
}

# ─── Magnetron type database ──────────────────────────────────────────────────
TYPES = {
    "s_cw": {
        "label":   "Strapped CW",
        "duty":    "cw",
        "QL":      250,
        "linac_suitability": 40,
        "eta_fn":  lambda f: 87 if f < 1.5 else 67,
        "zdc_fn":  lambda f: 4.5 if f < 1.5 else 5.5,
        "pros": [
            "65–90% overall efficiency",
            "Simple, robust construction",
            "Dominant for 0.9 & 2.45 GHz ISM",
            "Permanent magnet operation typical",
        ],
        "cons": [
            "Strap manufacturing limits f ≲ 16 GHz",
            "Strap-vane voltage breakdown at high power",
            "Narrow tuning range 2–5%",
        ],
        "desc": (
            "CW operation with straps connecting alternate vanes. In the π-mode the straps "
            "are equipotential (carry no current) but perturb adjacent modes, increasing the "
            "π/π−1 frequency separation. Collins §5 derived the strap admittance. "
            "Standard design below 3 GHz. Manufacturing difficulty limits use above 16 GHz."
        ),
        "apps": "0.9 & 2.45 GHz industrial/domestic heating, medical linac injectors",
    },
    "s_pls": {
        "label":   "Strapped Pulsed",
        "duty":    "pulse",
        "QL":      250,
        "linac_suitability": 65,
        "eta_fn":  lambda f: 43 if f < 4 else (40 if f < 8 else (38 if f < 11 else 30)),
        "zdc_fn":  lambda f: 1.0 if f < 4 else (0.75 if f < 8 else (0.90 if f < 11 else 1.0)),
        "pros": [
            "1 kW–4.5 MW peak power range",
            "Proven decades-long radar heritage",
            "Compact and lightweight",
            "35–50% efficiency at L/S-band",
        ],
        "cons": [
            "Q_L ≈ 250 → pulling ~12 MHz at S-band",
            "Mode competition risk on turn-on",
            "Straps limit upper frequency to ~18 GHz",
        ],
        "desc": (
            "Most common radar magnetron from L- to Ku-band. Peak powers 1 kW–4.5 MW. "
            "Same strap mode-separation technique as CW. Q_L ≈ 250 gives frequency "
            "pulling ~12–15 MHz (S=1.5) at S-band. π-mode selection on pulse turn-on is "
            "critical. Cathode back-bombardment constrains duty cycle."
        ),
        "apps": "Radar (L/S/C/X/Ku), particle linacs, EW transmitters, medical linacs",
    },
    "rs": {
        "label":   "Rising Sun",
        "duty":    "pulse",
        "QL":      200,
        "linac_suitability": 10,
        "eta_fn":  lambda f: 22,
        "zdc_fn":  lambda f: 0.85,
        "pros": [
            "Practical at 25–35 GHz",
            "No strap voltage breakdown",
            "Large inherent mode separation",
        ],
        "cons": [
            "~20–25% efficiency only",
            "Cyclotron resonance risk: keep ωc/ω ∉ [0.75, 1.25]",
            "Avoid N_v divisible by 4",
        ],
        "desc": (
            "Alternating deep/shallow cavities (ratio d₂/d₁) provide bi-periodicity, "
            "greatly increasing π/π−1 separation without straps — essential at 25–35 GHz "
            "where strap fabrication is impractical. N_v divisible by 4 must be avoided; "
            "keep ωc/ω ∉ [0.75, 1.25] to prevent cyclotron resonance with the φ=0 mode."
        ),
        "apps": "Ka/W-band radar, cloud-profiling radar, mm-wave scientific/EW sources",
    },
    "coa": {
        "label":   "Coaxial",
        "duty":    "pulse",
        "QL":      1000,
        "linac_suitability": 90,
        "eta_fn":  lambda f: 40 if f < 12 else (30 if f < 18 else 25),
        "zdc_fn":  lambda f: 0.95 if f < 12 else (1.35 if f < 18 else 1.2),
        "pros": [
            "Q_L ≈ 1000 → pulling ≈ 3 MHz (S-band)",
            "20% mechanical tuning range",
            "Pushing ≈ 0.08 MHz/A (vs 0.4 strapped)",
            "Non-contacting TE₀₁₁ tuning plunger",
        ],
        "cons": [
            "Larger physical size for given frequency",
            "Slow anode voltage rise required",
            "Higher complexity and cost",
        ],
        "desc": (
            "External TE₀₁₁ coaxial cavity surrounds the anode and couples via slots to "
            "alternate vanes. Energy storage in the outer cavity raises Q_L to ~1000. "
            "Per Carter Table 15.1: pulling 3 vs 14 MHz, pushing 0.08 vs 0.4 MHz/A, "
            "tuning range 20% vs 10% (all relative to conventional strapped). "
            "Non-contacting plunger possible because TE₀₁₁ has no end-wall currents. "
            "PREFERRED for stability-critical linac applications."
        ),
        "apps": "ATC radar, precision navigation, stability-critical linacs, frequency-agile EW",
    },
    "la": {
        "label":   "Long Anode",
        "duty":    "pulse",
        "QL":      200,
        "linac_suitability": 30,
        "eta_fn":  lambda f: 50,
        "zdc_fn":  lambda f: 0.35,
        "pros": [
            "2–5 MW peak power achievable",
            "No straps required",
            "~45–55% efficiency",
        ],
        "cons": [
            "Very large physical size",
            "Very low Z_dc (high currents)",
            "Electromagnet required — no permanent magnet",
        ],
        "desc": (
            "Axial anode length ~λ allows MW-class peak power without increasing diameter "
            "or vane count. Symmetric coaxial output coupler connects alternate vanes and "
            "selects the π-mode via coupled-mode H_(n, N_v/2−n) pairs. Pure π-mode is "
            "maintained even with <1% adjacent-mode separation — straps not required. "
            "(Carter §15.3.4)"
        ),
        "apps": "Long-range L-band radar, high-power test transmitters, large phased arrays",
    },
}

# ─── Reference magnetron database ─────────────────────────────────────────────
REFERENCE_MAGNETRONS = [
    {"id": "4j50",    "model": "4J50 (Litton)",         "type": "s_pls", "f": 9.375, "P": 250,  "duty": 0.001, "Va": 16.0, "Ia": 10,  "Bz": 630, "Nv": 16, "eta": 42, "Zdc": 1.60, "QL": 250,  "cath": "disp", "app": "X-band radar; Carter Tables 15.2–15.3 reference design"},
    {"id": "4j52",    "model": "4J52",                   "type": "s_pls", "f": 9.375, "P": 180,  "duty": 0.001, "Va": 16.0, "Ia": 27,  "Bz": 490, "Nv": 16, "eta": 42, "Zdc": 0.59, "QL": 250,  "cath": "disp", "app": "X-band radar; McDowell (1998) simulation reference"},
    {"id": "4j78",    "model": "4J78",                   "type": "s_pls", "f": 9.375, "P": 300,  "duty": 0.001, "Va": 18.0, "Ia": 28,  "Bz": 660, "Nv": 16, "eta": 42, "Zdc": 0.64, "QL": 250,  "cath": "disp", "app": "X-band radar"},
    {"id": "mg5193",  "model": "MG5193 (Teledyne e2v)",  "type": "s_pls", "f": 2.998, "P": 2600, "duty": 0.001, "Va": 45.0, "Ia": 110, "Bz": 155, "Nv": 16, "eta": 40, "Zdc": 0.41, "QL": 250,  "cath": "disp", "app": "S-band medical linac; Teledyne e2v datasheet A1A-MG5193 v10 (2023)"},
    {"id": "mg7095",  "model": "MG7095 (Teledyne e2v)",  "type": "s_pls", "f": 2.998, "P": 3100, "duty": 0.001, "Va": 48.0, "Ia": 130, "Bz": 158, "Nv": 16, "eta": 40, "Zdc": 0.37, "QL": 250,  "cath": "disp", "app": "S-band high-power linac; e2v ARMMS RF/Microwave Society (2017)"},
    {"id": "mg6370",  "model": "MG6370 (e2v)",           "type": "s_pls", "f": 2.856, "P": 5500, "duty": 0.001, "Va": 55.0, "Ia": 200, "Bz": 180, "Nv": 16, "eta": 36, "Zdc": 0.275,"QL": 250,  "cath": "disp", "app": "S-band high-energy linac"},
    {"id": "vmc3105", "model": "VMC3105 (CPI)",           "type": "coa",   "f": 5.60,  "P": 2500, "duty": 0.001, "Va": 45.0, "Ia": 125, "Bz": 280, "Nv": 20, "eta": 44, "Zdc": 0.36, "QL": 1000, "cath": "disp", "app": "C-band coaxial — cargo/linac; CPI VMC3105 datasheet"},
    {"id": "vmc3109", "model": "VMC3109 (CPI)",           "type": "coa",   "f": 5.65,  "P": 2500, "duty": 0.001, "Va": 45.0, "Ia": 125, "Bz": 285, "Nv": 20, "eta": 44, "Zdc": 0.36, "QL": 1000, "cath": "disp", "app": "C-band coaxial linac/cargo ±10 MHz tunable; CPI datasheet"},
    {"id": "l3l6170", "model": "L3 L6170-02",             "type": "coa",   "f": 9.30,  "P": 2000, "duty": 0.001, "Va": 35.0, "Ia": 135, "Bz": 700, "Nv": 20, "eta": 42, "Zdc": 0.26, "QL": 1000, "cath": "disp", "app": "X-band compact linac; Faillace et al. (2021)"},
    {"id": "sfd349",  "model": "SFD349 (CPI)",            "type": "coa",   "f": 9.05,  "P": 200,  "duty": 0.002, "Va": 15.0, "Ia": 35,  "Bz": 620, "Nv": 20, "eta": 38, "Zdc": 0.43, "QL": 1000, "cath": "disp", "app": "X-band radar transmitter; CPI datasheet"},
    {"id": "cw_09",   "model": "Industrial CW 0.9 GHz",   "type": "s_cw",  "f": 0.915, "P": 50,   "duty": 1.0,   "Va": 15.0, "Ia": 4.0, "Bz": 62,  "Nv": 12, "eta": 85, "Zdc": 3.75, "QL": 250,  "cath": "thw",  "app": "0.9 GHz industrial ISM heating; Carter Table 15.4"},
    {"id": "cw_245",  "model": "Industrial CW 2.45 GHz",  "type": "s_cw",  "f": 2.450, "P": 6,    "duty": 1.0,   "Va": 4.5,  "Ia": 3.5, "Bz": 160, "Nv": 10, "eta": 68, "Zdc": 1.29, "QL": 250,  "cath": "thw",  "app": "Domestic microwave oven 0.7–6 kW (typical)"},
    {"id": "coa_l13", "model": "L-band coaxial ~1.3 GHz", "type": "coa",   "f": 1.30,  "P": 50,   "duty": 0.001, "Va": 15.0, "Ia": 8.5, "Bz": 88,  "Nv": 12, "eta": 39, "Zdc": 1.76, "QL": 1000, "cath": "disp", "app": "1.3 GHz linac (estimated from Carter Table 15.4 + scaling)"},
    {"id": "rl50",    "model": "RL50 (L-band search)",     "type": "s_pls", "f": 1.30,  "P": 1000, "duty": 0.001, "Va": 20.0, "Ia": 60,  "Bz": 90,  "Nv": 14, "eta": 42, "Zdc": 0.33, "QL": 250,  "cath": "disp", "app": "L-band search radar (Collins §typical designs)"},
    {"id": "qk571",   "model": "QK571 (X-band marine)",    "type": "s_pls", "f": 9.41,  "P": 200,  "duty": 0.001, "Va": 16.0, "Ia": 32,  "Bz": 610, "Nv": 16, "eta": 39, "Zdc": 0.50, "QL": 250,  "cath": "disp", "app": "X-band marine radar"},
    {"id": "cw500k",  "model": "CW 500 kW (Shibata 1991)", "type": "s_cw",  "f": 0.915, "P": 500,  "duty": 1.0,   "Va": 44.0, "Ia": 16,  "Bz": 121, "Nv": 14, "eta": 80, "Zdc": 2.75, "QL": 250,  "cath": "thw",  "app": "500 kW CW industrial linac; Shibata et al. (1991) design case Carter §15.7.3"},
]

# ─── Quick presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "linac13": {
        "desc":  "1.3 GHz L-band linac injector, 50 kW peak",
        "f":     1.3,   "P":    50,    "type": "coa",  "cath": "disp",
        "eta":   None,  "Zdc":  None,  "etaC": 88,    "Rp":   2.0,  "fill": 0.45,
    },
    "linac_s": {
        "desc":  "S-band medical linac (MG5193-class), 2.6 MW peak",
        "f":     2.998, "P":    2600,  "type": "s_pls","cath": "disp",
        "eta":   40,    "Zdc":  0.41,  "etaC": 90,    "Rp":   2.0,  "fill": 0.45,
    },
    "radar_x": {
        "desc":  "X-band pulsed radar (4J50-class), 250 kW peak",
        "f":     9.375, "P":    250,   "type": "s_pls","cath": "disp",
        "eta":   None,  "Zdc":  None,  "etaC": 90,    "Rp":   2.0,  "fill": 0.45,
    },
    "oven": {
        "desc":  "2.45 GHz CW domestic/industrial, 6 kW",
        "f":     2.45,  "P":    6,     "type": "s_cw", "cath": "thw",
        "eta":   None,  "Zdc":  None,  "etaC": 90,    "Rp":   2.0,  "fill": 0.45,
    },
    "ka_rs": {
        "desc":  "Ka-band rising sun radar, 80 kW peak",
        "f":     30.0,  "P":    80,    "type": "rs",   "cath": "disp",
        "eta":   None,  "Zdc":  None,  "etaC": 85,    "Rp":   2.1,  "fill": 0.42,
    },
}

# ─── Core design calculations ──────────────────────────────────────────────────

def compute_dc_point(f_ghz, P_kw, eta_pct, Zdc_kohm, etaC_pct):
    """
    Returns DC operating point or None if parameters are unphysical.

    Parameters
    ----------
    f_ghz    : frequency (GHz)
    P_kw     : output power (kW) — peak or CW
    eta_pct  : overall efficiency (%)
    Zdc_kohm : DC impedance Va/Ia (kΩ)
    etaC_pct : circuit efficiency (%)
    """
    eta   = eta_pct / 100.0
    etaC  = etaC_pct / 100.0
    Zdc   = Zdc_kohm * 1e3
    etaE  = eta / etaC

    if not (0.05 < etaE < 0.993):
        return None

    Pdc = P_kw * 1e3 / eta
    Va  = math.sqrt(Pdc * Zdc)
    Ia  = Pdc / Va

    if not (100 < Va < 1.5e6):
        return None

    # Carter eq. 15.49: etaE ≈ (2r−3)/(2r−1)  → r = (3−etaE)/(2(1−etaE))
    r = (3 - etaE) / (2 * (1 - etaE))
    if not (1.3 < r < 35):
        return None

    V0  = Va / (2 * r - 1)         # characteristic voltage (V)  eq.15.22
    VH  = r * r * V0               # Hull cut-off voltage (V)
    lam = C / (f_ghz * 1e9)        # free-space wavelength (m)

    # Skin depth and surface resistance in Cu
    delta_s = math.sqrt(RCU / (math.pi * f_ghz * 1e9 * MU0))  # m
    Rs      = math.sqrt(math.pi * f_ghz * 1e9 * MU0 * RCU)    # Ω/□

    return {
        "f_ghz": f_ghz, "P_kw": P_kw,
        "eta":   eta,   "etaE": etaE, "etaC": etaC,
        "Zdc":   Zdc,
        "Pdc":   Pdc,   "Va":  Va,   "Ia":  Ia,
        "r":     r,     "V0":  V0,   "VH":  VH,
        "VaVH":  Va / VH,
        "lam":   lam,
        "delta_s_um": delta_s * 1e6,
        "Rs_mOhm": Rs * 1e3,
    }


def sweep_vanes(dc, type_id, cath_id, etaC_pct, Rp, fill, duty_cycle=1.0, la_ratio=None):
    """
    Sweep N_v from 6 to 34 (even only) and compute geometry + checks.
    Returns list of row dicts; rec = row with highest score that has no fatal issues.
    """
    t       = TYPES[type_id]
    cath    = CATHODES[cath_id]
    f       = dc["f_ghz"]
    Va, Ia  = dc["Va"], dc["Ia"]
    r       = dc["r"]
    V0, VH  = dc["V0"], dc["VH"]
    lam     = dc["lam"]
    is_cw   = (t["duty"] == "cw")
    duty_cycle = float(duty_cycle) if duty_cycle is not None else (1.0 if is_cw else 0.001)
    duty_cycle = max(0.0, min(1.0, duty_cycle))
    Jlim    = cath["Jcw"] if is_cw else cath["Jpulse"]

    if la_ratio is None or not math.isfinite(float(la_ratio)) or float(la_ratio) <= 0:
        la_ratio = 1.5 if type_id == "la" else 0.16
    la_ratio = float(la_ratio)
    La      = la_ratio * lam  # anode axial length (m)
    QL      = t["QL"]
    Rs      = dc["Rs_mOhm"] * 1e-3  # Ω/□

    rows = []
    for Nv in range(6, 36, 2):
        n   = Nv // 2
        ws  = 2 * math.pi * f * 1e9 / n    # synchronous angular velocity (rad/s)

        # Anode radius — Carter eq. 15.22
        ra = math.sqrt(2 * EOM * V0) / ws  # m

        # Cathode radius — Collins/Clogston / Carter eq. 15.154 (closed form)
        # xi = rc/ra = (Nv*(r−1) − R'*r) / (Nv*(r−1) + R'*r)
        xnum = Nv * (r - 1) - Rp * r
        xden = Nv * (r - 1) + Rp * r
        if xden <= 0:
            rows.append({"Nv": Nv, "ok": False, "msg": "Denominator ≤ 0 (increase Nv or reduce R')"})
            continue
        xi = xnum / xden
        if not (0.12 <= xi <= 0.93):
            msg = ("R' too large — reduce R' or increase Nv" if xi <= 0 else
                   "r_c/r_a > 0.93 — gap too narrow" if xi > 0.93 else
                   "Geometry infeasible")
            rows.append({"Nv": Nv, "ok": False, "msg": msg})
            continue
        rc = xi * ra  # m

        # Characteristic and operating magnetic fields — Carter eq. 15.23
        B0 = 2 * ws / (EOM * (1 - xi**2))   # T
        Bz = r * B0                           # T

        # Cathode current density
        # McDowell (1998): fraction α ≈ 0.75 returns to cathode → Ic = Ia / (1−α) = Ia / 0.25
        Ic = Ia / 0.25
        Jc = Ic / (2 * math.pi * rc * La * 1e4)  # A/cm²

        # Cavity Q from first principles (Kroll in Collins §3)
        # Cavity depth ≈ 0.30λ (Kroll: λ = 4d overestimates freq by ~20%)
        d_cav  = 0.30 * lam
        w_gap  = fill * 2 * math.pi * ra / Nv
        Vc     = d_cav * w_gap * La                               # cavity volume (m³)
        Ac     = 2 * (d_cav * w_gap + d_cav * La + w_gap * La)   # surface area (m²)
        QU     = 2 * math.pi * f * 1e9 * MU0 * Vc / (Rs * Ac)
        if type_id == "coa":
            QU *= 5.5   # external cavity energy storage multiplier

        # Vane thickness (tangential) and vane-tip temperature (CW thermal limit)
        pitch_m = 2 * math.pi * ra / Nv
        t_vane_m = (1 - fill) * pitch_m

        # Vane-tip temperature (thermal conduction model) using duty-cycle-corrected
        # average anode dissipation. Carter Eq. 15.155 assumes Pa = Pdc - Prf.
        # For pulsed operation, use average dissipation Pa_avg ≈ Pa_peak * duty.
        Pa_peak_W = max(0.0, dc["Pdc"] - dc["P_kw"] * 1e3)
        Pa_avg_W = Pa_peak_W * duty_cycle
        Tv_tip_K = _vane_tip_temperature_K(Pa_avg_W, Nv, t_vane_m, La)

        # Anode power density (CW only) — Collins Ch. 8 limit ≤ 25 W/cm²
        # Use an effective copper area based on total cavity surface area
        # (Nv cavities), not just the smooth cylindrical area 2πr_a L_a.
        # The cylindrical area alone substantially overestimates Pd.
        Pd_cw = 0.0
        if is_cw:
            Pa_W = max(0.0, dc["Pdc"] - dc["P_kw"] * 1e3)
            A_cu_cm2 = (Nv * Ac) * 1e4
            Pd_cw = (Pa_W / A_cu_cm2) if A_cu_cm2 > 0 else 0.0  # W/cm²

        # π−1 mode threshold — Carter eq. 15.40 for n → n−1
        nm1  = max(n - 1, 1)
        ws1  = 2 * math.pi * f * 1e9 / nm1
        B0n1 = 2 * ws1 / (EOM * (1 - xi**2))
        rn1  = Bz / B0n1
        V0n1 = ra**2 * ws1**2 / (2 * EOM)
        VTn1 = V0n1 * (2 * rn1 - 1)
        msep = (VTn1 - Va) / Va * 100  # mode separation (%)

        # Collins reduced (dimensionless) parameters per Eqs. (25)–(26).
        # Keep ωc/ωs as a separate internal variable for existing heuristic checks.
        wc_over_ws = EOM * Bz / ws  # ωc/ωs

        # Characteristic (scale) factors (Collins Eq. 26):
        #   \bar{B} = B0,  \bar{V} = V0
        #   \bar{I} depends on a1(r_a/r_c) (Fig. 10-9)
        ra_over_rc = ra / rc
        a1 = _a1_ra_over_rc(ra_over_rc)
        Bbar = B0
        Vbar = V0
        Ibar = (
            (2 * math.pi * a1)
            / ((1 - (rc / ra) ** 2) ** 2 * (ra_over_rc + 1))
            * (1 / EOM)  # m/e
            * (ws**3)
            * (ra**2)
            * EPS0
            * La
        )
        Gbar = Ibar / Vbar
        Pbar = Ibar * Vbar

        # Reduced variables (Collins Eq. 25)
        b = Bz / Bbar
        v = Va / Vbar
        ii = Ia / Ibar

        # Estimate load (slot) conductance G_L from an equivalent parallel-RLC model.
        # This is an approximation; Collins defines g = G_L / \bar{G}.
        G_L = None
        if ra > rc and QL and QU and QU > 0:
            try:
                C_eq = 2 * math.pi * EPS0 * La / math.log(ra / rc)  # F
                if C_eq > 0 and math.isfinite(C_eq):
                    # External Q from 1/QL = 1/QU + 1/Qext (when QL < QU)
                    if QL > 0 and QU > QL:
                        Qext = 1.0 / (1.0 / QL - 1.0 / QU)
                        if Qext > 0 and math.isfinite(Qext):
                            G_L = ws * C_eq / Qext
                    # Fallback: use total loaded Q if Qext is not meaningful
                    if G_L is None and QL > 0:
                        G_L = ws * C_eq / QL
            except (ValueError, ZeroDivisionError, OverflowError):
                G_L = None

        g = (G_L / Gbar) if (G_L is not None and Gbar > 0 and math.isfinite(Gbar)) else float("nan")

        P_out = dc["P_kw"] * 1e3
        p = (P_out / Pbar) if (Pbar > 0 and math.isfinite(Pbar)) else float("nan")

        # Issue flags
        issues = []
        if msep < 0:
            issues.append(("fatal", "π−1 mode below Va"))
        elif msep < 8:
            issues.append(("warn",  f"Low mode sep {msep:.1f}%"))
        if Jc > Jlim * 1.05:
            issues.append(("fatal", f"Jc {Jc:.2f} > limit {Jlim} A/cm²"))
        elif Jc > Jlim * 0.78:
            issues.append(("warn",  f"Jc {Jc:.2f} near limit {Jlim} A/cm²"))
        if wc_over_ws < 2.8:
            issues.append(("warn",  f"ωc/ωs = {wc_over_ws:.2f} < 3 (reduced η)"))
        if wc_over_ws > 15:
            issues.append(("warn",  f"ωc/ωs = {wc_over_ws:.2f} > 15 (excessive B)"))
        if dc["VaVH"] > 0.92:
            issues.append(("warn",  "Va/VH > 0.92 — near cut-off"))
        if dc["VaVH"] < 0.18:
            issues.append(("warn",  "Va/VH < 0.18 — very low (outside typical range)"))
        # NOTE: Pd_cw is retained for CLI/internal analysis, but is intentionally
        # not surfaced as a UI warning/fatal constraint.

        if math.isfinite(Tv_tip_K):
            if Tv_tip_K > T_VANE_FATAL_K:
                issues.append(("fatal", f"Vane tip T {Tv_tip_K - 273.15:.0f}°C > 700°C"))
            elif Tv_tip_K > T_VANE_WARN_K:
                issues.append(("warn",  f"Vane tip T {Tv_tip_K - 273.15:.0f}°C > 500°C"))
        if type_id == "rs" and Nv % 4 == 0:
            issues.append(("warn",  "Nv divisible by 4 — avoid"))
        if type_id == "rs" and abs(wc_over_ws - 1) < 0.28:
            issues.append(("fatal", "ωc/ω ≈ 1 — cyclotron resonance!"))

        fatal = any(s == "fatal" for s, _ in issues)
        temp_penalty = 0.0
        if math.isfinite(Tv_tip_K) and Tv_tip_K > T_VANE_WARN_K:
            temp_penalty = (Tv_tip_K - T_VANE_WARN_K) * 0.35
        score = (
            (min(msep, 40) if msep > 0 else -200)
            - max(0, (Jc / Jlim - 0.5)) * 25
            - abs(wc_over_ws - 4.5) * 2.5
            - abs(dc["VaVH"] - 0.72) * 55
            - temp_penalty
            - (180 if fatal else 0)
        )

        rows.append({
            "Nv": Nv, "ok": True, "fatal": fatal,
            "ra_mm": ra * 1e3, "rc_mm": rc * 1e3, "xi": xi,
            "B0_mT": B0 * 1e3, "Bz_mT": Bz * 1e3,
            "pitch_mm": 2 * math.pi * ra / Nv * 1e3,
            "La_mm": La * 1e3,
            "w_gap_mm": w_gap * 1e3,
            "d_cav_mm": d_cav * 1e3,
            "t_vane_mm": t_vane_m * 1e3,
            "Jc": Jc, "Jlim": Jlim,
            "Pd_cw": Pd_cw,
            "Tv_tip_C": (Tv_tip_K - 273.15) if math.isfinite(Tv_tip_K) else float("nan"),
            "VTn1_kV": VTn1 / 1e3,
            "msep": msep,
            "QU": QU,
            "b": b, "v": v, "ii": ii, "g": g, "p": p,
            "issues": issues,
            "score": score,
        })

    good = [rr for rr in rows if rr.get("ok") and not rr.get("fatal")]
    rec  = max(good, key=lambda x: x["score"]) if good else None
    return rows, rec


# ─── Type suitability scoring ──────────────────────────────────────────────────

def type_suitability(type_id, f_ghz, P_kw, is_cw):
    """Score 0–100 for how appropriate this type is for the operating point."""
    base = TYPES[type_id]["linac_suitability"]
    bonus = 0
    if P_kw < 200  and type_id == "coa": bonus += 15
    if P_kw > 1000 and type_id == "la":  bonus += 10
    if f_ghz > 20  and type_id == "rs":  bonus += 25
    if f_ghz < 2   and type_id == "la":  bonus += 10
    if is_cw       and type_id == "s_cw":bonus += 20
    return min(100, base + bonus)


def justify_type(type_id, f_ghz, P_kw, is_cw, QL):
    """Return a multi-line justification string for the chosen type."""
    scores = {tid: type_suitability(tid, f_ghz, P_kw, is_cw) for tid in TYPES}
    best   = max(scores, key=scores.get)
    lines  = []
    for tid, sc in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * (sc // 5) + "░" * (20 - sc // 5)
        col = green if sc >= 70 else (yellow if sc >= 45 else red)
        marker = "  ← selected" if tid == type_id else ("  ← recommended" if tid == best and tid != type_id else "")
        lines.append(f"  {TYPES[tid]['label']:<20} {col(bar)} {sc:3d}%{marker}")

    pull_mhz = 0.417 * (f_ghz * 1e3) / QL  # MHz
    justification = []
    if best == "coa":
        justification = [
            f"  The coaxial type is recommended because Q_L ≈ 1000 reduces",
            f"  frequency pulling to {pull_mhz:.2f} MHz (S=1.5), vs {0.417*(f_ghz*1e3)/250:.2f} MHz for strapped.",
            f"  Frequency pushing is ~0.08 MHz/A vs ~0.4 MHz/A (Carter Table 15.1).",
            f"  The 20% tuning range tracks linac cavity thermal drift.",
            f"  Non-contacting TE₀₁₁ plunger eliminates contact wear.",
            f"  Injection-locking at ~10 dB below output is feasible (Adler 1946;",
            f"  Tahir et al. IEEE Trans. Electron Dev. 52, 2096 (2005)).",
            f"  A circulator between the magnetron and linac cavity is mandatory.",
        ]
    return lines, justification, best


# ─── Frequency stability summary ──────────────────────────────────────────────

def freq_stability_table(f_ghz, type_ids=None):
    """Return a comparison table of frequency stability for all types."""
    if type_ids is None:
        type_ids = list(TYPES.keys())
    rows = []
    for tid in type_ids:
        QL    = TYPES[tid]["QL"]
        pull  = 0.417 * (f_ghz * 1e3) / QL   # MHz
        push  = 0.6   * (f_ghz * 1e3) / QL   # MHz (electron loading)
        rows.append((TYPES[tid]["label"], QL, pull, push))
    return rows


# ─── Formatted output functions ───────────────────────────────────────────────

def print_header(title):
    print()
    print(hr("═"))
    print(bold(f"  {title}"))
    print(hr("═"))

def print_section(title):
    print()
    print(hr("─"))
    print(bold(cyan(f"  {title}")))
    print(hr("─"))

def field(label, value, unit="", width=38, color_fn=None):
    v = f"{value}" if color_fn is None else color_fn(f"{value}")
    u = dim(f" {unit}") if unit else ""
    print(f"  {label:<{width}} {v}{u}")

def nd(v, dp=2):
    """Format a float to dp decimal places, or '—' if None/non-finite."""
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{dp}f}"

def print_dc_results(dc, t_id):
    t = TYPES[t_id]
    print_section("DC Operating Point   [Carter §15.2; Collins & Clogston §16]")
    VaVH_c = green if 0.5 <= dc["VaVH"] <= 0.9 else yellow
    field("Anode voltage Vₐ",          nd(dc["Va"]/1e3, 2),      "kV")
    field("Anode current Iₐ",          nd(dc["Ia"], 2),           "A")
    field("DC input power Pdc",         nd(dc["Pdc"]/1e3, 2),     "kW")
    field("Overall efficiency η",       f"{dc['eta']*100:.1f}",   "%")
    field("Electronic eff. ηₑ = η/ηc", f"{dc['etaE']*100:.1f}",  "%")
    field("Bz/B₀  r = (3−ηₑ)/(2(1−ηₑ))",nd(dc["r"],3),         "[Carter eq.15.49]")
    field("Characteristic voltage V₀",  nd(dc["V0"]/1e3, 3),     "kV")
    field("Hull cut-off voltage VH",    nd(dc["VH"]/1e3, 2),     "kV")
    field("Vₐ/VH  [Collins: 0.50–0.90]",
          VaVH_c(nd(dc["VaVH"], 3)),    "")
    field("Free-space wavelength λ",    nd(dc["lam"]*1e3, 2),    "mm")
    field("Cu skin depth δₛ",           nd(dc["delta_s_um"], 4), "μm")
    field("Surface resistance Rₛ",      nd(dc["Rs_mOhm"], 4),    "mΩ/□")

    print()
    print(bold("  Frequency stability"))
    QL   = TYPES[t_id]["QL"]
    pull = 0.417 * (dc["f_ghz"] * 1e3) / QL
    dF   = 0.6   * (dc["f_ghz"] * 1e3) / QL
    field("Loaded Q_L",                  QL,                        "")
    field("Pulling figure (S=1.5)",       nd(pull, 2),              "MHz  [Carter eq.15.93]")
    field("Electron frequency loading Δf",nd(dF, 2),               "MHz  [Carter eq.15.88]")
    field("Pushing figure",              "~100",                    "kHz/A (typical)")


def col_flag(issues, is_cw=False):
    """Return a coloured status string from an issues list."""
    if not issues:
        return green("OK")
    worst = "fatal" if any(s == "fatal" for s, _ in issues) else "warn"
    msgs  = "; ".join(m for _, m in issues[:2])
    return (red if worst == "fatal" else yellow)(msgs[:45])


def print_vane_table(rows, rec, is_cw):
    print_section(f"Vane-count sweep  Nv = 6…34  [★ = recommended]")

    # Header
    cw_col = f"  {'Pd':>7}" if is_cw else ""
    print(bold(
        f"  {'Nv':>3}  {'ra':>7}  {'rc':>7}  {'rc/ra':>6}  "
        f"{'B0':>7}  {'Bz':>7}  {'Pitch':>6}  {'La':>6}"
        f"  {'Jc/lim':>10}{cw_col}"
        f"  {'VTπ-1':>7}  {'Sep%':>5}  {'QU':>5}  {'b':>4}  Status"
    ))
    print(dim(
        f"  {'':>3}  {'mm':>7}  {'mm':>7}  {'':>6}  "
        f"{'mT':>7}  {'mT':>7}  {'mm':>6}  {'mm':>6}"
        f"  {'A/cm²':>10}{cw_col}"
        f"  {'kV':>7}  {'':>5}  {'':>5}  {'':>4}"
    ))
    print(hr())

    for rr in rows:
        Nv  = rr["Nv"]
        star = "★" if (rec and rr["ok"] and rec["Nv"] == Nv) else " "

        if not rr.get("ok"):
            pfx = bold(f"  {Nv:>3}{star}") if star.strip() else f"  {Nv:>3}{star}"
            print(f"{pfx}  {dim(rr['msg'])}")
            continue

        jc_str  = f"{rr['Jc']:5.2f}/{rr['Jlim']}"
        jc_col  = (green if rr["Jc"] <= rr["Jlim"] * 0.78 else
                   (yellow if rr["Jc"] <= rr["Jlim"] * 1.05 else red))
        ms_col  = (green if rr["msep"] >= 15 else
                   (yellow if rr["msep"] >= 8 else red))
        b_col   = (green if 3 <= rr["b"] <= 6 else
                   (yellow if 2.5 <= rr["b"] <= 7 else red))
        pd_str  = f"  {rr['Pd_cw']:7.1f}" if is_cw else ""

        status  = col_flag(rr["issues"])
        if star.strip():
            status = bold(green("★ best"))
            if rr["issues"]:
                status = bold(yellow("★ " + "; ".join(m for _, m in rr["issues"][:1])))

        line = (
            f"  {Nv:>3}{star}"
            f"  {rr['ra_mm']:7.2f}"
            f"  {rr['rc_mm']:7.2f}"
            f"  {rr['xi']:6.3f}"
            f"  {rr['B0_mT']:7.1f}"
            f"  {rr['Bz_mT']:7.1f}"
            f"  {rr['pitch_mm']:6.2f}"
            f"  {rr['La_mm']:6.1f}"
            f"  {jc_col(jc_str):>10}"
            f"{pd_str}"
            f"  {rr['VTn1_kV']:7.2f}"
            f"  {ms_col('{:5.1f}'.format(rr['msep']))}"
            f"  {int(rr['QU']):5d}"
            f"  {b_col('{:4.2f}'.format(rr['b']))}"
            f"  {status}"
        )
        if star.strip():
            print(bold(line))
        else:
            print(line)

    print()
    print(dim("  ra,rc: Carter eq.15.22/15.154  ·  B0,Bz: eq.15.23  ·  Jc limits: Liu et al. MetalMat (2024)"))
    print(dim("  Mode sep.: eq.15.40  ·  QU: Kroll in Collins §3  ·  CW Pd limit: Collins Ch.8 (≤25 W/cm²)"))


def print_recommended(rec, dc, fill, type_id):
    if rec is None:
        print_section("Recommended Design")
        print(red("  ✗ No valid configuration found — adjust Nv range, R', or operating point."))
        return

    t  = TYPES[type_id]
    QL = t["QL"]
    QU = rec["QU"]
    beta   = QL / max(QU, 1)
    etaC_q = QU / (QU + QL)
    V1onVg = (math.sin(math.pi * fill / 2) / (math.pi * fill / 2)) / math.pi  # Carter eq.15.20 factor
    pull   = 0.417 * (dc["f_ghz"] * 1e3) / QL

    print_section(f"Recommended Design  [Nv = {rec['Nv']}]   [Carter §15.7; Collins §16]")

    print(bold("  Geometry"))
    field("  Anode radius rₐ",           nd(rec["ra_mm"], 3),   "mm")
    field("  Cathode radius r꜀",          nd(rec["rc_mm"], 3),   "mm")
    field("  Interaction gap rₐ − r꜀",   nd(rec["ra_mm"] - rec["rc_mm"], 3), "mm")
    field("  Vane pitch at rₐ",          nd(rec["pitch_mm"], 3), "mm")
    field("  Gap width w (fill × pitch)", nd(rec["w_gap_mm"], 3), "mm")
    field("  Vane thickness t ≈ (1−w/p)×p", nd(rec["t_vane_mm"], 3), "mm")
    field("  Cavity depth ≈ 0.30λ  [Kroll]",nd(rec["d_cav_mm"], 2), "mm")
    field("  Anode length Lₐ",           nd(rec["La_mm"], 2),   "mm")

    print()
    print(bold("  Magnetic field"))
    B0_col = lambda x: x
    Bz_col = lambda x: x
    field("  Characteristic field B₀",   nd(rec["B0_mT"], 2),  "mT  [Carter eq.15.23]")
    field("  Operating field Bz",         nd(rec["Bz_mT"], 2),  "mT")

    print()
    print(bold("  Cathode emission check"))
    jc_col = green if rec["Jc"] <= rec["Jlim"] * 0.78 else (yellow if rec["Jc"] <= rec["Jlim"] * 1.05 else red)
    field("  Jc actual / limit",
          jc_col(f"{nd(rec['Jc'], 2)} / {rec['Jlim']}"), "A/cm²")
    if rec["Pd_cw"] > 0:
        pd_col = green if rec["Pd_cw"] <= 25 else (yellow if rec["Pd_cw"] <= 50 else red)
        field("  Anode power density Pd",  pd_col(nd(rec["Pd_cw"], 1)), "W/cm²  [Collins Ch.8 limit: 25]")

    print()
    print(bold("  Mode separation"))
    ms_col = green if rec["msep"] >= 15 else (yellow if rec["msep"] >= 8 else red)
    field("  π − (π−1) mode separation", ms_col(nd(rec["msep"], 1)), "%")
    field("  V_{T,π−1}",                 nd(rec["VTn1_kV"], 2),       "kV")

    print()
    print(bold("  Circuit coupling  [Kroll in Collins §3]"))
    field("  Estimated unloaded QU",      f"{int(QU)}",             "")
    field("  External QL",                QL,                       "(from type)")
    match_str = ("≈ matched" if 0.5 <= beta <= 2 else
                 "under-coupled" if beta < 0.5 else "over-coupled")
    field("  Coupling β = QL/QU",         nd(beta, 3),              f"  ({match_str})")
    field("  Circuit eff. QU/(QU+QL)",    f"{etaC_q*100:.1f}",     "%  (estimated)")
    field("  Wave/gap volt. V₁/Vg",       nd(V1onVg, 4),            "[fill factor, Carter eq.15.20]")
    field("  Frequency pulling (S=1.5)",  nd(pull, 2),              "MHz  [eq.15.93]")

    print()
    print(bold("  Collins/Clogston reduced parameters"))
    field("  b = B/B̄",                    nd(rec["b"], 3),     "")
    field("  v = V/V̄",                    nd(rec["v"], 4),     "")
    field("  i = I/Ī",                     nd(rec["ii"], 4),    "")
    field("  g = G_L/Ḡ",                  nd(rec.get("g"), 4), "")
    field("  p = P/P̄",                    nd(rec.get("p"), 4), "")


def print_type_comparison(f_ghz, P_kw, is_cw, selected_id):
    print_section("Magnetron Type Comparison")

    score_lines, justification, best = justify_type(selected_id, f_ghz, P_kw, is_cw, TYPES[selected_id]["QL"])
    print(bold("  Suitability scores for this operating point:"))
    for line in score_lines:
        print(line)

    if justification:
        print()
        print(bold("  Design justification:"))
        for line in justification:
            print(green(line) if selected_id == best else yellow(line))

    print()
    print(bold("  Frequency stability comparison:"))
    rows = freq_stability_table(f_ghz)
    print(f"  {'Type':<22} {'QL':>6}  {'Pulling (MHz)':>14}  {'Loading Δf (MHz)':>17}")
    print(dim(f"  {'':<22} {'':>6}  {'[Carter 15.93]':>14}  {'[Carter 15.88]':>17}"))
    print(hr())
    for label, QL, pull, dF in rows:
        sel = " ← selected" if TYPES[[k for k, v in TYPES.items() if v["label"] == label][0]]["label"] == TYPES[selected_id]["label"] else ""
        print(f"  {label:<22} {QL:>6}  {pull:>14.2f}  {dF:>17.2f}{sel}")


def print_cathode_info(cath_id, type_id):
    cath = CATHODES[cath_id]
    t    = TYPES[type_id]
    is_cw = t["duty"] == "cw"
    Jlim = cath["Jcw"] if is_cw else cath["Jpulse"]
    print_section(f"Cathode: {cath['label']}")
    field("Max Jc (" + ("CW" if is_cw else "pulsed") + ")", Jlim, "A/cm²")
    field("Max operating temperature", cath["Tmax"], "°C")
    print()
    # Word-wrap note
    note = cath["note"]
    words = note.split()
    line, max_w = "  ", 85
    for w in words:
        if len(line) + len(w) + 1 > max_w:
            print(dim(line))
            line = "  " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(dim(line))


def print_reference_db():
    print_section("Reference Magnetron Database")
    print(bold(f"  {'ID':<10} {'Model':<28} {'Type':<16} {'f(GHz)':>7} {'P(kW)':>8} "
               f"{'Va(kV)':>7} {'Ia(A)':>6} {'Bz(mT)':>7} {'Nv':>3} {'η%':>4} {'Zdc':>5} {'QL':>5}"))
    print(hr())
    for m in REFERENCE_MAGNETRONS:
        type_lbl = TYPES[m["type"]]["label"][:14]
        print(
            f"  {m['id']:<10} {m['model']:<28} {type_lbl:<16}"
            f"  {m['f']:>6.3f} {m['P']:>8.0f} {m['Va']:>7.1f} {m['Ia']:>6.0f}"
            f" {m['Bz']:>7.0f} {m['Nv']:>3} {m['eta']:>4} {m['Zdc']:>5.2f} {m['QL']:>5}"
        )
    print()
    print(dim("  Sources: Collins MIT Rad Lab Vol.6 (1948); Carter (2018) Tables 15.2–15.4;"))
    print(dim("  Teledyne e2v MG5193 v10 (2023), MG7095 ARMMS (2017); CPI VMC3105/VMC3109;"))
    print(dim("  McDowell IEEE Trans. Plasma Sci. 26 (1998); Shibata et al. (1991); Faillace et al. (2021)"))


def _issues_to_list(issues):
    return [{"level": level, "message": msg} for level, msg in issues]


def _resolve_design_inputs(raw_params):
    """Resolve inputs from API params with optional preset/database loading."""
    params = dict(raw_params or {})

    def _to_float(name, default=None):
        val = params.get(name, default)
        if val is None or val == "":
            return None
        return float(val)

    preset_id = params.get("preset")
    db_id = params.get("load_db")

    f_ghz = None
    P_kw = None
    t_id = None
    cath = "disp"
    duty_cycle = None
    eta = None
    Zdc = None
    etaC = 90.0
    Rp = 2.0
    fill = 0.45

    if preset_id:
        if preset_id not in PRESETS:
            raise ValueError(f"Invalid preset '{preset_id}'. Valid: {', '.join(PRESETS.keys())}")
        ps = PRESETS[preset_id]
        f_ghz = ps["f"]
        P_kw = ps["P"]
        t_id = ps["type"]
        cath = ps.get("cath", cath)
        eta = ps.get("eta")
        Zdc = ps.get("Zdc")
        etaC = ps.get("etaC", etaC)
        Rp = ps.get("Rp", Rp)
        fill = ps.get("fill", fill)

    if db_id:
        db_id = str(db_id).lower()
        match = next((m for m in REFERENCE_MAGNETRONS if m["id"] == db_id), None)
        if match is None:
            raise ValueError(f"Database ID '{db_id}' not found. Use --list-db for valid IDs.")
        f_ghz = match["f"] if f_ghz is None else f_ghz
        P_kw = match["P"] if P_kw is None else P_kw
        t_id = match["type"] if t_id is None else t_id
        cath = match["cath"] if cath == "disp" else cath
        eta = match["eta"] if eta is None else eta
        Zdc = match["Zdc"] if Zdc is None else Zdc
        duty_cycle = match.get("duty") if duty_cycle is None else duty_cycle

    # Explicit params override preset/db values.
    f_exp = _to_float("freq")
    p_exp = _to_float("power")
    duty_exp = _to_float("duty")
    eta_exp = _to_float("eta")
    zdc_exp = _to_float("zdc")
    etac_exp = _to_float("etac")
    rp_exp = _to_float("rp")
    fill_exp = _to_float("fill")
    la_exp = _to_float("la_ratio")
    if la_exp is None:
        la_exp = _to_float("la")

    f_ghz = f_exp if f_exp is not None else f_ghz
    P_kw = p_exp if p_exp is not None else P_kw
    duty_cycle = duty_exp if duty_exp is not None else duty_cycle
    eta = eta_exp if eta_exp is not None else eta
    Zdc = zdc_exp if zdc_exp is not None else Zdc
    etaC = etac_exp if etac_exp is not None else etaC
    Rp = rp_exp if rp_exp is not None else Rp
    fill = fill_exp if fill_exp is not None else fill

    la_ratio = la_exp

    t_exp = params.get("type")
    if t_exp:
        t_id = t_exp
    c_exp = params.get("cath")
    if c_exp:
        cath = c_exp

    if f_ghz is None or P_kw is None or t_id is None:
        raise ValueError("Missing required inputs: freq, power, and type (or provide preset/load_db).")

    # Duty-cycle handling (UI can use a combined strapped type selector).
    # Map combined strapped type to the existing CW/pulsed internal keys.
    # Requirement: duty==1 is considered CW.
    if str(t_id).strip().lower() in {"s", "strapped"}:
        dc_for_map = 1.0 if duty_cycle is None else float(duty_cycle)
        t_id = "s_cw" if dc_for_map >= 1.0 else "s_pls"

    if t_id not in TYPES:
        valid = ", ".join(sorted(set(TYPES.keys()) | {"s"}))
        raise ValueError(f"Invalid type '{t_id}'. Valid: {valid}")

    # Normalized anode length La/λ.
    if la_ratio is None:
        la_ratio = 1.5 if t_id == "la" else 0.16
    if not math.isfinite(float(la_ratio)):
        raise ValueError("la_ratio must be a finite number")
    la_ratio = float(la_ratio)
    if not (0.01 <= la_ratio <= 10.0):
        raise ValueError("la_ratio must be in [0.01, 10]")
    # If duty was not provided, choose a reasonable default by type.
    # CW defaults to 1.0; pulsed defaults to 0.001 (typical radar/linac order).
    if duty_cycle is None:
        duty_cycle = 1.0 if TYPES[t_id]["duty"] == "cw" else 0.001
    if not (0 < duty_cycle <= 1.0):
        raise ValueError("duty must be in (0, 1]")

    if cath not in CATHODES:
        raise ValueError(f"Invalid cath '{cath}'. Valid: {', '.join(CATHODES.keys())}")
    if not (0.1 <= fill <= 0.9):
        raise ValueError("fill must be in [0.1, 0.9]")
    if etaC <= 0 or etaC >= 100:
        raise ValueError("etac must be between 0 and 100")

    return {
        "f_ghz": f_ghz,
        "P_kw": P_kw,
        "type_id": t_id,
        "cath_id": cath,
        "duty_cycle": duty_cycle,
        "eta_override": eta,
        "zdc_override": Zdc,
        "etac_pct": etaC,
        "rp": Rp,
        "fill": fill,
        "la_ratio": la_ratio,
        "preset": preset_id,
        "load_db": db_id,
    }


def compute_design_payload(inputs):
    """Compute magnetron design and return a JSON-serializable payload."""
    t_id = inputs["type_id"]
    cath_id = inputs["cath_id"]
    f_ghz = inputs["f_ghz"]
    P_kw = inputs["P_kw"]

    t = TYPES[t_id]
    eta_pct = inputs["eta_override"] if inputs["eta_override"] is not None else t["eta_fn"](f_ghz)
    zdc_k = inputs["zdc_override"] if inputs["zdc_override"] is not None else t["zdc_fn"](f_ghz)

    dc = compute_dc_point(f_ghz, P_kw, eta_pct, zdc_k, inputs["etac_pct"])
    if dc is None:
        raise ValueError("Could not compute a valid DC operating point with the provided parameters.")

    rows, rec = sweep_vanes(
        dc,
        t_id,
        cath_id,
        inputs["etac_pct"],
        inputs["rp"],
        inputs["fill"],
        inputs.get("duty_cycle", 1.0),
        inputs.get("la_ratio"),
    )
    score_lines, justification, best_type = justify_type(
        t_id,
        f_ghz,
        P_kw,
        t["duty"] == "cw",
        t["QL"],
    )

    rows_out = []
    for rr in rows:
        if not rr.get("ok"):
            rows_out.append({
                "Nv": rr["Nv"],
                "ok": False,
                "message": rr.get("msg"),
            })
            continue
        rows_out.append({
            "Nv": rr["Nv"],
            "ok": True,
            "fatal": rr["fatal"],
            "ra_mm": rr["ra_mm"],
            "rc_mm": rr["rc_mm"],
            "xi": rr["xi"],
            "gap_mm": rr["ra_mm"] - rr["rc_mm"],
            "B0_mT": rr["B0_mT"],
            "Bz_mT": rr["Bz_mT"],
            "pitch_mm": rr["pitch_mm"],
            "La_mm": rr["La_mm"],
            "w_gap_mm": rr["w_gap_mm"],
            "d_cav_mm": rr["d_cav_mm"],
            "t_vane_mm": rr["t_vane_mm"],
            "Jc_A_per_cm2": rr["Jc"],
            "Jlim_A_per_cm2": rr["Jlim"],
            "vane_tip_temp_C": rr.get("Tv_tip_C"),
            "VTn1_kV": rr["VTn1_kV"],
            "mode_sep_pct": rr["msep"],
            "QU": rr["QU"],
            "b": rr["b"],
            "v": rr["v"],
            "ii": rr["ii"],
            "g": rr.get("g"),
            "p": rr.get("p"),
            "issues": _issues_to_list(rr["issues"]),
            "score": rr["score"],
        })

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "freq_GHz": f_ghz,
            "power_kW": P_kw,
            "type": t_id,
            "type_label": t["label"],
            "duty": t["duty"],
            "duty_cycle": inputs.get("duty_cycle", 1.0),
            "cathode": cath_id,
            "cathode_label": CATHODES[cath_id]["label"],
            "eta_pct": eta_pct,
            "zdc_kohm": zdc_k,
            "etac_pct": inputs["etac_pct"],
            "rp": inputs["rp"],
            "fill": inputs["fill"],
            "la_ratio": inputs.get("la_ratio"),
            "preset": inputs["preset"],
            "load_db": inputs["load_db"],
        },
        "dc_operating_point": {
            "Va_kV": dc["Va"] / 1e3,
            "Ia_A": dc["Ia"],
            "Pdc_kW": dc["Pdc"] / 1e3,
            "eta_pct": dc["eta"] * 100,
            "etaE_pct": dc["etaE"] * 100,
            "r": dc["r"],
            "V0_kV": dc["V0"] / 1e3,
            "VH_kV": dc["VH"] / 1e3,
            "Va_over_VH": dc["VaVH"],
            "lambda_mm": dc["lam"] * 1e3,
            "delta_s_um": dc["delta_s_um"],
            "Rs_mOhm_per_sq": dc["Rs_mOhm"],
        },
        "recommended": None,
        "vane_sweep": rows_out,
        "type_analysis": {
            "selected": t_id,
            "recommended_type": best_type,
            "score_lines": score_lines,
            "justification": justification,
        },
        "frequency_stability": [
            {
                "type": label,
                "QL": ql,
                "pulling_MHz": pull,
                "loading_df_MHz": df,
            }
            for label, ql, pull, df in freq_stability_table(f_ghz)
        ],
    }

    if rec is not None:
        payload["recommended"] = {
            "Nv": rec["Nv"],
            "ra_mm": rec["ra_mm"],
            "rc_mm": rec["rc_mm"],
            "xi": rec["xi"],
            "gap_mm": rec["ra_mm"] - rec["rc_mm"],
            "B0_mT": rec["B0_mT"],
            "Bz_mT": rec["Bz_mT"],
            "pitch_mm": rec["pitch_mm"],
            "La_mm": rec["La_mm"],
            "w_gap_mm": rec["w_gap_mm"],
            "t_vane_mm": rec["t_vane_mm"],
            "d_cav_mm": rec["d_cav_mm"],
            "Jc_A_per_cm2": rec["Jc"],
            "Jlim_A_per_cm2": rec["Jlim"],
            "vane_tip_temp_C": rec.get("Tv_tip_C"),
            "mode_sep_pct": rec["msep"],
            "VTn1_kV": rec["VTn1_kV"],
            "QU": rec["QU"],
            "b": rec["b"],
            "v": rec["v"],
            "ii": rec["ii"],
            "g": rec.get("g"),
            "p": rec.get("p"),
            "issues": _issues_to_list(rec["issues"]),
            "score": rec["score"],
        }

    # Always report dimensionless parameters when at least one non-fatal candidate exists.
    rec_for_params = rec
    if rec_for_params is None:
        rec_for_params = next((rr for rr in rows if rr.get("ok") and not rr.get("fatal")), None)
    if rec_for_params is not None:
        payload["dimensionless_parameters"] = _dimensionless_blocks_from_dc_rec(dc, rec_for_params)

    payload["typical_parameter_ranges"] = _get_typical_parameter_ranges()

    return payload


def _dimensionless_blocks_from_dc_rec(dc, rec):
    """Return Collins/Clogston dimensionless parameter block.

    Collins/Clogston reduced parameters follow Collins Eq. (25) definitions:
      b=B/B̄, v=V/V̄, i=I/Ī, g=G_L/Ḡ, p=P/P̄.
    """
    if dc is None or rec is None:
        return None

    collins = {
        "b": rec.get("b", float("nan")),
        "v": rec.get("v", float("nan")),
        "i": rec.get("ii", float("nan")),
        "g": rec.get("g", float("nan")),
        "p": rec.get("p", float("nan")),
    }

    return {"collins": collins}


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile for pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


@lru_cache(maxsize=1)
def _get_typical_parameter_ranges():
    """Estimate typical parameter ranges from the reference magnetron database.

    The ranges are computed by running the same design equations on each reference
    operating point and taking the recommended (non-fatal) candidate. We report
    the 10th–90th percentile interval as a "typical" range.
    """
    collins_vals = {"b": [], "v": [], "i": [], "g": [], "p": []}

    for m in REFERENCE_MAGNETRONS:
        try:
            inputs = _resolve_design_inputs({"load_db": m["id"]})
            t_id = inputs["type_id"]
            cath_id = inputs["cath_id"]
            t = TYPES[t_id]
            eta_pct = inputs["eta_override"] if inputs["eta_override"] is not None else t["eta_fn"](inputs["f_ghz"])
            zdc_k = inputs["zdc_override"] if inputs["zdc_override"] is not None else t["zdc_fn"](inputs["f_ghz"])
            dc = compute_dc_point(inputs["f_ghz"], inputs["P_kw"], eta_pct, zdc_k, inputs["etac_pct"])
            if dc is None:
                continue
            _rows, rec = sweep_vanes(dc, t_id, cath_id, inputs["etac_pct"], inputs["rp"], inputs["fill"])
            if rec is None:
                continue

            blocks = _dimensionless_blocks_from_dc_rec(dc, rec)
            if not blocks:
                continue

            for k, v in blocks["collins"].items():
                fv = float(v)
                if math.isfinite(fv):
                    collins_vals[k].append(fv)
        except Exception:
            continue

    def _mk_ranges(d):
        out = {}
        for k, vals in d.items():
            vv = sorted(vals)
            if not vv:
                out[k] = {"n": 0, "p10": float("nan"), "p90": float("nan")}
                continue
            out[k] = {
                "n": len(vv),
                "p10": _percentile(vv, 0.10),
                "p90": _percentile(vv, 0.90),
            }
        return out

    return {
        "collins": _mk_ranges(collins_vals),
        "source": "REFERENCE_MAGNETRONS (10th–90th percentile of recommended candidates)",
    }


def create_app():
    if Flask is None:
        raise RuntimeError("Flask is not installed. Install it with: pip install flask")

    def _json_sanitize(obj):
        """Recursively convert non-finite floats (NaN/±Inf) to None for strict JSON."""
        if obj is None:
            return None
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_sanitize(v) for v in obj]
        return obj

    app = Flask(__name__)

    # UI template moved to templates/magnetron_design.html

    @app.get("/")
    def api_root():
        return jsonify({
            "service": "Magnetron Design Calculator API",
            "endpoints": {
                "health": "/health",
                "ui": "/ui",
                "calculate_get": "/calculate?freq=9.375&power=250&type=s_pls&cath=disp",
                "calculate_post": "/calculate",
                "estimate": "/estimate?type=s_pls&freq=9.375",
                "types": "/types",
                "cathodes": "/cathodes",
                "presets": "/presets",
                "reference_db": "/reference-db",
            },
        })

    @app.get("/ui")
    def ui_page():
        # TYPES contains callables (eta_fn, zdc_fn); keep only UI-safe fields.
        # UI combines strapped CW/pulsed into a single selector entry (id: "s").
        ui_types = {}
        for k, v in TYPES.items():
            if k in {"s_cw", "s_pls"}:
                continue
            ui_types[k] = {
                "label": v["label"],
                "duty": v["duty"],
                "QL": v["QL"],
                "apps": v["apps"],
                "desc": v["desc"],
            }
        # Use CW strapped metadata as the base UI description.
        base = TYPES["s_cw"]
        ui_types["s"] = {
            "label": "Strapped",
            "duty": "duty",
            "QL": base["QL"],
            "apps": base["apps"],
            "desc": "Strapped magnetron; CW vs pulsed behavior controlled by duty cycle.",
        }
        return render_template(
            "magnetron_design.html",
            types_data=json.dumps(ui_types),
            cathodes_data=json.dumps(CATHODES),
            presets_data=json.dumps(PRESETS),
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "timestamp_utc": datetime.now(timezone.utc).isoformat()})

    @app.get("/estimate")
    def estimate():
        """Return UI-friendly estimated efficiency and DC impedance.

        This exists because the UI cannot receive Python callables (eta_fn/zdc_fn)
        via JSON; the UI calls this endpoint when type/frequency changes.
        """
        type_id = request.args.get("type", "").strip()
        freq_s = request.args.get("freq", "").strip()
        duty_s = request.args.get("duty", "").strip()
        try:
            freq = float(freq_s)
        except Exception:
            return jsonify({"error": "Invalid freq"}), 400

        duty_cycle = 1.0
        if duty_s:
            try:
                duty_cycle = float(duty_s)
            except Exception:
                return jsonify({"error": "Invalid duty"}), 400
        if not (0 < duty_cycle <= 1.0):
            return jsonify({"error": "duty must be in (0, 1]"}), 400

        # Combined strapped type for UI
        if type_id.lower() in {"s", "strapped"}:
            type_id = "s_cw" if duty_cycle >= 1.0 else "s_pls"

        if type_id not in TYPES:
            return jsonify({"error": "Invalid type"}), 400

        try:
            eta = TYPES[type_id]["eta_fn"](freq)
            zdc = TYPES[type_id]["zdc_fn"](freq)
        except Exception:
            return jsonify({"error": "Estimation failed"}), 500

        return jsonify({"eta": eta, "zdc": zdc})

    @app.get("/types")
    def list_types():
        return jsonify({
            "types": [
                {
                    "id": k,
                    "label": v["label"],
                    "duty": v["duty"],
                    "QL": v["QL"],
                    "applications": v["apps"],
                }
                for k, v in TYPES.items()
            ]
        })

    @app.get("/cathodes")
    def list_cathodes():
        return jsonify({
            "cathodes": [
                {
                    "id": k,
                    "label": v["label"],
                    "Jcw": v["Jcw"],
                    "Jpulse": v["Jpulse"],
                    "Tmax_C": v["Tmax"],
                }
                for k, v in CATHODES.items()
            ]
        })

    @app.get("/presets")
    def list_presets():
        return jsonify({"presets": PRESETS})

    @app.get("/reference-db")
    def reference_db():
        return jsonify({"reference_magnetrons": REFERENCE_MAGNETRONS})

    @app.route("/calculate", methods=["GET", "POST"])
    def calculate():
        body = request.get_json(silent=True) or {}
        # JSON body takes precedence over query string for overlapping keys.
        raw_params = {**request.args.to_dict(), **body}
        try:
            inputs = _resolve_design_inputs(raw_params)
            payload = compute_design_payload(inputs)
            return jsonify(_json_sanitize(payload))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive API guard
            return jsonify({"error": "Internal server error", "detail": str(exc)}), 500

    return app


# ─── Argument parsing ─────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Magnetron Design Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-f",  "--freq",    type=float, metavar="GHz",
                   help="Operating frequency (GHz)")
    p.add_argument("-p",  "--power",   type=float, metavar="kW",
                   help="Output power in kW (peak for pulsed, average for CW)")
    p.add_argument("-t",  "--type",    choices=list(TYPES.keys()), metavar="TYPE",
                   help=f"Magnetron type: {', '.join(TYPES.keys())}")
    p.add_argument("--cath", choices=list(CATHODES.keys()), default="disp",
                   metavar="CATHODE",
                   help=f"Cathode material: {', '.join(CATHODES.keys())} (default: disp)")
    p.add_argument("--eta",   type=float, metavar="%",
                   help="Overall efficiency %% (overrides auto-estimate)")
    p.add_argument("--zdc",   type=float, metavar="kΩ",
                   help="DC impedance Va/Ia in kΩ (overrides auto-estimate)")
    p.add_argument("--etac",  type=float, default=90.0, metavar="%",
                   help="Circuit efficiency %% (default: 90)")
    p.add_argument("--rp",    type=float, default=2.0, metavar="R'",
                   help="Modified Slater factor R' (default: 2.0)")
    p.add_argument("--fill",  type=float, default=0.45, metavar="w/p",
                   help="Fill factor w/p (default: 0.45)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), metavar="NAME",
                   help=f"Load a preset: {', '.join(PRESETS.keys())}")
    p.add_argument("--list-db",      action="store_true",
                   help="Print reference magnetron database and exit")
    p.add_argument("--load-db",      metavar="ID",
                   help="Load operating parameters from a database entry ID")
    p.add_argument("--compare-types", action="store_true",
                   help="Print type comparison and suitability analysis")
    p.add_argument("--no-color",     action="store_true",
                   help="Disable ANSI colour output")
    p.add_argument("--flask", action="store_true",
                   help="Start Flask API server")
    p.add_argument("--host", default="127.0.0.1",
                   help="Flask host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=5000,
                   help="Flask port (default: 5000)")
    return p


def interactive_prompt(prompt, default, cast=float, choices=None):
    suffix = f" [{default}]" if default is not None else ""
    if choices:
        suffix = f" ({'/'.join(choices)}){suffix}"
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return cast(default) if cast else default
        try:
            val = cast(raw) if cast else raw
            if choices and str(val) not in choices:
                print(f"  ✗ Must be one of: {', '.join(choices)}")
                continue
            return val
        except ValueError:
            print(f"  ✗ Invalid input — enter a {cast.__name__}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR
    parser = build_parser()
    args   = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    if args.flask:
        if Flask is None:
            print(red("  ✗ Flask is not installed. Install it with: pip install flask"))
            sys.exit(1)
        app = create_app()
        print(cyan(f"\n  Starting Flask API on http://{args.host}:{args.port}"))
        print(dim("  Endpoints: /health, /calculate, /types, /cathodes, /presets, /reference-db"))
        app.run(host=args.host, port=args.port)
        return

    if args.list_db:
        print_reference_db()
        return

    # ── Resolve parameters ──────────────────────────────────────────────
    f_ghz = args.freq
    P_kw  = args.power
    t_id  = args.type
    cath  = args.cath
    eta   = args.eta
    Zdc   = args.zdc
    etaC  = args.etac
    Rp    = args.rp
    fill  = args.fill

    # Apply preset first
    if args.preset:
        ps = PRESETS[args.preset]
        f_ghz = f_ghz or ps["f"]
        P_kw  = P_kw  or ps["P"]
        t_id  = t_id  or ps["type"]
        cath  = cath  or ps.get("cath", "disp")
        eta   = eta   or ps.get("eta")
        Zdc   = Zdc   or ps.get("Zdc")
        etaC  = etaC  if args.etac != 90.0 else ps.get("etaC", 90.0)
        Rp    = Rp    if args.rp   != 2.0  else ps.get("Rp", 2.0)
        fill  = fill  if args.fill != 0.45 else ps.get("fill", 0.45)
        print(bold(cyan(f"\n  Preset: {ps['desc']}")))

    # Load from database
    if args.load_db:
        db_id = args.load_db.lower()
        match = next((m for m in REFERENCE_MAGNETRONS if m["id"] == db_id), None)
        if match is None:
            print(red(f"  ✗ Database ID '{db_id}' not found. Use --list-db to see all IDs."))
            sys.exit(1)
        f_ghz = f_ghz or match["f"]
        P_kw  = P_kw  or match["P"]
        t_id  = t_id  or match["type"]
        cath  = cath  or match["cath"]
        eta   = eta   or match["eta"]
        Zdc   = Zdc   or match["Zdc"]
        print(bold(cyan(f"\n  Loaded from DB: {match['model']}")))
        print(dim(f"  {match['app']}"))

    # Interactive prompts for missing required fields
    if f_ghz is None or P_kw is None or t_id is None:
        print_header("Magnetron Design Calculator — Interactive Mode")
        print()
        print(bold("  Available types:"))
        for k, v in TYPES.items():
            print(f"    {k:<8}  {v['label']}")
        print()
        if f_ghz is None:
            f_ghz = interactive_prompt("Frequency (GHz)", 9.375)
        if P_kw is None:
            P_kw  = interactive_prompt("Output power (kW)", 250)
        if t_id is None:
            t_id  = interactive_prompt("Magnetron type", "s_pls", cast=str,
                                       choices=list(TYPES.keys()))
        print()
        print(bold("  Optional overrides (press Enter to use auto-estimate):"))
        eta_in  = interactive_prompt("Overall efficiency % (blank=auto)", None,
                                     cast=lambda x: float(x) if x else None)
        if eta_in:
            eta = eta_in
        Zdc_in  = interactive_prompt("DC impedance kΩ (blank=auto)", None,
                                     cast=lambda x: float(x) if x else None)
        if Zdc_in:
            Zdc = Zdc_in

    # Resolve auto-estimated values
    t = TYPES[t_id]
    eta_pct = eta  if eta  is not None else t["eta_fn"](f_ghz)
    Zdc_k   = Zdc  if Zdc  is not None else t["zdc_fn"](f_ghz)

    # ── Print header ────────────────────────────────────────────────────
    print_header("Magnetron Design Calculator")
    print(f"\n  {'Frequency':>22}  {bold(f'{f_ghz:.4g} GHz')}")
    print(f"  {'Output power':>22}  {bold(f'{P_kw:.4g} kW')}  {'(average — CW)' if t['duty'] == 'cw' else '(peak — pulsed)'}")
    print(f"  {'Type':>22}  {bold(t['label'])}")
    print(f"  {'Cathode':>22}  {bold(CATHODES[cath]['label'])}")
    print(f"  {'Efficiency (auto/override)':>22}  {eta_pct:.1f}%  {'(override)' if eta is not None else dim('(auto-estimated)')}")
    print(f"  {'DC impedance (auto/override)':>22}  {Zdc_k:.3g} kΩ  {'(override)' if Zdc is not None else dim('(auto-estimated)')}")
    print(f"  {'Circuit efficiency ηc':>22}  {etaC:.1f}%")
    print(f"  {'Modified Slater factor R′':>22}  {Rp:.2f}")
    print(f"  {'Fill factor w/p':>22}  {fill:.2f}")
    print(dim(f"\n  Physics: Carter (2018) §15 · Collins & Clogston MIT Rad Lab Vol.6 (1948) §16"))
    print(dim(f"  McDowell IEEE Trans. Plasma Sci. 26 (1998) · Liu et al. MetalMat (2024)"))

    # ── Compute ─────────────────────────────────────────────────────────
    dc = compute_dc_point(f_ghz, P_kw, eta_pct, Zdc_k, etaC)
    if dc is None:
        print(red("\n  ✗ Could not compute a valid DC operating point."))
        print(red("    Check that efficiency and impedance give a physically realizable Va."))
        sys.exit(1)

    print_dc_results(dc, t_id)

    rows, rec = sweep_vanes(dc, t_id, cath, etaC, Rp, fill)
    print_vane_table(rows, rec, t["duty"] == "cw")

    print_recommended(rec, dc, fill, t_id)

    print_cathode_info(cath, t_id)

    if args.compare_types:
        print_type_comparison(f_ghz, P_kw, t["duty"] == "cw", t_id)

    # ── Design description ──────────────────────────────────────────────
    print_section(f"Type Description: {t['label']}")
    words = t["desc"].split()
    line = "  "
    for w in words:
        if len(line) + len(w) > 88:
            print(line)
            line = "  " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)
    print()
    print(bold("  Pros:"))
    for p_item in t["pros"]:
        print(green(f"    ✓ {p_item}"))
    print(bold("  Cons:"))
    for c_item in t["cons"]:
        print(yellow(f"    ✗ {c_item}"))
    print(dim(f"\n  Applications: {t['apps']}"))

    print()
    print(hr("═"))
    print(dim("  References:"))
    print(dim("  Carter R.G. (2018) Microwave and RF Vacuum Electronic Power Sources, Cambridge UP"))
    print(dim("  Collins G.B. et al. (1948) Microwave Magnetrons, MIT Rad Lab Series Vol.6, McGraw-Hill"))
    print(dim("  Clogston A.M. (1948) 'Principles of Design', in Collins §16"))
    print(dim("  McDowell H.L. (1998) IEEE Trans. Plasma Sci. 26, 733–754"))
    print(dim("  Liu Z. et al. (2024) MetalMat — cathode emission database"))
    print(dim("  Teledyne e2v MG5193 datasheet A1A-MG5193 v10 (2023)"))
    print(dim("  Adler R. (1946) Proc. IRE 34, 351 · Tahir I. et al. (2005) IEEE Trans. Electron Dev. 52, 2096"))
    print(hr("═"))
    print()


if __name__ == "__main__":
    main()

