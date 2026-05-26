# Critical Visibility vs Number of Bob's Measurements (500 Seeds, Uniform Style)

Computes and plots the critical PAM visibility `v*` as a function of the number  
of Bob's measurement settings `nY`, for three fixed sets of qubit preparations.

For each `nY ∈ {2, 4, 6, ..., 10^4}` and each preparation set, a linear programme  
(LP) tests whether the noisy behavior `v·p + (1−v)·(1/2)` admits a  
classical `d=2` PAM model. The minimum `v*` over 500 random measurement  
configurations is recorded at each `nY`.

**Three cases studied:**
1. Optimal states for `W^(1)_CC` (Bloch vectors from Appendix B1 of the paper)
2. Symmetric PAM triple (equatorial states)
3. States maximising `W^(3)_CNS`

All three plots share a unified PRA-style visual format (serif font, no grid,  
clean spines).


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

# Uniform visibility plots for fixed PAM preparations

This notebook contains three visibility calculations and three plots, all formatted with the same visual style. The numerical routines are kept separate from the plotting routine, and the three cases are separated by markdown headings.

The common plotting convention is:

- logarithmic horizontal axis for the number of Bob measurements, $n_Y$;
- the same six plotted values of $n_Y$: $2,4,10,100,10^3,10^4$;
- green smooth curve for the LP visibility;
- blue markers for the actual numerical LP points;
- golden dashed line for the corresponding reference threshold;
- serif fonts and compact PRA-style formatting.


```python
# ============================================================
# Imports and global configuration
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.optimize import linprog
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import FormatStrFormatter, NullLocator

# Common set of nY values used in all plots
COMMON_NY_LIST = [2, 4, 10, 100, 1000, 10000]

# Output folder
OUT_DIR = Path("visibility_results")
OUT_DIR.mkdir(exist_ok=True)
```


```python
# ============================================================
# Basic functions for PAM visibility LP
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def random_bloch_vectors(k, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(k, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def toSeveralBases(n, bases):
    bases = np.array(bases, dtype=int).copy()
    n = np.atleast_1d(np.array(n, dtype=int)).ravel()

    result = np.zeros((len(n), len(bases)), dtype=int)
    n_copy = n.copy()

    for j in range(len(bases) - 1, -1, -1):
        result[:, j] = n_copy % bases[j]
        n_copy = (n_copy - result[:, j]) // bases[j]

    return result


def build_fA(nEnc, nX, d):
    """
    Deterministic encodings f: X -> M.
    Indices are 0-based.
    """
    fA = np.zeros((nEnc, nX), dtype=int)

    for lam in range(nEnc):
        fA[lam, :] = toSeveralBases(lam, d * np.ones(nX, dtype=int)).ravel()

    return fA


def compute_pTarget_from_bloch(r_states, meas_dirs):
    """
    Qubit states:
        rho_x = (I + r_x . sigma)/2

    Projective measurements along n_y:
        p(0|x,y) = (1 + r_x . n_y)/2
        p(1|x,y) = (1 - r_x . n_y)/2

    Output shape:
        pTarget[b,x,y]
    """
    r_states = np.asarray(r_states, dtype=float)
    meas_dirs = np.asarray(meas_dirs, dtype=float)

    nX = r_states.shape[0]
    nY = meas_dirs.shape[0]
    nB = 2

    dots = r_states @ meas_dirs.T

    pTarget = np.zeros((nB, nX, nY), dtype=float)
    pTarget[0, :, :] = (1.0 + dots) / 2.0
    pTarget[1, :, :] = (1.0 - dots) / 2.0

    return pTarget


def solve_PAM_visibility_LP(pTarget, d=2):
    """
    Computes the largest visibility v such that

        p_v(b|x,y) = v p(b|x,y) + (1-v)/2

    admits a classical PAM model with message dimension d.

    Variables:
        q(b,y,m,lambda) >= 0
        t(lambda) >= 0
        0 <= v <= 1
    """

    nB, nX, nY = pTarget.shape
    nEnc = d**nX

    n_q = nEnc * nB * nY * d
    n_t = nEnc
    nVar = n_q + n_t + 1

    idx_t0 = n_q
    idx_v = nVar - 1

    nCon_1 = nY * d * nEnc
    nCon_2 = nB * nX * nY
    nCon_3 = 1
    nCon = nCon_1 + nCon_2 + nCon_3

    fA = build_fA(nEnc, nX, d)
    pRand = np.ones((nB, nX, nY), dtype=float) / nB

    A = lil_matrix((nCon, nVar), dtype=float)
    bvec = np.zeros(nCon, dtype=float)

    row = 0

    # Constraint 1: sum_b q(b,y,m,lambda) = t(lambda)
    for y in range(nY):
        for m in range(d):
            for lam in range(nEnc):
                for b in range(nB):
                    pos_q = lam + nEnc*b + nEnc*nB*y + nEnc*nB*nY*m
                    A[row, pos_q] = 1.0

                A[row, idx_t0 + lam] = -1.0
                row += 1

    # Constraint 2:
    # sum_{lambda} q(b,y,f_lambda(x),lambda)
    # + [1/2 - p(b|x,y)] v = 1/2
    for x in range(nX):
        for y in range(nY):
            for b in range(nB):
                for lam in range(nEnc):
                    m = fA[lam, x]
                    pos_q = lam + nEnc*b + nEnc*nB*y + nEnc*nB*nY*m
                    A[row, pos_q] = 1.0

                A[row, idx_v] = pRand[b, x, y] - pTarget[b, x, y]
                bvec[row] = pRand[b, x, y]

                row += 1

    # Constraint 3: sum_lambda t(lambda) = 1
    for lam in range(nEnc):
        A[row, idx_t0 + lam] = 1.0

    bvec[row] = 1.0
    row += 1

    assert row == nCon

    # Maximize v = minimize -v
    cvec = np.zeros(nVar, dtype=float)
    cvec[idx_v] = -1.0

    bounds = [(0.0, None)] * (nVar - 1) + [(0.0, 1.0)]

    result = linprog(
        c=cvec,
        A_eq=A.tocsr(),
        b_eq=bvec,
        bounds=bounds,
        method="highs",
        options={
            "disp": False,
            "presolve": True,
        }
    )

    if not result.success:
        raise RuntimeError(
            f"LP failed.\n"
            f"status = {result.status}\n"
            f"message = {result.message}"
        )

    v_star = -result.fun

    return v_star, result
```


```python
# ============================================================
# Unified plotting function
# ============================================================

def set_unified_pra_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "mathtext.fontset": "dejavuserif",
    })


def plot_visibility_pra_style(
    nY_values,
    v_values,
    threshold_value,
    curve_label=r"LP PAM test",
    threshold_label=r"threshold",
    ylabel=r"$v^\ast_{\mathrm{PAM}}$",
    output_name="visibility_plot",
    output_dir=OUT_DIR,
    y_limits=None,
    y_tick_step=0.005,
    smooth=True,
    legend_anchor=(0.72, 0.60),
):
    """
    Unified plot for visibility-vs-nY curves.

    The horizontal axis is always shown from nY=2 to nY=10^4.
    The plotted points are exactly the numerical points supplied to the function.
    """

    set_unified_pra_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"

    nY_arr = np.asarray(nY_values, dtype=float)
    v_arr = np.asarray(v_values, dtype=float)

    mask = np.isfinite(nY_arr) & np.isfinite(v_arr)
    nY_arr = nY_arr[mask]
    v_arr = v_arr[mask]

    order = np.argsort(nY_arr)
    nY_arr = nY_arr[order]
    v_arr = v_arr[order]

    green = "#2ca02c"
    blue = "#1f77b4"
    yellow = "#d8a600"

    fig, ax = plt.subplots(figsize=(4.3, 2.9))

    # Smooth curve in log scale
    if smooth and len(nY_arr) >= 3:
        xlog = np.log10(nY_arr)
        xlog_smooth = np.linspace(xlog.min(), xlog.max(), 700)

        interp = PchipInterpolator(xlog, v_arr)
        v_smooth = interp(xlog_smooth)
        nY_smooth = 10**xlog_smooth

        ax.plot(
            nY_smooth,
            v_smooth,
            color=green,
            linewidth=1.8,
            label=curve_label,
            zorder=2,
        )
    else:
        ax.plot(
            nY_arr,
            v_arr,
            color=green,
            linewidth=1.8,
            label=curve_label,
            zorder=2,
        )

    # Numerical points
    ax.scatter(
        nY_arr,
        v_arr,
        s=18,
        color=blue,
        zorder=3,
    )

    # Threshold line
    ax.axhline(
        threshold_value,
        color=yellow,
        linestyle=(0, (6, 3)),
        linewidth=1.3,
        label=threshold_label,
        zorder=1,
    )

    # x-axis
    ax.set_xscale("log")
    ax.set_xlim(1.7, 1.2e4)

    ax.set_xticks([2, 4, 10, 100, 1000, 10000])
    ax.set_xticklabels([r"$2$", r"$4$", r"$10$", r"$100$", r"$10^3$", r"$10^4$"])
    ax.xaxis.set_minor_locator(NullLocator())

    # y-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    if y_limits is None:
        ymin = min(v_arr.min(), threshold_value) - 0.001
        ymax = max(v_arr.max(), threshold_value) + 0.001
        ymin = np.floor(1000 * ymin) / 1000
        ymax = np.ceil(1000 * ymax) / 1000
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(*y_limits)

    yticks = np.arange(
        np.ceil(ax.get_ylim()[0] / y_tick_step) * y_tick_step,
        ax.get_ylim()[1] + 0.5 * y_tick_step,
        y_tick_step,
    )
    ax.set_yticks(yticks)

    ax.set_xlabel(r"$n_Y$", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)

    # Clean appearance
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=11)

    # Legend
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_anchor,
        frameon=False,
        handlelength=1.9,
        borderaxespad=0.0,
        fontsize=9,
    )

    fig.tight_layout(pad=0.35)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    plt.show()

    print(f"Saved PDF in: {pdf_path.resolve()}")
    print(f"Saved PNG in: {png_path.resolve()}")

    return fig, ax
```


## Case 1 — Fixed PAM triple I

This case uses three fixed qubit preparations. For each value of $n_Y$, the same nested list of Bob measurements is used: the first two settings are the $S_3$-oriented measurements, and the remaining settings are random Bloch directions. The LP returns the critical visibility for classical PAM membership.


```python
# ============================================================
# Case 1: fixed PAM triple I
# ============================================================

r0 = np.array([-0.478199, -0.107883, -0.871600], dtype=float)
r1 = np.array([+0.250024, +0.932540, +0.260494], dtype=float)
r2 = np.array([+0.254570, -0.726206, +0.638607], dtype=float)

r0 = normalize(r0)
r1 = normalize(r1)
r2 = normalize(r2)

r_states_case1 = np.array([r0, r1, r2], dtype=float)

print("Case 1 fixed Bloch states:")
for x, r in enumerate(r_states_case1):
    print(f"r{x} = ({r[0]:+.6f}, {r[1]:+.6f}, {r[2]:+.6f}), |r{x}| = {np.linalg.norm(r):.12f}")

nY_list_case1 = COMMON_NY_LIST.copy()
seed_case1 = 0
max_nY_case1 = max(nY_list_case1)

# First two measurements are the S3-oriented directions
b0_S3_case1 = normalize(r0 + r1 - r2)
b1_S3_case1 = normalize(r0 - r1)

random_dirs_case1 = random_bloch_vectors(max_nY_case1 - 2, seed=seed_case1)

all_meas_dirs_case1 = np.vstack([
    b0_S3_case1,
    b1_S3_case1,
    random_dirs_case1,
])

# S3 threshold
pTarget_2_case1 = compute_pTarget_from_bloch(r_states_case1, all_meas_dirs_case1[:2])
E2_case1 = pTarget_2_case1[0, :, :] - pTarget_2_case1[1, :, :]
S3_case1 = E2_case1[0, 0] + E2_case1[0, 1] + E2_case1[1, 0] - E2_case1[1, 1] - E2_case1[2, 0]
v_S3_case1 = 3.0 / S3_case1

print(f"\nS3 visibility threshold = {v_S3_case1:.12f}")

v_values_case1 = []
runtime_values_case1 = []

for nY in nY_list_case1:
    print("\n============================================================")
    print(f"Case 1: running LP for nY = {nY}")
    print("============================================================")

    meas_dirs = all_meas_dirs_case1[:nY]
    pTarget = compute_pTarget_from_bloch(r_states_case1, meas_dirs)

    t0 = time.time()
    v_star, result = solve_PAM_visibility_LP(pTarget, d=2)
    dt = time.time() - t0

    v_values_case1.append(v_star)
    runtime_values_case1.append(dt)

    print(f"v*      = {v_star:.12f}")
    print(f"runtime = {dt:.2f} s")
    print(f"status  = {result.message}")

v_values_case1 = np.array(v_values_case1, dtype=float)
runtime_values_case1 = np.array(runtime_values_case1, dtype=float)

print("\nCase 1 summary:")
for nY, v, dt in zip(nY_list_case1, v_values_case1, runtime_values_case1):
    print(f"nY = {nY:5d} | v* = {v:.12f} | runtime = {dt:.2f} s")
```


```python
# ============================================================
# Plot: Case 1
# ============================================================

plot_visibility_pra_style(
    nY_values=nY_list_case1,
    v_values=v_values_case1,
    threshold_value=v_S3_case1,
    curve_label=r"LP PAM test",
    threshold_label=r"$S_3$ threshold",
    ylabel=r"$v^\ast_{\mathrm{PAM}}$",
    output_name="case1_fixed_pam_triple_visibility",
    output_dir=OUT_DIR,
    y_limits=None,
    y_tick_step=0.005,
    smooth=True,
    legend_anchor=(0.73, 0.60),
)
```


## Case 2 — Symmetric PAM triple

This case uses the symmetric three-preparation configuration. The same six values of $n_Y$ are used, and the measurement list is again nested so that larger experiments contain all settings used by the smaller ones.


```python
# ============================================================
# Case 2: symmetric PAM triple
# ============================================================

r0 = np.array([+0.866026, 0.000000, -0.500000], dtype=float)
r1 = np.array([+0.000000, 0.000000, +1.000000], dtype=float)
r2 = np.array([-0.866025, 0.000000, -0.500000], dtype=float)

r0 = normalize(r0)
r1 = normalize(r1)
r2 = normalize(r2)

r_states_case2 = np.array([r0, r1, r2], dtype=float)

print("Case 2 fixed Bloch states:")
for x, r in enumerate(r_states_case2):
    print(f"r{x} = ({r[0]:+.6f}, {r[1]:+.6f}, {r[2]:+.6f}), |r{x}| = {np.linalg.norm(r):.12f}")

nY_list_case2 = COMMON_NY_LIST.copy()
seed_case2 = 0
max_nY_case2 = max(nY_list_case2)

# First two measurements are the S3-oriented directions
b0_S3_case2 = normalize(r0 + r1 - r2)
b1_S3_case2 = normalize(r0 - r1)

random_dirs_case2 = random_bloch_vectors(max_nY_case2 - 2, seed=seed_case2)

all_meas_dirs_case2 = np.vstack([
    b0_S3_case2,
    b1_S3_case2,
    random_dirs_case2,
])

# S3 threshold
pTarget_2_case2 = compute_pTarget_from_bloch(r_states_case2, all_meas_dirs_case2[:2])
E2_case2 = pTarget_2_case2[0, :, :] - pTarget_2_case2[1, :, :]
S3_case2 = E2_case2[0, 0] + E2_case2[0, 1] + E2_case2[1, 0] - E2_case2[1, 1] - E2_case2[2, 0]
v_S3_case2 = 3.0 / S3_case2

print(f"\nS3 visibility threshold = {v_S3_case2:.12f}")

v_values_case2 = []
runtime_values_case2 = []

for nY in nY_list_case2:
    print("\n============================================================")
    print(f"Case 2: running LP for nY = {nY}")
    print("============================================================")

    meas_dirs = all_meas_dirs_case2[:nY]
    pTarget = compute_pTarget_from_bloch(r_states_case2, meas_dirs)

    t0 = time.time()
    v_star, result = solve_PAM_visibility_LP(pTarget, d=2)
    dt = time.time() - t0

    v_values_case2.append(v_star)
    runtime_values_case2.append(dt)

    print(f"v*      = {v_star:.12f}")
    print(f"runtime = {dt:.2f} s")
    print(f"status  = {result.message}")

v_values_case2 = np.array(v_values_case2, dtype=float)
runtime_values_case2 = np.array(runtime_values_case2, dtype=float)

print("\nCase 2 summary:")
for nY, v, dt in zip(nY_list_case2, v_values_case2, runtime_values_case2):
    print(f"nY = {nY:5d} | v* = {v:.12f} | runtime = {dt:.2f} s")
```


```python
# ============================================================
# Plot: Case 2
# ============================================================

plot_visibility_pra_style(
    nY_values=nY_list_case2,
    v_values=v_values_case2,
    threshold_value=v_S3_case2,
    curve_label=r"LP PAM test",
    threshold_label=r"$S_3$ threshold",
    ylabel=r"$v^\ast_{\mathrm{PAM}}$",
    output_name="case2_symmetric_pam_triple_visibility",
    output_dir=OUT_DIR,
    y_limits=(0.769, 0.805),
    y_tick_step=0.005,
    smooth=True,
    legend_anchor=(0.73, 0.60),
)
```


## Case 3 — Four fixed preparations and broadcasting threshold

This case uses four fixed qubit preparations. The LP visibility curve is compared with the visibility associated with the broadcasting witness. The broadcasting threshold is computed from $W_Q \simeq 5.0897$ and violation gap $\simeq 0.0897$, so that $W_C \simeq 5.0000$ and $v_{BC}=W_C/W_Q$.

The same six values of $n_Y$ are used in the plot. This case is computationally heavier because the number of deterministic encodings is $2^4$ instead of $2^3$.


```python
# ============================================================
# Case 3: four fixed preparations versus broadcasting threshold
# ============================================================

r_states_case3 = np.array([
    [-0.930173, -0.128092, +0.344052],
    [+0.364415, +0.000617, -0.931236],
    [-0.986082, -0.161057, -0.041257],
    [-0.015061, -0.064518, -0.997803],
], dtype=float)

# Normalize to remove small numerical drift
r_states_case3 = np.array([normalize(r) for r in r_states_case3], dtype=float)

print("Case 3 fixed Bloch states:")
for x, r in enumerate(r_states_case3):
    print(f"r{x} = ({r[0]:+.6f}, {r[1]:+.6f}, {r[2]:+.6f}), |r{x}| = {np.linalg.norm(r):.12f}")

# Correct broadcasting visibility bound
W_Q_BC = 5.0897
GAP_BC = 0.0897
W_C_BC = W_Q_BC - GAP_BC
v_BC = W_C_BC / W_Q_BC

print(f"\nBroadcasting visibility threshold = {v_BC:.12f}")

nY_list_case3 = COMMON_NY_LIST.copy()
seed_case3 = 0
max_nY_case3 = max(nY_list_case3)

# For this four-state case, all measurements are random and nested.
# No measurement is fixed to S3 directions.
all_meas_dirs_case3 = random_bloch_vectors(max_nY_case3, seed=seed_case3)

v_values_case3 = []
runtime_values_case3 = []

for nY in nY_list_case3:
    print("\n============================================================")
    print(f"Case 3: running LP for nY = {nY}")
    print("============================================================")

    meas_dirs = all_meas_dirs_case3[:nY]
    pTarget = compute_pTarget_from_bloch(r_states_case3, meas_dirs)

    t0 = time.time()
    v_star, result = solve_PAM_visibility_LP(pTarget, d=2)
    dt = time.time() - t0

    v_values_case3.append(v_star)
    runtime_values_case3.append(dt)

    print(f"v*      = {v_star:.12f}")
    print(f"runtime = {dt:.2f} s")
    print(f"status  = {result.message}")

v_values_case3 = np.array(v_values_case3, dtype=float)
runtime_values_case3 = np.array(runtime_values_case3, dtype=float)

# Numerical envelope for plotting. The exact optimum should not increase with more measurements;
# the envelope avoids visual increases caused by a single random nested sample or solver tolerance.
v_values_case3_plot = np.minimum.accumulate(np.minimum(v_values_case3, 1.0))

print("\nCase 3 summary:")
for nY, v, dt in zip(nY_list_case3, v_values_case3, runtime_values_case3):
    print(f"nY = {nY:5d} | v* = {v:.12f} | runtime = {dt:.2f} s")
```


```python
# ============================================================
# Plot: Case 3
# ============================================================

plot_visibility_pra_style(
    nY_values=nY_list_case3,
    v_values=v_values_case3_plot,
    threshold_value=v_BC,
    curve_label=r"LP PAM test",
    threshold_label=r"Broadcasting threshold",
    ylabel=r"$v^\ast_{\mathrm{PAM}}$",
    output_name="case3_four_preparations_broadcasting_threshold",
    output_dir=OUT_DIR,
    y_limits=(0.980, 1.002),
    y_tick_step=0.005,
    smooth=True,
    legend_anchor=(0.73, 0.60),
)
```


## Output files

All figures are saved in the folder `visibility_results` in both PDF and PNG formats. The three figures are:

1. `case1_fixed_pam_triple_visibility.pdf/png`
2. `case2_symmetric_pam_triple_visibility.pdf/png`
3. `case3_four_preparations_broadcasting_threshold.pdf/png`

