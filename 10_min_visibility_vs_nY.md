# Minimum Critical Visibility vs Number of Bob's Measurements

Sweeps over `nY ∈ {2, 3, ..., 20}` and computes the minimum critical PAM  
visibility `v*` over 500 random measurement seeds for a fixed set of three  
qubit preparations.

**Preparations used:**
- `r0 = (+0.866026, 0, -0.500000)` — equatorial, 120° apart
- `r1 = (0, 0, +1.000000)`
- `r2 = (-0.866025, 0, -0.500000)`

**Method:** for each `nY`, 500 independently sampled projective measurement  
configurations are tested via a linear programme (HiGHS solver via  
`scipy.optimize.linprog`). The minimum `v*` across seeds is recorded.

**Output:** console table, saved plot (`min_visibility_vs_nY.pdf/.jpg`),  
and summary table printed at the end.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

# Minimum Visibility $v^*$ vs Number of Bob's Measurements

PAM (3, 2, 2, 2) scenario with states:
- $\vec{r}_0 = (+0.866026,\ 0,\ -0.500000)$
- $\vec{r}_1 = (0,\ 0,\ +1.000000)$
- $\vec{r}_2 = (-0.866025,\ 0,\ -0.500000)$


## Cell 1 — Imports


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import lil_matrix
from scipy.optimize import linprog
import time

matplotlib.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 13,
    'axes.labelsize'  : 14,
    'legend.fontsize' : 12,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
    'figure.dpi'      : 150,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})
```


## Cell 2 — Helper functions


```python
def toSeveralBases(n, bases):
    bases = np.array(bases, dtype=int).copy()
    bases[bases == 0] = 1
    n = np.atleast_1d(np.array(n, dtype=int)).ravel()
    result = np.zeros((len(n), len(bases)), dtype=int)
    n_copy = n.copy()
    for j in range(len(bases) - 1, -1, -1):
        result[:, j] = n_copy % bases[j]
        n_copy = (n_copy - result[:, j]) // bases[j]
    return result


def random_unitary(d):
    Z = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R) / np.abs(np.diag(R))
    return Q * D


def build_measurements(nY, nB, seed):
    np.random.seed(seed)
    M = np.zeros((2, 2, nB, nY), dtype=complex)
    M[:,:,0,0] = np.diag([1.0, 0.0])
    M[:,:,1,0] = np.diag([0.0, 1.0])
    for k in range(1, nY):
        U = random_unitary(2)
        M[:,:,0,k] = U @ M[:,:,0,0] @ U.conj().T
        M[:,:,1,k] = U @ M[:,:,1,0] @ U.conj().T
    return M


def compute_pTarget(rho, M, nX, nB, nY):
    pTarget = np.zeros((nB, nX, nY))
    for x in range(nX):
        for y in range(nY):
            for b in range(nB):
                pTarget[b,x,y] = np.real(np.trace(rho[:,:,x] @ M[:,:,b,y]))
    return pTarget


def build_fA(nEnc, nX, d):
    fA = np.zeros((nEnc, nX), dtype=int)
    for lam in range(nEnc):
        fA[lam, :] = toSeveralBases(lam, d * np.ones(nX, dtype=int)).ravel() + 1
    return fA


def solve_LP(rho, nX, d, nB, nY, seed):
    nEnc   = d**nX
    nVar   = 1 + nEnc * (nB * nY * d + 1)
    nCon   = 1 + nY * d * nEnc + nB * nX * nY
    nShift = nEnc * nY * nB * d

    fA      = build_fA(nEnc, nX, d)
    M       = build_measurements(nY, nB, seed)
    pTarget = compute_pTarget(rho, M, nX, nB, nY)
    pRand   = np.ones((nB, nX, nY)) / nB

    A = lil_matrix((nCon, nVar), dtype=float)
    tick = 0

    # Constraint 1: marginalisation
    for y in range(nY):
        for m in range(d):
            for lam in range(nEnc):
                for b in range(nB):
                    pos = lam + nEnc*b + nEnc*nB*y + nEnc*nB*nY*m
                    A[tick, pos] = 1.0
                A[tick, nShift + lam] = -1.0
                tick += 1

    # Constraint 2: classical model
    for x in range(nX):
        for y in range(nY):
            for b in range(nB):
                for m in range(d):
                    for lam in range(nEnc):
                        if fA[lam, x] == m + 1:
                            pos = lam + nEnc*b + nEnc*nB*y + nEnc*nB*nY*m
                            A[tick, pos] = 1.0
                tick += 1

    # Constraint 3: normalisation
    A[nCon - 1, nShift : nShift + nEnc] = 1.0

    # Last column: coefficient of v
    tick2 = 0
    for x in range(nX):
        for y in range(nY):
            for b in range(nB):
                A[nEnc*nY*d + tick2, nVar - 1] = pRand[b,x,y] - pTarget[b,x,y]
                tick2 += 1

    A = A.tocsr()

    bvec = np.zeros(nCon)
    bvec[-1] = 1.0
    tick2 = 0
    for x in range(nX):
        for y in range(nY):
            for b in range(nB):
                bvec[nEnc*nY*d + tick2] = pRand[b, x, y]
                tick2 += 1

    cvec = np.zeros(nVar)
    cvec[-1] = 1.0

    result = linprog(
        c      = -cvec,
        A_eq   = A,
        b_eq   = bvec,
        bounds = [(0.0, None)] * nVar,
        method = 'highs',
        options= {'disp': False}
    )
    return -result.fun
```


## Cell 3 — Quantum states


```python
id2  = np.eye(2, dtype=complex)
X_op = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
Y_op = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
Z_op = np.array([[1,  0 ], [0, -1 ]], dtype=complex)

r0 = np.array([ 0.866026,  0.000000, -0.500000])
r1 = np.array([ 0.000000,  0.000000,  1.000000])
r2 = np.array([-0.866025,  0.000000, -0.500000])

nX = 3
rho = np.zeros((2, 2, nX), dtype=complex)
rho[:,:,0] = (id2 + r0[0]*X_op + r0[1]*Y_op + r0[2]*Z_op) / 2
rho[:,:,1] = (id2 + r1[0]*X_op + r1[1]*Y_op + r1[2]*Z_op) / 2
rho[:,:,2] = (id2 + r2[0]*X_op + r2[1]*Y_op + r2[2]*Z_op) / 2

d  = 2
nB = 2

print("States:")
for x, r in enumerate([r0, r1, r2]):
    print(f"  rho[{x}]: r = {r},  |r| = {np.linalg.norm(r):.6f}")
```


## Cell 4 — Sweep: minimum $v^*$ vs $n_Y$

For each $n_Y \in \{2, 3, \ldots, 20\}$ we run **500 random seeds** and keep only
the minimum $v^*$. This large number of seeds is what ensures we get close
to the true minimum over all possible measurement configurations.


```python
nY_values = list(range(2, 21))   # nY = 2, 3, ..., 20
n_seeds   = 500                  # seeds per nY — large enough to approach true minimum

v_min_per_nY = []

t_total = time.time()
for nY in nY_values:
    t0 = time.time()
    v_list = []
    for seed in range(n_seeds):
        v = solve_LP(rho, nX, d, nB, nY, seed=seed)
        v_list.append(v)
    v_min = min(v_list)
    v_min_per_nY.append(v_min)
    dt = time.time() - t0
    print(f"nY = {nY:2d}:  min v* = {v_min:.6f}   ({dt:.1f}s)")

print(f"\nTotal time: {time.time()-t_total:.1f}s")
```


```python
def trunc3(x):
    return np.floor(x * 1000) / 1000

# E use no text:
f'{trunc3(yi):.3f}',
```


## Cell 5 — Plot


```python
nY_arr = np.array(nY_values)
v_arr  = np.array(v_min_per_nY)

def trunc3(x):
    return np.floor(x * 1000) / 1000

selected_nY  = [2, 3, 4, 5, 6, 8, 13, 20]
selected_idx = [list(nY_arr).index(n) for n in selected_nY]

v_sel = v_arr[selected_idx]
x_pos = np.arange(len(selected_nY))

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    x_pos, v_sel,
    color='yellow',           # curva amarela
    linewidth=2.5,
    marker='o',
    markersize=9,             # bolinhas maiores
    markerfacecolor='yellow',
    markeredgecolor='black',  # contorno para destacar
    markeredgewidth=0.8,
    label=r'Minimum $v$',
    zorder=3
)

for xi, yi in zip(x_pos, v_sel):
    ax.text(
        xi, yi + 0.005,
        f'{trunc3(yi):.3f}',
        ha='center', va='bottom',
        fontsize=13,
        color='green',        # números verdes
        fontweight='bold'     # negrito
    )

ax.set_xlabel("Number of Bob's measurements ($n_Y$)", labelpad=8)
ax.set_ylabel(r'Critical visibility $v$', labelpad=8)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(n) for n in selected_nY])
ax.set_xlim([-0.4, len(selected_nY) - 0.6])
ax.set_ylim([0.750, 0.850])
ax.set_yticks(np.arange(0.750, 0.851, 0.010))
ax.legend(frameon=False, loc='upper right')

# sem grid
ax.grid(False)

plt.tight_layout()
plt.savefig('min_visibility_vs_nY.jpg', bbox_inches='tight', dpi=300)
plt.savefig('min_visibility_vs_nY.pdf', bbox_inches='tight')
plt.show()
print("Figures saved.")
```


## Cell 6 — Summary table


```python
print(f"{'nY':>4}  {'min v*':>10}")
print("-" * 18)
for nY, v in zip(nY_values, v_min_per_nY):
    print(f"{nY:>4}  {v:>10.6f}")
```

