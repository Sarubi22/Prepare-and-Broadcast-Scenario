# Broadcast Tradeoffs: S3_B vs S3_C and CHSH_{BC|x} vs S3_C

Computes support-function upper and lower bounds for broadcast tradeoffs  
involving the S3 dimension witnesses and the CHSH inequality in the  
three-preparation broadcasting scenario.

**Witnesses:**

- `S3_A = <A0>^[0] + <A1>^[0] + <A0>^[1] - <A1>^[1] - <A0>^[2]`  
- `CHSH_{BC|x} = <B0C0>^[x] + <B0C1>^[x] + <B1C0>^[x] - <B1C1>^[x]`

**Tradeoff targets:**

| Target | Horizontal | Vertical |
|--------|-----------|---------|
| `s3` | S3_C | S3_B |
| `chsh_s3_x0` | S3_C | CHSH_{BC\|x=0} |
| `chsh_s3_x1` | S3_C | CHSH_{BC\|x=1} |
| `chsh_s3_x2` | S3_C | CHSH_{BC\|x=2} |

**Upper bound:** lifted SDP at level `(q, t)` with sparse moment matrix,  
gauge fixing `|ψ_0⟩ = |0⟩`.

**Lower bound:** physical optimisation over isometry and local observables,  
with preparations analytically optimised via `λ_max(K_x)`.


---

## Dependencies

```bash
pip install numpy scipy cvxpy scs pandas matplotlib
```

---

## Usage

```bash
# Both bounds for s3 and all chsh_s3 targets
python broadcast_s3_chsh_tradeoffs.py --kind both --targets s3,chsh_s3_all --levels 1:1

# Upper bound at two SDP levels
python broadcast_s3_chsh_tradeoffs.py --kind upper --targets chsh_s3_all --levels 1:1,1:2

# Lower bound only
python broadcast_s3_chsh_tradeoffs.py --kind lower --targets s3,chsh_s3_x0,chsh_s3_x1,chsh_s3_x2

# Generate plots
python broadcast_s3_chsh_tradeoffs.py --kind plot --targets all --levels 1:1,1:2
```

---

## Full Source

```python
#!/usr/bin/env python3
"""
broadcast_s3_chsh_tradeoffs.py

Support-function lower and upper bounds for broadcast tradeoffs involving

    S3_B, S3_C, and CHSH_{BC|x}, x = 0,1,2.

The code is a direct adaptation of the QRAC support-function script, but now
uses three preparations x=0,1,2 and witness-type scores instead of QRAC
success probabilities.

Targets implemented
-------------------
1. s3
       horizontal score: S3_C
       vertical score:   S3_B
       support objective: S3_B + lambda S3_C

2. chsh_s3_x0, chsh_s3_x1, chsh_s3_x2
       horizontal score: S3_C
       vertical score:   CHSH_{BC|x}
       support objective: CHSH_{BC|x} + lambda S3_C

Conventions
-----------
S3_A = <A0>^[0] + <A1>^[0] + <A0>^[1] - <A1>^[1] - <A0>^[2].

CHSH_{BC|x} = <B0 C0>^[x] + <B0 C1>^[x]
            + <B1 C0>^[x] - <B1 C1>^[x].

Upper bound
-----------
Lifted SDP at level (q,t), with non-holomorphic scalar moments, gauge fixing
|psi_0> = |0>, and sparse moment matrix construction.

Lower bound
-----------
Physical optimization over a qubit-to-two-qubit isometry and local dichotomic
qubit observables. For fixed isometry and observables, the three preparations
are optimized analytically by lambda_max(K_x).

Install
-------
pip install numpy scipy cvxpy scs matplotlib

Examples
--------
python broadcast_s3_chsh_tradeoffs.py --kind both --targets s3,chsh_s3_all --levels 1:1
python broadcast_s3_chsh_tradeoffs.py --kind upper --targets chsh_s3_all --levels 1:1,1:2
python broadcast_s3_chsh_tradeoffs.py --kind lower --targets s3,chsh_s3_x0,chsh_s3_x1,chsh_s3_x2
python broadcast_s3_chsh_tradeoffs.py --kind plot --targets all --levels 1:1,1:2
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
from cvxpy.constraints.psd import PSD
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


# ============================================================
# Scenario data
# ============================================================

D = 2
N_X = 3
N_B = 2
N_C = 2
N_ALPHA = N_X * D

# S3 coefficients for x=0,1,2 and setting 0,1.
# S3 = E00 + E01 + E10 - E11 - E20.
S3_COEFF = np.array(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
    ],
    dtype=float,
)

CHSH_SIGNS = np.array(
    [
        [1.0, 1.0],
        [1.0, -1.0],
    ],
    dtype=float,
)

S3_CLASSICAL = 3.0
S3_QUBIT_MAX = 1.0 + 2.0 * np.sqrt(2.0)
CHSH_LOCAL = 2.0
CHSH_TSIRELSON = 2.0 * np.sqrt(2.0)


def var_id(x: int, i: int) -> int:
    return x * D + i


# Gauge fixing |psi_0> = |0> for the upper SDP.
REF_X = 0
REF_ONE = var_id(REF_X, 0)
REF_ZERO = var_id(REF_X, 1)


# ============================================================
# Scalar monomials in alpha and alpha^*, with gauge simplification
# ============================================================

Exp = Tuple[int, ...]
Scalar = Tuple[Exp, Exp]


def zero_exp() -> Exp:
    return tuple(0 for _ in range(N_ALPHA))


def unit_exp(k: int) -> Exp:
    e = [0] * N_ALPHA
    e[k] = 1
    return tuple(e)


def add_exp(a: Exp, b: Exp) -> Exp:
    return tuple(x + y for x, y in zip(a, b))


def zero_scalar() -> Scalar:
    z = zero_exp()
    return (z, z)


def scalar_a(k: int) -> Scalar:
    return (unit_exp(k), zero_exp())


def scalar_astar(k: int) -> Scalar:
    return (zero_exp(), unit_exp(k))


def adj_scalar(s: Scalar) -> Scalar:
    return (s[1], s[0])


def scalar_degree(s: Scalar) -> int:
    return sum(s[0]) + sum(s[1])


def simplify_scalar(s: Optional[Scalar]) -> Optional[Scalar]:
    if s is None:
        return None

    a = list(s[0])
    astar = list(s[1])

    # |psi_0> = |0>: alpha_{0,1}=0 and alpha_{0,0}=1.
    if a[REF_ZERO] > 0 or astar[REF_ZERO] > 0:
        return None

    a[REF_ONE] = 0
    astar[REF_ONE] = 0
    return (tuple(a), tuple(astar))


def mul_scalar(s1: Optional[Scalar], s2: Optional[Scalar]) -> Optional[Scalar]:
    if s1 is None or s2 is None:
        return None
    return simplify_scalar((add_exp(s1[0], s2[0]), add_exp(s1[1], s2[1])))


def generate_A(max_degree: int) -> List[Scalar]:
    if max_degree < 0:
        return []

    variables: List[Scalar] = []
    for x in range(N_X):
        for i in range(D):
            k = var_id(x, i)
            variables.append(scalar_a(k))
            variables.append(scalar_astar(k))

    out = {zero_scalar()}

    def rec(start: int, remaining: int, cur: Optional[Scalar]) -> None:
        if cur is None:
            return
        out.add(cur)
        if remaining == 0:
            return
        for r in range(start, len(variables)):
            rec(r, remaining - 1, mul_scalar(cur, variables[r]))

    rec(0, max_degree, zero_scalar())
    return sorted(out, key=lambda s: (scalar_degree(s), s))


def scalar_astar_a(x: int, bra_i: int, ket_j: int) -> Optional[Scalar]:
    return mul_scalar(scalar_astar(var_id(x, bra_i)), scalar_a(var_id(x, ket_j)))


# ============================================================
# Measurement words
# ============================================================

Word = Tuple[Tuple[int, ...], Tuple[int, ...]]
ID_WORD: Word = (tuple(), tuple())


def reduce_involutions(seq: Sequence[int]) -> Tuple[int, ...]:
    stack: List[int] = []
    for a in seq:
        if stack and stack[-1] == a:
            stack.pop()
        else:
            stack.append(a)
    return tuple(stack)


def reduce_word(w: Word) -> Word:
    b, c = w
    return (reduce_involutions(b), reduce_involutions(c))


def mul_word(w1: Word, w2: Word) -> Word:
    return reduce_word((w1[0] + w2[0], w1[1] + w2[1]))


def adj_word(w: Word) -> Word:
    return reduce_word((tuple(reversed(w[0])), tuple(reversed(w[1]))))


def word_length(w: Word) -> int:
    return len(w[0]) + len(w[1])


def B_word(y: int) -> Word:
    return ((y,), tuple())


def C_word(z: int) -> Word:
    return (tuple(), (z,))


def BC_word(y: int, z: int) -> Word:
    return ((y,), (z,))


def generate_S(max_len: int) -> List[Word]:
    out = {ID_WORD}
    for b_len in range(max_len + 1):
        for c_len in range(max_len + 1 - b_len):
            for b_seq in product(range(N_B), repeat=b_len):
                for c_seq in product(range(N_C), repeat=c_len):
                    out.add(reduce_word((b_seq, c_seq)))
    return sorted(out, key=lambda w: (word_length(w), w))


# ============================================================
# Sparse moment indexer and matrix
# ============================================================

MomentKey = Tuple[Scalar, int, int, Word]
Row = Tuple[int, Scalar, Word]


class MomentIndexer:
    def __init__(self):
        self.key_to_pos: Dict[MomentKey, Tuple[int, Optional[int], bool]] = {}
        self.pos_to_key: List[Tuple[MomentKey, str]] = []
        self.frozen = False

    @staticmethod
    def star_key(key: MomentKey) -> MomentKey:
        scalar, i, j, word = key
        ss = simplify_scalar(adj_scalar(scalar))
        if ss is None:
            raise RuntimeError("Unexpected zero scalar under adjoint.")
        return (ss, j, i, adj_word(word))

    @classmethod
    def canonicalize(cls, scalar: Scalar, i: int, j: int, word: Word):
        scalar = simplify_scalar(scalar)
        if scalar is None:
            return None, False, False
        raw = (scalar, i, j, reduce_word(word))
        star = cls.star_key(raw)
        if raw <= star:
            return raw, False, raw == star
        return star, True, False

    def handle(self, scalar: Optional[Scalar], i: int, j: int, word: Word):
        scalar = simplify_scalar(scalar)
        if scalar is None:
            return None

        key, conjugated, selfadjoint = self.canonicalize(scalar, i, j, word)
        if key is None:
            return None

        if key not in self.key_to_pos:
            if self.frozen:
                raise RuntimeError(f"New moment after freeze: {key}")
            re_pos = len(self.pos_to_key)
            self.pos_to_key.append((key, "re"))
            if selfadjoint:
                self.key_to_pos[key] = (re_pos, None, True)
            else:
                im_pos = len(self.pos_to_key)
                self.pos_to_key.append((key, "im"))
                self.key_to_pos[key] = (re_pos, im_pos, False)

        re_pos, im_pos, is_real = self.key_to_pos[key]
        return re_pos, im_pos, is_real, conjugated

    def expr(self, y, scalar: Optional[Scalar], i: int, j: int, word: Word):
        h = self.handle(scalar, i, j, word)
        if h is None:
            return 0
        re_pos, im_pos, is_real, conjugated = h
        if is_real:
            return y[re_pos]
        return y[re_pos] - 1j * y[im_pos] if conjugated else y[re_pos] + 1j * y[im_pos]


def build_rows(A: List[Scalar], S: List[Word]) -> List[Row]:
    return [(i, m, u) for i in range(D) for m in A for u in S]


def build_moment_matrix(rows: List[Row], idx: MomentIndexer):
    n = len(rows)
    data: List[complex] = []
    row_ind: List[int] = []
    col_ind: List[int] = []

    for a, (i, m, u) in enumerate(rows):
        for b, (j, nmon, v) in enumerate(rows):
            scalar = mul_scalar(adj_scalar(m), nmon)
            if scalar is None:
                continue
            word = mul_word(adj_word(u), v)
            h = idx.handle(scalar, i, j, word)
            if h is None:
                continue
            re_pos, im_pos, is_real, conjugated = h
            flat = a * n + b

            row_ind.append(flat)
            col_ind.append(re_pos)
            data.append(1.0 + 0.0j)

            if not is_real:
                row_ind.append(flat)
                col_ind.append(im_pos)
                data.append(-1.0j if conjugated else 1.0j)

    A_sparse = sp.coo_matrix(
        (np.array(data, dtype=complex), (row_ind, col_ind)),
        shape=(n * n, len(idx.pos_to_key)),
    ).tocsr()

    idx.frozen = True
    y = cp.Variable(len(idx.pos_to_key), name="moments")
    M_vec = A_sparse @ y
    M = cp.reshape(M_vec, (n, n), order="C")
    M = 0.5 * (M + M.H)
    return M, y


# ============================================================
# SDP constraints and score expressions
# ============================================================


def add_frame_constraints(constraints: list, idx: MomentIndexer, y) -> None:
    one = zero_scalar()
    for i in range(D):
        for j in range(D):
            constraints.append(idx.expr(y, one, i, j, ID_WORD) == (1.0 if i == j else 0.0))


def add_preparation_constraints(constraints: list, idx: MomentIndexer, y, A_qminus: List[Scalar], S: List[Word]) -> None:
    visible_words = sorted({mul_word(adj_word(u), v) for u in S for v in S}, key=lambda w: (word_length(w), w))

    for x in range(N_X):
        if x == REF_X:
            continue
        norm_terms = [mul_scalar(scalar_astar(var_id(x, k)), scalar_a(var_id(x, k))) for k in range(D)]

        for mu in A_qminus:
            for nu in A_qminus:
                scalar_base = mul_scalar(adj_scalar(mu), nu)
                for i in range(D):
                    for j in range(D):
                        for word in visible_words:
                            lhs = 0
                            for term in norm_terms:
                                lhs += idx.expr(y, mul_scalar(scalar_base, term), i, j, word)
                            lhs -= idx.expr(y, scalar_base, i, j, word)
                            constraints.append(lhs == 0)


def add_weighted_frame_constraints(constraints: list, idx: MomentIndexer, y) -> None:
    scalar_only = sorted({key[0] for key, _ in idx.pos_to_key if key[3] == ID_WORD}, key=str)
    for scalar in scalar_only:
        ref = idx.expr(y, scalar, 0, 0, ID_WORD)
        constraints.append(idx.expr(y, scalar, 0, 1, ID_WORD) == 0)
        constraints.append(idx.expr(y, scalar, 1, 0, ID_WORD) == 0)
        constraints.append(idx.expr(y, scalar, 1, 1, ID_WORD) == ref)


def E_B(idx: MomentIndexer, y, x: int, yy: int):
    out = 0
    for i in range(D):
        for j in range(D):
            out += idx.expr(y, scalar_astar_a(x, i, j), i, j, B_word(yy))
    return cp.real(out)


def E_C(idx: MomentIndexer, y, x: int, zz: int):
    out = 0
    for i in range(D):
        for j in range(D):
            out += idx.expr(y, scalar_astar_a(x, i, j), i, j, C_word(zz))
    return cp.real(out)


def E_BC(idx: MomentIndexer, y, x: int, yy: int, zz: int):
    out = 0
    for i in range(D):
        for j in range(D):
            out += idx.expr(y, scalar_astar_a(x, i, j), i, j, BC_word(yy, zz))
    return cp.real(out)


def s3_b_expr(idx: MomentIndexer, y):
    return sum(S3_COEFF[x, yy] * E_B(idx, y, x, yy) for x in range(N_X) for yy in range(N_B))


def s3_c_expr(idx: MomentIndexer, y):
    return sum(S3_COEFF[x, zz] * E_C(idx, y, x, zz) for x in range(N_X) for zz in range(N_C))


def chsh_expr(idx: MomentIndexer, y, x_chsh: int):
    return sum(CHSH_SIGNS[yy, zz] * E_BC(idx, y, x_chsh, yy, zz) for yy in range(N_B) for zz in range(N_C))


def target_expressions(target: str, idx: MomentIndexer, y):
    """
    Returns vertical_score, horizontal_score, vertical_name, horizontal_name.
    The support objective is vertical_score + lambda horizontal_score.
    """
    S3B = s3_b_expr(idx, y)
    S3C = s3_c_expr(idx, y)

    if target == "s3":
        return S3B, S3C, "S3_B", "S3_C"

    if target.startswith("chsh_s3_x"):
        x = int(target.replace("chsh_s3_x", ""))
        return chsh_expr(idx, y, x), S3C, f"CHSH_x{x}", "S3_C"

    raise ValueError(f"Unknown target: {target}")


# ============================================================
# Upper support SDP
# ============================================================


def build_upper_template(q: int, t: int, target: str):
    A_q = generate_A(q)
    A_qminus = generate_A(q - 1)
    S_t = generate_S(t)

    idx = MomentIndexer()
    rows = build_rows(A_q, S_t)
    M, y = build_moment_matrix(rows, idx)

    constraints = [PSD(M)]
    add_frame_constraints(constraints, idx, y)
    add_preparation_constraints(constraints, idx, y, A_qminus, S_t)
    add_weighted_frame_constraints(constraints, idx, y)

    vertical, horizontal, vertical_name, horizontal_name = target_expressions(target, idx, y)

    lam_param = cp.Parameter(nonneg=True, value=1.0, name="lambda")
    problem = cp.Problem(cp.Maximize(cp.real(vertical + lam_param * horizontal)), constraints)

    return {
        "problem": problem,
        "lambda": lam_param,
        "vertical": vertical,
        "horizontal": horizontal,
        "vertical_name": vertical_name,
        "horizontal_name": horizontal_name,
        "target": target,
        "q": q,
        "t": t,
        "rows": len(rows),
        "vars": len(idx.pos_to_key),
        "constraints": len(constraints),
        "A_size": len(A_q),
        "S_size": len(S_t),
    }


def solve_upper(template: dict, lam: float, eps: float, max_iters: int) -> dict:
    template["lambda"].value = float(lam)
    value = template["problem"].solve(
        solver=cp.SCS,
        eps=eps,
        max_iters=max_iters,
        normalize=True,
        warm_start=True,
    )

    raw_vertical = float(template["vertical"].value) if template["vertical"].value is not None else np.nan
    raw_horizontal = float(template["horizontal"].value) if template["horizontal"].value is not None else np.nan
    support = float(value) if value is not None else np.nan

    return {
        "kind": "upper",
        "target": template["target"],
        "q": template["q"],
        "t": template["t"],
        "lambda": float(lam),
        "support_upper": support,
        "raw_vertical_sdp": raw_vertical,
        "raw_horizontal_sdp": raw_horizontal,
        "vertical_name": template["vertical_name"],
        "horizontal_name": template["horizontal_name"],
        "check_support_raw": raw_vertical + float(lam) * raw_horizontal,
        "support_residual": support - (raw_vertical + float(lam) * raw_horizontal),
        "status": template["problem"].status,
        "rows": template["rows"],
        "vars": template["vars"],
        "constraints": template["constraints"],
        "A_size": template["A_size"],
        "S_size": template["S_size"],
    }


def run_upper(targets: Sequence[str], levels: Sequence[Tuple[int, int]], lambdas: Sequence[float], eps: float, max_iters: int, out_prefix: str) -> None:
    for target in targets:
        for q, t in levels:
            filename = f"{out_prefix}_{target}_upper_q{q}_t{t}.csv"
            template = build_upper_template(q, t, target)
            print(f"\nUpper SDP target={target}, level (q,t)=({q},{t})")
            print(
                f"  |A_q|={template['A_size']}, |S_t|={template['S_size']}, "
                f"rows={template['rows']}, real_vars={template['vars']}"
            )

            rows = []
            for k, lam in enumerate(lambdas, start=1):
                print(f"  lambda {k}/{len(lambdas)} = {lam:.10g}")
                row = solve_upper(template, float(lam), eps, max_iters)
                print(
                    f"    status={row['status']}, F={row['support_upper']:.10f}, "
                    f"raw=({row['raw_horizontal_sdp']:.10f},{row['raw_vertical_sdp']:.10f})"
                )
                rows.append(row)
                save_csv(filename, rows)


# ============================================================
# Lower physical support optimization
# ============================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID2 = np.eye(2, dtype=complex)


def complex_from_real(vec: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    return vec[:n].reshape(shape) + 1j * vec[n:2 * n].reshape(shape)


def qr_isometry(A: np.ndarray, ncols: int = 2) -> np.ndarray:
    Q, R = np.linalg.qr(A)
    Q = Q[:, :ncols]
    diag = np.diag(R[:ncols, :ncols])
    phases = np.ones_like(diag)
    mask = np.abs(diag) > 1e-12
    phases[mask] = diag[mask] / np.abs(diag[mask])
    return Q * phases.conj()[None, :]


def bloch_observable(theta: float, phi: float) -> np.ndarray:
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return nx * SIGMA_X + ny * SIGMA_Y + nz * SIGMA_Z


def unpack_lower_params(theta: np.ndarray):
    rawU = complex_from_real(theta[:16], (4, 2))
    U = qr_isometry(rawU, 2)
    pos = 16

    Bobs = []
    for _ in range(N_B):
        Bobs.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    Charlies = []
    for _ in range(N_C):
        Charlies.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    return U, Bobs, Charlies


def lower_coefficients(target: str, lam: float):
    """
    Coefficients of the support objective
        vertical + lambda horizontal.

    The returned arrays satisfy
        objective = sum_x <psi_x| U^dagger O_x U |psi_x>,
    where O_x is assembled from coeff_B, coeff_C, coeff_BC.
    """
    coeff_B = np.zeros((N_X, N_B), dtype=float)
    coeff_C = np.zeros((N_X, N_C), dtype=float)
    coeff_BC = np.zeros((N_X, N_B, N_C), dtype=float)

    if target == "s3":
        # vertical=S3_B, horizontal=S3_C.
        coeff_B += S3_COEFF
        coeff_C += lam * S3_COEFF
        return coeff_B, coeff_C, coeff_BC, "S3_B", "S3_C"

    if target.startswith("chsh_s3_x"):
        x_chsh = int(target.replace("chsh_s3_x", ""))
        # vertical=CHSH_x, horizontal=S3_C.
        coeff_C += lam * S3_COEFF
        coeff_BC[x_chsh, :, :] += CHSH_SIGNS
        return coeff_B, coeff_C, coeff_BC, f"CHSH_x{x_chsh}", "S3_C"

    raise ValueError(f"Unknown target: {target}")


def lower_all_scores(theta: np.ndarray):
    U, Bobs, Charlies = unpack_lower_params(theta)

    # Use no target objective here. This function assumes the caller also passes
    # the optimized preparations through their density vectors. For simplicity,
    # the score reconstruction is done in reconstruct_lower_point below, because
    # the optimal preparations depend on lambda and target.
    raise NotImplementedError


def kx_operator(U, Bobs, Charlies, coeff_B, coeff_C, coeff_BC, x: int) -> np.ndarray:
    O = np.zeros((4, 4), dtype=complex)

    for y in range(N_B):
        O += coeff_B[x, y] * np.kron(Bobs[y], ID2)

    for z in range(N_C):
        O += coeff_C[x, z] * np.kron(ID2, Charlies[z])

    for y in range(N_B):
        for z in range(N_C):
            O += coeff_BC[x, y, z] * np.kron(Bobs[y], Charlies[z])

    K = U.conj().T @ O @ U
    return 0.5 * (K + K.conj().T)


def lower_support_value(theta: np.ndarray, target: str, lam: float) -> float:
    U, Bobs, Charlies = unpack_lower_params(theta)
    coeff_B, coeff_C, coeff_BC, _, _ = lower_coefficients(target, lam)

    value = 0.0
    for x in range(N_X):
        K = kx_operator(U, Bobs, Charlies, coeff_B, coeff_C, coeff_BC, x)
        value += np.max(np.linalg.eigvalsh(K))
    return float(np.real(value))


def reconstruct_lower_point(theta: np.ndarray, target: str, lam: float):
    U, Bobs, Charlies = unpack_lower_params(theta)
    coeff_B, coeff_C, coeff_BC, vertical_name, horizontal_name = lower_coefficients(target, lam)

    # Preparations optimal for the target support objective.
    psis = []
    for x in range(N_X):
        K = kx_operator(U, Bobs, Charlies, coeff_B, coeff_C, coeff_BC, x)
        vals, vecs = np.linalg.eigh(K)
        psis.append(vecs[:, np.argmax(vals)])

    E_B_val = np.zeros((N_X, N_B), dtype=float)
    E_C_val = np.zeros((N_X, N_C), dtype=float)
    E_BC_val = np.zeros((N_X, N_B, N_C), dtype=float)

    for x in range(N_X):
        phi = U @ psis[x]
        for y in range(N_B):
            E_B_val[x, y] = float(np.real(np.vdot(phi, np.kron(Bobs[y], ID2) @ phi)))
        for z in range(N_C):
            E_C_val[x, z] = float(np.real(np.vdot(phi, np.kron(ID2, Charlies[z]) @ phi)))
        for y in range(N_B):
            for z in range(N_C):
                E_BC_val[x, y, z] = float(np.real(np.vdot(phi, np.kron(Bobs[y], Charlies[z]) @ phi)))

    S3B = float(np.sum(S3_COEFF * E_B_val))
    S3C = float(np.sum(S3_COEFF * E_C_val))
    CHSH = [float(np.sum(CHSH_SIGNS * E_BC_val[x])) for x in range(N_X)]

    if target == "s3":
        vertical = S3B
        horizontal = S3C
    else:
        x_chsh = int(target.replace("chsh_s3_x", ""))
        vertical = CHSH[x_chsh]
        horizontal = S3C

    return {
        "vertical_lower": vertical,
        "horizontal_lower": horizontal,
        "vertical_name": vertical_name,
        "horizontal_name": horizontal_name,
        "S3_B": S3B,
        "S3_C": S3C,
        "CHSH_x0": CHSH[0],
        "CHSH_x1": CHSH[1],
        "CHSH_x2": CHSH[2],
    }


def lower_starts(previous: Optional[np.ndarray], rng: np.random.Generator, n_random: int) -> List[np.ndarray]:
    starts: List[np.ndarray] = []
    if previous is not None:
        starts.append(previous.copy())
        starts.append(previous + 0.03 * rng.normal(size=24))
    starts.append(0.3 * rng.normal(size=24))
    for _ in range(n_random):
        starts.append(rng.normal(size=24))
    return starts


def optimize_lower(target: str, lam: float, previous: Optional[np.ndarray], restarts: int, maxiter: int, seed: int):
    rng = np.random.default_rng(seed)
    best_theta = None
    best_val = -np.inf
    best_status = "none"
    scale = 1.0 + abs(lam)

    def objective(th):
        return -lower_support_value(th, target, lam) / scale

    for start in lower_starts(previous, rng, restarts):
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8},
        )
        val = lower_support_value(res.x, target, lam)
        if val > best_val:
            best_val = val
            best_theta = res.x.copy()
            best_status = str(res.message)

    point_data = reconstruct_lower_point(best_theta, target, lam)

    return {
        "kind": "lower",
        "target": target,
        "lambda": float(lam),
        "support_lower": best_val,
        "optimizer_status": best_status,
        **point_data,
        "theta": best_theta,
    }


def run_lower(targets: Sequence[str], lambdas: Sequence[float], restarts: int, maxiter: int, seed: int, out_prefix: str) -> None:
    for target in targets:
        filename = f"{out_prefix}_{target}_lower.csv"
        rows = []
        previous = None
        print(f"\nLower optimization target={target}")

        for k, lam in enumerate(lambdas, start=1):
            print(f"  lambda {k}/{len(lambdas)} = {lam:.10g}")
            row = optimize_lower(target, float(lam), previous, restarts, maxiter, seed + 1000 * k)
            previous = row.pop("theta")
            print(
                f"    F={row['support_lower']:.10f}, "
                f"point=({row['horizontal_lower']:.10f},{row['vertical_lower']:.10f})"
            )
            rows.append(row)
            save_csv(filename, rows)


# ============================================================
# Support reconstruction and plotting from CSV files
# ============================================================


def parse_csv_value(v):
    if v == "":
        return None
    if v == "nan":
        return np.nan
    if v == "inf":
        return np.inf
    if v == "-inf":
        return -np.inf
    try:
        return float(v)
    except ValueError:
        return v


def load_csv(filename: str) -> List[dict]:
    path = Path(filename)
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: parse_csv_value(v) for k, v in row.items()})
    return rows


def upper_points_from_support(rows: List[dict]) -> List[dict]:
    rows = [
        r for r in rows
        if r.get("lambda") is not None
        and r.get("support_upper") is not None
        and np.isfinite(r["lambda"])
        and np.isfinite(r["support_upper"])
    ]
    rows = sorted(rows, key=lambda r: r["lambda"])

    pts = []
    for r1, r2 in zip(rows[:-1], rows[1:]):
        lam1 = float(r1["lambda"])
        lam2 = float(r2["lambda"])
        F1 = float(r1["support_upper"])
        F2 = float(r2["support_upper"])
        if abs(lam2 - lam1) < 1e-12:
            continue
        horizontal = (F2 - F1) / (lam2 - lam1)
        vertical = F1 - lam1 * horizontal
        if np.isfinite(horizontal) and np.isfinite(vertical):
            pts.append({"horizontal_upper": horizontal, "vertical_upper": vertical})
    return pts


def target_axis_labels(target: str) -> Tuple[str, str]:
    if target == "s3":
        return r"$S_3^{(C)}$", r"$S_3^{(B)}$"
    if target.startswith("chsh_s3_x"):
        x = target.replace("chsh_s3_x", "")
        return r"$S_3^{(C)}$", rf"$\mathrm{{CHSH}}_{{BC}}^{{[{x}]}}$"
    return "horizontal", "vertical"


def plot_target(target: str, levels: Sequence[Tuple[int, int]], out_prefix: str, filename: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.6))

    # Classical rectangle and quantum reference maxima.
    if target == "s3":
        x_classical = S3_CLASSICAL
        y_classical = S3_CLASSICAL
        x_quantum = S3_QUBIT_MAX
        y_quantum = S3_QUBIT_MAX
    else:
        x_classical = S3_CLASSICAL
        y_classical = CHSH_LOCAL
        x_quantum = S3_QUBIT_MAX
        y_quantum = CHSH_TSIRELSON

    ax.fill_between([0.0, x_classical], [0.0, 0.0], [y_classical, y_classical], alpha=0.20, label="classical region")
    ax.axvline(x_quantum, linestyle=":", linewidth=1.2, label="single-score quantum bound")
    ax.axhline(y_quantum, linestyle=":", linewidth=1.2)

    # Upper bounds.
    for q, t in levels:
        upper_file = f"{out_prefix}_{target}_upper_q{q}_t{t}.csv"
        upper_rows = load_csv(upper_file)
        upper_pts = upper_points_from_support(upper_rows)
        upper_pts = sorted(upper_pts, key=lambda p: p["horizontal_upper"])
        if upper_pts:
            ax.plot(
                [p["horizontal_upper"] for p in upper_pts],
                [p["vertical_upper"] for p in upper_pts],
                "o-",
                markersize=3,
                linewidth=1.6,
                label=rf"SDP upper $(q,t)=({q},{t})$",
            )

    # Lower points.
    lower_file = f"{out_prefix}_{target}_lower.csv"
    lower_rows = load_csv(lower_file)
    lower_rows = [
        r for r in lower_rows
        if r.get("horizontal_lower") is not None
        and r.get("vertical_lower") is not None
        and np.isfinite(r["horizontal_lower"])
        and np.isfinite(r["vertical_lower"])
    ]
    lower_rows = sorted(lower_rows, key=lambda r: r["horizontal_lower"])
    if lower_rows:
        ax.scatter(
            [r["horizontal_lower"] for r in lower_rows],
            [r["vertical_lower"] for r in lower_rows],
            s=20,
            label="lower-bound points",
            zorder=5,
        )

    xlabel, ylabel = target_axis_labels(target)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, x_quantum * 1.04)
    ax.set_ylim(0.0, y_quantum * 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    if filename is None:
        filename = f"{out_prefix}_{target}_plot.pdf"
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close(fig)


# ============================================================
# CSV and CLI
# ============================================================


def csv_value(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return None


def save_csv(filename: str, rows: List[dict]) -> None:
    clean_rows = []
    for r in rows:
        clean = {k: csv_value(v) for k, v in r.items() if csv_value(v) is not None}
        clean_rows.append(clean)

    keys: List[str] = []
    for r in clean_rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    with Path(filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean_rows)


def expand_targets(s: str) -> List[str]:
    out: List[str] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "all":
            out.extend(["s3", "chsh_s3_x0", "chsh_s3_x1", "chsh_s3_x2"])
        elif item == "chsh_s3_all":
            out.extend(["chsh_s3_x0", "chsh_s3_x1", "chsh_s3_x2"])
        else:
            out.append(item)
    # keep order, remove duplicates
    dedup: List[str] = []
    for item in out:
        if item not in dedup:
            dedup.append(item)
    return dedup


def parse_levels(s: str) -> List[Tuple[int, int]]:
    levels = []
    for item in s.split(","):
        if not item.strip():
            continue
        q, t = item.split(":")
        levels.append((int(q), int(t)))
    return levels


def parse_lambdas(s: Optional[str]) -> np.ndarray:
    if s is not None and s.strip():
        return np.array([float(x) for x in s.split(",") if x.strip()], dtype=float)
    return np.concatenate([
        np.linspace(0.0, 1.0, 21),
        np.linspace(1.1, 5.0, 20),
        np.array([7.0, 10.0, 15.0, 25.0, 40.0, 100.0]),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["upper", "lower", "both", "plot"], default="both")
    p.add_argument("--targets", default="all", help="s3,chsh_s3_x0,chsh_s3_x1,chsh_s3_x2,chsh_s3_all,all")
    p.add_argument("--levels", default="1:1", help="Upper levels, e.g. 1:1 or 1:1,1:2")
    p.add_argument("--lambdas", default=None, help="Comma-separated lambdas.")
    p.add_argument("--out-prefix", default="broadcast_tradeoff")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=30000)
    p.add_argument("--lower-restarts", type=int, default=1)
    p.add_argument("--lower-maxiter", type=int, default=900)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    targets = expand_targets(args.targets)
    levels = parse_levels(args.levels)
    lambdas = parse_lambdas(args.lambdas)

    if args.kind in ("upper", "both"):
        run_upper(targets, levels, lambdas, args.eps, args.max_iters, args.out_prefix)

    if args.kind in ("lower", "both"):
        run_lower(targets, lambdas, args.lower_restarts, args.lower_maxiter, args.seed, args.out_prefix)

    if args.kind == "plot":
        for target in targets:
            plot_target(target, levels, args.out_prefix)


if __name__ == "__main__":
    main()

```
