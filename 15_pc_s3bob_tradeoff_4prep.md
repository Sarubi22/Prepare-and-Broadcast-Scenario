# Tradeoff: S3_B vs Charlie's QRAC Score P_C (4 Preparations)

Computes support-function upper and lower bounds for the tradeoff between  
Bob's S3 dimension witness and Charlie's marginal 2→1 QRAC success  
probability, in the four-preparation broadcasting scenario.

**Scores:**

- `S3_B = <B0>^[0] + <B1>^[0] + <B0>^[1] - <B1>^[1] - <B0>^[2]`
- `P_C = 1/2 + (1/16) Σ_{x,z} (−1)^{x_z} <Cz>^[x]`

The support objective is `S3_B + λ P_C`. Sweeping `λ ≥ 0` traces the  
boundary of the achievable `(P_C, S3_B)` region.

**Upper bound:** lifted SDP relaxation at level `(q, t)` — same sparse  
non-holomorphic moment construction used throughout the paper.

**Lower bound:** physical optimisation over a qubit-to-two-qubit isometry  
and local qubit observables. For fixed channel and measurements, the four  
preparations are optimised analytically via `λ_max(K_x)`.


---

## Dependencies

```bash
pip install numpy scipy cvxpy scs pandas matplotlib
```

---

## Usage

```bash
# Both bounds at level (1,1)
python pc_s3bob_tradeoff_4prep.py --kind both --levels 1:1

# Upper bound at multiple SDP levels
python pc_s3bob_tradeoff_4prep.py --kind upper --levels 1:1,1:2

# Lower bound with 20 restarts
python pc_s3bob_tradeoff_4prep.py --kind lower --lower-restarts 20

# Custom lambda grid, both bounds
python pc_s3bob_tradeoff_4prep.py --kind both --levels 1:1,1:2 --lambdas 0,0.1,0.25,0.5,1,2,5,10,25
```

---

## Full Source

```python
#!/usr/bin/env python3
"""
pc_s3bob_tradeoff_4prep.py

Support-function tradeoff between Charlie's 2->1 QRAC success probability P_C
and Bob's S_3 witness in the four-preparation broadcast scenario.

Preparations:
    x = 0,1,2,3 <-> 00,01,10,11.

Scores:
    S3_B = <B0>^[0] + <B1>^[0] + <B0>^[1]
           - <B1>^[1] - <B0>^[2],

    P_C  = 1/2 + (1/16) sum_{x,z} (-1)^{x_z} <C_z>^[x].

Upper bound:
    lifted SDP relaxation at level (q,t),

        max_L S3_B(L) + lambda P_C(L).

Lower bound:
    direct physical support-function optimization over an isometry
    U : C^2 -> C^2 tensor C^2 and qubit observables B_0,B_1,C_0,C_1.
    For fixed U,B,C, the four preparations are optimized analytically
    by taking the maximum eigenvector of K_x(lambda).

Install:
    pip install numpy scipy cvxpy scs

Examples:
    python pc_s3bob_tradeoff_4prep.py --kind both --levels 1:1
    python pc_s3bob_tradeoff_4prep.py --kind upper --levels 1:1,1:2
    python pc_s3bob_tradeoff_4prep.py --kind lower --lower-restarts 20
    python pc_s3bob_tradeoff_4prep.py --kind both --levels 1:1,1:2 --lambdas 0,0.1,0.25,0.5,1,2,5,10,25
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
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


# ============================================================
# Basic data
# ============================================================

D = 2
N_X = 4
N_B = 2
N_C = 2
N_ALPHA = N_X * D

X_BITS: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]

REF_ONE = 0  # alpha_{0,0}
REF_ZERO = 1 # alpha_{0,1}

S3_B_TERMS = [
    (+1.0, 0, 0),
    (+1.0, 0, 1),
    (+1.0, 1, 0),
    (-1.0, 1, 1),
    (-1.0, 2, 0),
]

Exp = Tuple[int, ...]
Scalar = Tuple[Exp, Exp]
Word = Tuple[Tuple[int, ...], Tuple[int, ...]]
MomentKey = Tuple[Scalar, int, int, Word]
Row = Tuple[int, Scalar, Word]

ID_WORD: Word = (tuple(), tuple())


def bit_sign(bit: int) -> int:
    return 1 if bit == 0 else -1


def var_id(x: int, i: int) -> int:
    return x * D + i


# ============================================================
# Scalar monomials with gauge |psi_0> = |0>
# ============================================================

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

    variables = []
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
    return mul_scalar(
        scalar_astar(var_id(x, bra_i)),
        scalar_a(var_id(x, ket_j)),
    )


# ============================================================
# Measurement words
# ============================================================

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


def generate_S(max_len: int) -> List[Word]:
    out = {ID_WORD}
    for b_len in range(max_len + 1):
        for c_len in range(max_len + 1 - b_len):
            for b_seq in product(range(N_B), repeat=b_len):
                for c_seq in product(range(N_C), repeat=c_len):
                    out.add(reduce_word((b_seq, c_seq)))
    return sorted(out, key=lambda w: (word_length(w), w))


# ============================================================
# Sparse moment indexer
# ============================================================

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
# SDP constraints
# ============================================================

def add_frame_constraints(constraints: list, idx: MomentIndexer, y) -> None:
    one = zero_scalar()
    for i in range(D):
        for j in range(D):
            constraints.append(
                idx.expr(y, one, i, j, ID_WORD) == (1.0 if i == j else 0.0)
            )


def add_preparation_constraints(
    constraints: list,
    idx: MomentIndexer,
    y,
    A_qminus: List[Scalar],
    S: List[Word],
) -> None:
    visible_words = sorted(
        {mul_word(adj_word(u), v) for u in S for v in S},
        key=lambda w: (word_length(w), w),
    )

    for x in range(N_X):
        if x == 0:
            continue

        norm_terms = [
            mul_scalar(scalar_astar(var_id(x, k)), scalar_a(var_id(x, k)))
            for k in range(D)
        ]

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
    scalar_only = sorted(
        {key[0] for key, _ in idx.pos_to_key if key[3] == ID_WORD},
        key=str,
    )

    for scalar in scalar_only:
        ref = idx.expr(y, scalar, 0, 0, ID_WORD)
        constraints.append(idx.expr(y, scalar, 0, 1, ID_WORD) == 0)
        constraints.append(idx.expr(y, scalar, 1, 0, ID_WORD) == 0)
        constraints.append(idx.expr(y, scalar, 1, 1, ID_WORD) == ref)


# ============================================================
# SDP scores
# ============================================================

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


def S3_B_expr(idx: MomentIndexer, y):
    return sum(coeff * E_B(idx, y, x, yy) for coeff, x, yy in S3_B_TERMS)


def P_C_expr(idx: MomentIndexer, y):
    corr_sum = 0
    for x, bits in enumerate(X_BITS):
        for zz in range(N_C):
            corr_sum += bit_sign(bits[zz]) * E_C(idx, y, x, zz)
    return 0.5 + corr_sum / 16.0


# ============================================================
# Upper support SDP
# ============================================================

def build_upper_template(q: int, t: int):
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

    S3B = S3_B_expr(idx, y)
    PC = P_C_expr(idx, y)

    lam_param = cp.Parameter(nonneg=True, value=1.0, name="lambda")
    problem = cp.Problem(cp.Maximize(cp.real(S3B + lam_param * PC)), constraints)

    return {
        "problem": problem,
        "lambda": lam_param,
        "S3B": S3B,
        "PC": PC,
        "q": q,
        "t": t,
        "rows": len(rows),
        "real_vars": len(idx.pos_to_key),
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

    S3B = float(template["S3B"].value) if template["S3B"].value is not None else np.nan
    PC = float(template["PC"].value) if template["PC"].value is not None else np.nan
    support = float(value) if value is not None else np.nan

    return {
        "kind": "upper",
        "q": template["q"],
        "t": template["t"],
        "lambda": float(lam),
        "support_upper": support,
        "raw_S3B_sdp": S3B,
        "raw_PC_sdp": PC,
        "check_support_raw": S3B + float(lam) * PC,
        "support_residual": support - (S3B + float(lam) * PC),
        "status": template["problem"].status,
        "rows": template["rows"],
        "real_vars": template["real_vars"],
        "constraints": template["constraints"],
        "A_size": template["A_size"],
        "S_size": template["S_size"],
        "eps": eps,
        "max_iters": max_iters,
    }


def run_upper(levels: Sequence[Tuple[int, int]], lambdas: Sequence[float], eps: float, max_iters: int, out_prefix: str) -> None:
    for q, t in levels:
        filename = f"{out_prefix}_upper_q{q}_t{t}.csv"
        template = build_upper_template(q, t)

        print(f"\nUpper SDP level (q,t)=({q},{t})")
        print(
            f"  |A_q|={template['A_size']}, |S_t|={template['S_size']}, "
            f"rows={template['rows']}, real_vars={template['real_vars']}, constraints={template['constraints']}"
        )

        rows = []
        for k, lam in enumerate(lambdas, start=1):
            print(f"  lambda {k}/{len(lambdas)} = {lam:.10g}")
            row = solve_upper(template, float(lam), eps, max_iters)
            print(
                f"    status={row['status']}, F={row['support_upper']:.10f}, "
                f"raw=(S3B={row['raw_S3B_sdp']:.10f}, PC={row['raw_PC_sdp']:.10f})"
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


def lower_param_count() -> int:
    return 16 + 8


def unpack_lower_params(theta: np.ndarray):
    rawU = complex_from_real(theta[:16], (4, 2))
    U = qr_isometry(rawU, 2)

    pos = 16

    Bobs = []
    for _ in range(2):
        Bobs.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    Charlies = []
    for _ in range(2):
        Charlies.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    return U, Bobs, Charlies


def lower_operator_for_x(U, Bobs, Charlies, x: int, lam: float) -> np.ndarray:
    O = np.zeros((4, 4), dtype=complex)

    for coeff, xx, yy in S3_B_TERMS:
        if xx == x:
            O += coeff * np.kron(Bobs[yy], ID2)

    bits = X_BITS[x]
    for zz in range(N_C):
        O += (lam / 16.0) * bit_sign(bits[zz]) * np.kron(ID2, Charlies[zz])

    K = U.conj().T @ O @ U
    return 0.5 * (K + K.conj().T)


def lower_support_value(theta: np.ndarray, lam: float) -> float:
    U, Bobs, Charlies = unpack_lower_params(theta)

    val = 0.5 * lam
    for x in range(N_X):
        Kx = lower_operator_for_x(U, Bobs, Charlies, x, lam)
        val += np.max(np.linalg.eigvalsh(Kx))

    return float(np.real(val))


def reconstruct_lower_point(theta: np.ndarray, lam: float) -> Tuple[float, float]:
    U, Bobs, Charlies = unpack_lower_params(theta)

    psis = []
    for x in range(N_X):
        Kx = lower_operator_for_x(U, Bobs, Charlies, x, lam)
        vals, vecs = np.linalg.eigh(Kx)
        psis.append(vecs[:, np.argmax(vals)])

    S3B = 0.0
    for coeff, x, yy in S3_B_TERMS:
        phi = U @ psis[x]
        S3B += coeff * float(np.real(np.vdot(phi, np.kron(Bobs[yy], ID2) @ phi)))

    corr_sum = 0.0
    for x, bits in enumerate(X_BITS):
        phi = U @ psis[x]
        for zz in range(N_C):
            corr_sum += bit_sign(bits[zz]) * float(np.real(np.vdot(phi, np.kron(ID2, Charlies[zz]) @ phi)))

    PC = 0.5 + corr_sum / 16.0
    return S3B, PC


def lower_starts(previous: Optional[np.ndarray], rng: np.random.Generator, n_random: int) -> List[np.ndarray]:
    npar = lower_param_count()
    starts: List[np.ndarray] = []

    if previous is not None:
        starts.append(previous.copy())
        starts.append(previous + 0.03 * rng.normal(size=npar))
        starts.append(previous + 0.10 * rng.normal(size=npar))

    starts.append(0.3 * rng.normal(size=npar))

    for _ in range(n_random):
        starts.append(rng.normal(size=npar))

    return starts


def optimize_lower_for_lambda(
    lam: float,
    previous: Optional[np.ndarray],
    restarts: int,
    maxiter: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    best_theta = None
    best_val = -np.inf
    best_status = "none"
    scale = 1.0 + abs(lam)

    def objective(theta):
        return -lower_support_value(theta, lam) / scale

    for start in lower_starts(previous, rng, restarts):
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8},
        )

        val = lower_support_value(res.x, lam)
        if val > best_val:
            best_val = val
            best_theta = res.x.copy()
            best_status = str(res.message)

    S3B, PC = reconstruct_lower_point(best_theta, lam)

    return {
        "kind": "lower",
        "lambda": float(lam),
        "support_lower": best_val,
        "S3B_lower": S3B,
        "PC_lower": PC,
        "check_support_lower": S3B + float(lam) * PC,
        "support_residual": best_val - (S3B + float(lam) * PC),
        "optimizer_status": best_status,
        "restarts": restarts,
        "maxiter": maxiter,
        "seed": seed,
        "theta": best_theta,
    }


def run_lower(lambdas: Sequence[float], restarts: int, maxiter: int, seed: int, out_prefix: str) -> None:
    filename = f"{out_prefix}_lower.csv"

    rows = []
    previous = None

    for k, lam in enumerate(lambdas, start=1):
        print(f"\nLower support {k}/{len(lambdas)}: lambda={lam:.10g}")

        row = optimize_lower_for_lambda(
            float(lam),
            previous=previous,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed + 1000 * k,
        )
        previous = row.pop("theta")

        print(
            f"  F={row['support_lower']:.10f}, "
            f"point=(S3B={row['S3B_lower']:.10f}, PC={row['PC_lower']:.10f})"
        )

        rows.append(row)
        save_csv(filename, rows)


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
        clean = {}
        for k, v in r.items():
            vv = csv_value(v)
            if vv is not None:
                clean[k] = vv
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


def parse_levels(s: str) -> List[Tuple[int, int]]:
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Levels must be written as q:t, e.g. 1:1,1:2")
        q_str, t_str = item.split(":", 1)
        out.append((int(q_str), int(t_str)))
    return out


def parse_lambdas(s: Optional[str]) -> np.ndarray:
    if s is not None and s.strip():
        return np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)

    return np.concatenate(
        [
            np.linspace(0.0, 1.0, 21),
            np.linspace(1.1, 5.0, 20),
            np.array([7.0, 10.0, 15.0, 25.0, 40.0, 100.0]),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["upper", "lower", "both"], default="both")
    parser.add_argument("--levels", default="1:1", help="Upper levels, e.g. 1:1 or 1:1,1:2")
    parser.add_argument("--lambdas", default=None, help="Comma-separated lambdas.")
    parser.add_argument("--out-prefix", default="pc_s3bob_tradeoff")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--lower-restarts", type=int, default=5)
    parser.add_argument("--lower-maxiter", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()

    lambdas = parse_lambdas(args.lambdas)
    levels = parse_levels(args.levels)

    if args.kind in ("upper", "both"):
        run_upper(
            levels=levels,
            lambdas=lambdas,
            eps=args.eps,
            max_iters=args.max_iters,
            out_prefix=args.out_prefix,
        )

    if args.kind in ("lower", "both"):
        run_lower(
            lambdas=lambdas,
            restarts=args.lower_restarts,
            maxiter=args.lower_maxiter,
            seed=args.seed,
            out_prefix=args.out_prefix,
        )


if __name__ == "__main__":
    main()

```
