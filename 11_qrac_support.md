# Broadcast QRAC Tradeoff — Support-Function Bounds (2→1)

Computes upper and lower bounds for the tradeoff curve of the 2→1 broadcast  
QRAC in the four-preparation broadcasting scenario.

For a fixed weight `λ ≥ 0`, the **support function** `P_B + λ P_C` is maximised,  
where `P_B` and `P_C` are Bob's and Charlie's marginal QRAC success probabilities.  
Sweeping `λ` traces the boundary of the achievable `(P_C, P_B)` region.

**Upper bound:** lifted SDP relaxation at level `(q, t)` — scalar preparation  
monomials multiplied by frame moments `<e_i|u†v|e_j>` (same construction  
as the QQ broadcasting relaxation in the paper).

**Lower bound:** physical optimisation over a qubit-to-two-qubit isometry and  
local qubit observables. For fixed channel and measurements, the four  
preparations are optimised analytically via `λ_max(K_x)`.

Outputs: CSV files `qrac_upper.csv` and `qrac_lower.csv`.


---

## Dependencies

```bash
pip install numpy scipy cvxpy scs pandas matplotlib
```

---

## Usage

```bash
# Upper bound at levels (1,1) and (1,2)
python qrac_support.py --kind upper --levels 1:1,1:2

# Lower bound
python qrac_support.py --kind lower

# Both bounds
python qrac_support.py --kind both --levels 1:1,1:2

# Custom lambda grid
python qrac_support.py --kind upper --levels 1:2 --lambdas 0,0.25,0.5,1,2,5
```

---

## Full Source

```python
#!/usr/bin/env python3
"""
qrac_support_minimal.py

Minimal support-function code for the 2 -> 1 broadcast QRAC tradeoff.

Upper bound:
    lifted SDP at level (q,t), with objective
        max_L P_B(L) + lambda P_C(L).

Lower bound:
    physical support-function optimization
        max_{U,B,C} P_B + lambda P_C,
    with the preparations optimized analytically by lambda_max(K_x).

The script only saves CSV files. No plotting code is included.

Install:
    pip install numpy scipy cvxpy scs

Examples:
    python qrac_support_minimal.py --kind upper --levels 1:1,1:2
    python qrac_support_minimal.py --kind lower
    python qrac_support_minimal.py --kind both --levels 1:1,1:2
    python qrac_support_minimal.py --kind upper --levels 1:2 --lambdas 0,0.25,0.5,1,2,5
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
# Basic QRAC data
# ============================================================

X_BITS = [(0, 0), (0, 1), (1, 0), (1, 1)]
D = 2
N_X = 4
N_B = 2
N_C = 2
N_ALPHA = N_X * D


def bit_sign(bit: int) -> int:
    return 1 if bit == 0 else -1


def var_id(x: int, i: int) -> int:
    return x * D + i


def qrac_radius() -> float:
    return 1.0 / (2.0 * np.sqrt(2.0))


def reference_point(lam: float) -> Tuple[float, float]:
    radius = qrac_radius()
    den = np.sqrt(1.0 + lam * lam)
    return 0.5 + radius / den, 0.5 + radius * lam / den


def reference_support(lam: float) -> float:
    PB, PC = reference_point(lam)
    return PB + lam * PC


# Gauge fixing |psi_00> = |0>
REF_X = 0
REF_ONE = var_id(REF_X, 0)
REF_ZERO = var_id(REF_X, 1)


# ============================================================
# Scalar monomials in alpha and alpha^*, with gauge simplification
# ============================================================

Exp = Tuple[int, ...]
Scalar = Tuple[Exp, Exp]  # (alpha powers, alpha_star powers)


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

    # |psi_00> = |0>: alpha_{0,1}=0 and alpha_{0,0}=1.
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
    return mul_scalar(scalar_astar(var_id(x, bra_i)), scalar_a(var_id(x, ket_j)))


# ============================================================
# Measurement words
# ============================================================

Word = Tuple[Tuple[int, ...], Tuple[int, ...]]  # (Bob word, Charlie word)
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
# SDP constraints and scores
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


def qrac_scores(idx: MomentIndexer, y):
    PB = 0
    PC = 0
    for x, bits in enumerate(X_BITS):
        for yy in range(N_B):
            PB += 0.5 * (1 + bit_sign(bits[yy]) * E_B(idx, y, x, yy))
        for zz in range(N_C):
            PC += 0.5 * (1 + bit_sign(bits[zz]) * E_C(idx, y, x, zz))
    return PB / (N_X * N_B), PC / (N_X * N_C)


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

    PB, PC = qrac_scores(idx, y)
    lam_param = cp.Parameter(nonneg=True, value=1.0, name="lambda")
    problem = cp.Problem(cp.Maximize(cp.real(PB + lam_param * PC)), constraints)

    return {
        "problem": problem,
        "lambda": lam_param,
        "PB": PB,
        "PC": PC,
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

    raw_PB = float(template["PB"].value) if template["PB"].value is not None else np.nan
    raw_PC = float(template["PC"].value) if template["PC"].value is not None else np.nan
    support = float(value) if value is not None else np.nan
    ref = reference_support(lam)

    return {
        "kind": "upper",
        "q": template["q"],
        "t": template["t"],
        "lambda": float(lam),
        "support_upper": support,
        "raw_PB_sdp": raw_PB,
        "raw_PC_sdp": raw_PC,
        "check_support_raw": raw_PB + float(lam) * raw_PC,
        "support_residual": support - (raw_PB + float(lam) * raw_PC),
        "support_ref": ref,
        "support_gap_ref": support - ref,
        "status": template["problem"].status,
        "rows": template["rows"],
        "vars": template["vars"],
        "constraints": template["constraints"],
        "A_size": template["A_size"],
        "S_size": template["S_size"],
    }


def run_upper(levels: Sequence[Tuple[int, int]], lambdas: Sequence[float], eps: float, max_iters: int, out_prefix: str) -> None:
    for q, t in levels:
        filename = f"{out_prefix}_upper_q{q}_t{t}.csv"
        template = build_upper_template(q, t)
        print(f"\nUpper SDP level (q,t)=({q},{t})")
        print(f"  |A_q|={template['A_size']}, |S_t|={template['S_size']}, rows={template['rows']}, real_vars={template['vars']}")

        rows = []
        for k, lam in enumerate(lambdas, start=1):
            print(f"  lambda {k}/{len(lambdas)} = {lam:.10g}")
            row = solve_upper(template, float(lam), eps, max_iters)
            print(f"    status={row['status']}, F={row['support_upper']:.10f}, raw=({row['raw_PB_sdp']:.10f},{row['raw_PC_sdp']:.10f})")
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
    for _ in range(2):
        Bobs.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    Charlies = []
    for _ in range(2):
        Charlies.append(bloch_observable(theta[pos], theta[pos + 1]))
        pos += 2

    return U, Bobs, Charlies


def kx_operator(U, Bobs, Charlies, lam: float, x: int) -> np.ndarray:
    bits = X_BITS[x]
    OB = sum(bit_sign(bits[y]) * np.kron(Bobs[y], ID2) for y in range(N_B))
    OC = sum(bit_sign(bits[z]) * np.kron(ID2, Charlies[z]) for z in range(N_C))
    K = U.conj().T @ (OB + lam * OC) @ U
    return 0.5 * (K + K.conj().T)


def lower_support_value(theta: np.ndarray, lam: float) -> float:
    U, Bobs, Charlies = unpack_lower_params(theta)
    val = 0.5 * (1.0 + lam)
    for x in range(N_X):
        val += np.max(np.linalg.eigvalsh(kx_operator(U, Bobs, Charlies, lam, x))) / 16.0
    return float(np.real(val))


def reconstruct_lower_point(theta: np.ndarray, lam: float):
    U, Bobs, Charlies = unpack_lower_params(theta)
    PB_bias = 0.0
    PC_bias = 0.0

    for x, bits in enumerate(X_BITS):
        vals, vecs = np.linalg.eigh(kx_operator(U, Bobs, Charlies, lam, x))
        psi = vecs[:, np.argmax(vals)]
        phi = U @ psi

        for y in range(N_B):
            PB_bias += bit_sign(bits[y]) * float(np.real(np.vdot(phi, np.kron(Bobs[y], ID2) @ phi)))
        for z in range(N_C):
            PC_bias += bit_sign(bits[z]) * float(np.real(np.vdot(phi, np.kron(ID2, Charlies[z]) @ phi)))

    return 0.5 + PB_bias / 16.0, 0.5 + PC_bias / 16.0


def lower_starts(previous: Optional[np.ndarray], rng: np.random.Generator, n_random: int) -> List[np.ndarray]:
    starts: List[np.ndarray] = []
    if previous is not None:
        starts.append(previous.copy())
        starts.append(previous + 0.03 * rng.normal(size=24))
    starts.append(0.3 * rng.normal(size=24))
    for _ in range(n_random):
        starts.append(rng.normal(size=24))
    return starts


def optimize_lower(lam: float, previous: Optional[np.ndarray], restarts: int, maxiter: int, seed: int):
    rng = np.random.default_rng(seed)
    best_theta = None
    best_val = -np.inf
    best_status = "none"
    scale = 1.0 + abs(lam)

    def objective(th):
        return -lower_support_value(th, lam) / scale

    for start in lower_starts(previous, rng, restarts):
        res = minimize(objective, start, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8})
        val = lower_support_value(res.x, lam)
        if val > best_val:
            best_val = val
            best_theta = res.x.copy()
            best_status = str(res.message)

    PB, PC = reconstruct_lower_point(best_theta, lam)
    PB_ref, PC_ref = reference_point(lam)
    ref = reference_support(lam)

    return {
        "kind": "lower",
        "lambda": float(lam),
        "support_lower": best_val,
        "PB_lower": PB,
        "PC_lower": PC,
        "PB_ref": PB_ref,
        "PC_ref": PC_ref,
        "support_ref": ref,
        "support_gap_ref": best_val - ref,
        "radial_error": (PB - 0.5) ** 2 + (PC - 0.5) ** 2 - 1.0 / 8.0,
        "optimizer_status": best_status,
        "theta": best_theta,
    }


def run_lower(lambdas: Sequence[float], restarts: int, maxiter: int, seed: int, out_prefix: str) -> None:
    filename = f"{out_prefix}_lower.csv"
    rows = []
    previous = None

    for k, lam in enumerate(lambdas, start=1):
        print(f"  lambda {k}/{len(lambdas)} = {lam:.10g}")
        row = optimize_lower(float(lam), previous, restarts, maxiter, seed + 1000 * k)
        previous = row.pop("theta")
        print(f"    F={row['support_lower']:.10f}, point=({row['PB_lower']:.10f},{row['PC_lower']:.10f})")
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
        np.array([7.0, 10.0, 15.0, 25.0, 40.0, 200.0]),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["upper", "lower", "both"], default="both")
    p.add_argument("--levels", default="1:1", help="Upper levels, e.g. 1:1 or 1:1,1:2")
    p.add_argument("--lambdas", default=None, help="Comma-separated lambdas. Default is a moderate grid.")
    p.add_argument("--out-prefix", default="qrac_support")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=30000)
    p.add_argument("--lower-restarts", type=int, default=1)
    p.add_argument("--lower-maxiter", type=int, default=900)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    lambdas = parse_lambdas(args.lambdas)

    if args.kind in ("upper", "both"):
        run_upper(parse_levels(args.levels), lambdas, args.eps, args.max_iters, args.out_prefix)

    if args.kind in ("lower", "both"):
        run_lower(lambdas, args.lower_restarts, args.lower_maxiter, args.seed, args.out_prefix)


if __name__ == "__main__":
    main()

```
