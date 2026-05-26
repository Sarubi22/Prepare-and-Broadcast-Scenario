# Upper and Lower Bounds for PAB Witnesses CC1, CC2, CNS3, CNS4

Computes upper and lower bounds for the four linear broadcast witnesses  
studied in the paper:

| Witness | Classical bound | Description |
|---------|----------------|-------------|
| `CC1`   | 8              | Two-body, 3 preparations, fully classical |
| `CC2`   | 6              | Two-body, 3 preparations, fully classical |
| `CNS3`  | 4              | Two-body, 3 preparations, classical–NS |
| `CNS4`  | 5              | Two-body, 4 preparations, classical–NS |

**Upper bound:** lifted SDP relaxation at level `(q, t)` — sparse moment matrix  
with non-holomorphic scalar preparation monomials and frame moments.

**Lower bound:** physical optimisation over a qubit-to-two-qubit isometry and  
local qubit observables `B_0, B_1, C_0, C_1`. For fixed channel and  
measurements, each preparation is optimised analytically via `λ_max(K_x)`.

Outputs: `witness_bounds_upper.csv` and `witness_bounds_lower.csv`.


---

## Dependencies

```bash
pip install numpy scipy cvxpy scs pandas matplotlib
```

---

## Usage

```bash
# Both bounds for all witnesses at level (1,1)
python broadcast_witness_bounds.py --kind both --witnesses all --levels 1:1

# Upper bound at levels (1,1) and (1,2)
python broadcast_witness_bounds.py --kind upper --witnesses all --levels 1:1,1:2

# Lower bound with 50 random restarts
python broadcast_witness_bounds.py --kind lower --witnesses all --lower-restarts 50

# Specific witnesses
python broadcast_witness_bounds.py --kind both --witnesses CC1,CC2,CNS3,CNS4 --levels 1:1
```

---

## Full Source

```python
#!/usr/bin/env python3
"""
broadcast_witness_bounds.py

Upper and lower bounds for the four linear broadcast witnesses:

    CC1, CC2, CNS3, CNS4.

Upper bound:
    lifted SDP relaxation at level (q,t),

        max_L W(L),

    using the same sparse direct non-holomorphic moment construction used in
    the QRAC support-function code.

Lower bound:
    direct physical optimization over

        U : C^2 -> C^2 tensor C^2,
        B_0, B_1, C_0, C_1,

    with each preparation optimized analytically.  For fixed U,B,C and a
    witness W = sum_x <psi_x| K_x |psi_x>, the best preparation is the
    eigenvector of K_x with largest eigenvalue.  Thus the lower objective is

        sum_x lambda_max(K_x).

Install:
    pip install numpy scipy cvxpy scs

Examples:
    python broadcast_witness_bounds.py --kind both --witnesses all --levels 1:1
    python broadcast_witness_bounds.py --kind upper --witnesses all --levels 1:1,1:2
    python broadcast_witness_bounds.py --kind lower --witnesses all --lower-restarts 50
    python broadcast_witness_bounds.py --kind both --witnesses CC1,CC2,CNS3,CNS4 --levels 1:1 --lower-restarts 50

Outputs:
    witness_bounds_upper.csv
    witness_bounds_lower.csv
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
# Witness data
# ============================================================

# Terms have format (coefficient, x, y, z), meaning
#     coefficient * <B_y C_z>^[x].
WITNESSES = {
    "CC1": {
        "label": "W_CC^(1)",
        "n_x": 3,
        "classical_bound": 8.0,
        "terms": [
            (+3.0, 0, 0, 0),
            (+1.0, 0, 0, 1),
            (+1.0, 0, 1, 0),
            (-1.0, 0, 1, 1),
            (-2.0, 1, 0, 1),
            (-2.0, 1, 1, 0),
            (-3.0, 2, 0, 0),
            (+1.0, 2, 0, 1),
            (+1.0, 2, 1, 0),
            (+1.0, 2, 1, 1),
        ],
    },
    "CC2": {
        "label": "W_CC^(2)",
        "n_x": 3,
        "classical_bound": 6.0,
        "terms": [
            (+2.0, 0, 0, 0),
            (+2.0, 0, 1, 1),
            (+2.0, 1, 0, 1),
            (-2.0, 1, 1, 0),
            (-1.0, 2, 0, 0),
            (-1.0, 2, 0, 1),
            (+1.0, 2, 1, 0),
            (-1.0, 2, 1, 1),
        ],
    },
    "CNS3": {
        "label": "W_CNS^(3)",
        "n_x": 3,
        "classical_bound": 4.0,
        "terms": [
            (+1.0, 0, 0, 0),
            (-1.0, 0, 0, 1),
            (+1.0, 1, 0, 1),
            (-1.0, 1, 1, 1),
            (-1.0, 2, 0, 0),
            (+1.0, 2, 1, 1),
        ],
    },
    "CNS4": {
        "label": "W_CNS^(4)",
        "n_x": 4,
        "classical_bound": 5.0,
        "terms": [
            (+1.0, 0, 0, 1),
            (-1.0, 0, 1, 0),
            (-1.0, 1, 0, 1),
            (+1.0, 2, 0, 1),
            (+1.0, 2, 1, 0),
            (+1.0, 2, 1, 1),
            (+1.0, 3, 0, 1),
            (+1.0, 3, 1, 0),
            (-1.0, 3, 1, 1),
        ],
    },
}


def canonical_witness_name(name: str) -> str:
    key = name.strip().upper()
    aliases = {
        "W1": "CC1",
        "W2": "CC2",
        "WCC1": "CC1",
        "WCC2": "CC2",
        "CC_1": "CC1",
        "CC_2": "CC2",
        "W3": "CNS3",
        "W4": "CNS4",
        "W5": "CNS4",
        "CNS": "CNS3",
        "CNS_3": "CNS3",
        "CNS_4": "CNS4",
        "CNS_5": "CNS4",
        "WCNS3": "CNS3",
        "WCNS4": "CNS4",
        "WCNS5": "CNS4",
    }
    key = aliases.get(key, key)
    if key not in WITNESSES:
        raise ValueError(f"Unknown witness {name!r}. Use CC1, CC2, CNS3, CNS4, or all.")
    return key


# ============================================================
# Global dimensions set per witness
# ============================================================

D = 2
N_B = 2
N_C = 2

N_X: Optional[int] = None
N_ALPHA: Optional[int] = None

# Gauge fixing |psi_0> = |0>
REF_X = 0
REF_ONE: Optional[int] = None
REF_ZERO: Optional[int] = None

Exp = Tuple[int, ...]
Scalar = Tuple[Exp, Exp]
Word = Tuple[Tuple[int, ...], Tuple[int, ...]]
MomentKey = Tuple[Scalar, int, int, Word]
Row = Tuple[int, Scalar, Word]

ID_WORD: Word = (tuple(), tuple())


def set_problem_size(n_x: int) -> None:
    global N_X, N_ALPHA, REF_ONE, REF_ZERO
    N_X = int(n_x)
    N_ALPHA = N_X * D
    REF_ONE = 0
    REF_ZERO = 1


# ============================================================
# Scalar monomials in alpha and alpha^*
# ============================================================

def var_id(x: int, i: int) -> int:
    return x * D + i


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
    """
    Gauge fixing |psi_0> = |0>:
        alpha_{0,0}=1, alpha_{0,1}=0,
    and similarly for conjugates.
    """
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
    """
    Measurement algebra:
        B_y^2 = I,
        C_z^2 = I,
        [B_y, C_z] = 0.

    We commute Bob letters with Charlie letters by storing two subwords.
    We do not commute B0 with B1 and do not commute C0 with C1.
    """
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
# Sparse moment indexer and moment matrix
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
            constraints.append(idx.expr(y, one, i, j, ID_WORD) == (1.0 if i == j else 0.0))


def add_preparation_constraints(constraints: list, idx: MomentIndexer, y, A_qminus: List[Scalar], S: List[Word]) -> None:
    """
    Preparation normalization:
        L(mu^*nu g_x eta^{ij}_{u,v}) = 0
    for mu,nu in A_{q-1} and word pairs generated by S_t.

    The gauge-fixed preparation x=0 is skipped.
    """
    visible_words = sorted(
        {mul_word(adj_word(u), v) for u in S for v in S},
        key=lambda w: (word_length(w), w),
    )

    for x in range(N_X):
        if x == REF_X:
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
# Witness objective in the SDP
# ============================================================

def correlator_BC(idx: MomentIndexer, y, x: int, yy: int, zz: int):
    out = 0
    word = BC_word(yy, zz)

    for i in range(D):
        for j in range(D):
            out += idx.expr(
                y,
                scalar_astar_a(x, i, j),
                i,
                j,
                word,
            )

    return cp.real(out)


def witness_expr(idx: MomentIndexer, y, terms):
    out = 0
    for coeff, x, yy, zz in terms:
        out += coeff * correlator_BC(idx, y, x, yy, zz)
    return out


# ============================================================
# Upper SDP
# ============================================================

def build_upper_template(witness_name: str, q: int, t: int):
    wname = canonical_witness_name(witness_name)
    data = WITNESSES[wname]

    set_problem_size(data["n_x"])

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

    W = witness_expr(idx, y, data["terms"])
    problem = cp.Problem(cp.Maximize(cp.real(W)), constraints)

    return {
        "problem": problem,
        "W": W,
        "witness": wname,
        "label": data["label"],
        "n_x": data["n_x"],
        "classical_bound": data["classical_bound"],
        "q": q,
        "t": t,
        "rows": len(rows),
        "real_vars": len(idx.pos_to_key),
        "constraints": len(constraints),
        "A_size": len(A_q),
        "S_size": len(S_t),
        "num_terms": len(data["terms"]),
    }


def solve_upper(template: dict, eps: float, max_iters: int) -> dict:
    value = template["problem"].solve(
        solver=cp.SCS,
        eps=eps,
        max_iters=max_iters,
        normalize=True,
        warm_start=True,
    )

    upper = float(value) if value is not None else np.nan

    return {
        "kind": "upper",
        "witness": template["witness"],
        "label": template["label"],
        "n_preparations": template["n_x"],
        "q": template["q"],
        "t": template["t"],
        "status": template["problem"].status,
        "upper_bound": upper,
        "classical_bound": template["classical_bound"],
        "violation_upper_minus_classical": upper - template["classical_bound"],
        "rows": template["rows"],
        "real_vars": template["real_vars"],
        "constraints": template["constraints"],
        "A_size": template["A_size"],
        "S_size": template["S_size"],
        "num_terms": template["num_terms"],
        "eps": eps,
        "max_iters": max_iters,
    }


def run_upper(witnesses: Sequence[str], levels: Sequence[Tuple[int, int]], eps: float, max_iters: int, out_file: str) -> List[dict]:
    rows: List[dict] = []

    for witness in witnesses:
        for q, t in levels:
            print(f"\nUpper SDP: {witness} at level (q,t)=({q},{t})")

            template = build_upper_template(witness, q, t)
            print(
                f"  n_x={template['n_x']}, |A_q|={template['A_size']}, |S_t|={template['S_size']}, "
                f"rows={template['rows']}, real_vars={template['real_vars']}, constraints={template['constraints']}"
            )

            row = solve_upper(template, eps, max_iters)

            print(
                f"  status={row['status']}, upper={row['upper_bound']:.10f}, "
                f"classical={row['classical_bound']:.10f}, violation={row['violation_upper_minus_classical']:.10f}"
            )

            rows.append(row)
            save_csv(out_file, rows)

    return rows


# ============================================================
# Lower physical optimization
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
    # raw 4x2 complex isometry = 16 real parameters;
    # B0,B1,C0,C1 each have two Bloch angles = 8.
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


def witness_operator_for_x(terms, U, Bobs, Charlies, x: int) -> np.ndarray:
    O = np.zeros((4, 4), dtype=complex)

    for coeff, xx, yy, zz in terms:
        if xx == x:
            O += coeff * np.kron(Bobs[yy], Charlies[zz])

    K = U.conj().T @ O @ U
    return 0.5 * (K + K.conj().T)


def lower_value(theta: np.ndarray, terms, n_x: int) -> float:
    U, Bobs, Charlies = unpack_lower_params(theta)

    val = 0.0
    for x in range(n_x):
        Kx = witness_operator_for_x(terms, U, Bobs, Charlies, x)
        val += np.max(np.linalg.eigvalsh(Kx))

    return float(np.real(val))


def reconstruct_lower_point(theta: np.ndarray, terms, n_x: int):
    U, Bobs, Charlies = unpack_lower_params(theta)

    total = 0.0
    x_values = []

    for x in range(n_x):
        Kx = witness_operator_for_x(terms, U, Bobs, Charlies, x)
        vals, vecs = np.linalg.eigh(Kx)
        idx = int(np.argmax(vals))
        total += float(vals[idx])
        x_values.append(float(vals[idx]))

    return total, x_values


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


def optimize_lower_for_witness(
    witness_name: str,
    restarts: int,
    maxiter: int,
    seed: int,
) -> dict:
    wname = canonical_witness_name(witness_name)
    data = WITNESSES[wname]
    terms = data["terms"]
    n_x = data["n_x"]

    rng = np.random.default_rng(seed)

    best_theta = None
    best_val = -np.inf
    best_status = "none"

    def objective(theta):
        return -lower_value(theta, terms, n_x)

    # One pass with independent starts.  The previous argument is kept only to
    # reuse the start generation utility.
    for start in lower_starts(None, rng, restarts):
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8},
        )

        val = lower_value(res.x, terms, n_x)
        if val > best_val:
            best_val = val
            best_theta = res.x.copy()
            best_status = str(res.message)

    total, x_values = reconstruct_lower_point(best_theta, terms, n_x)

    row = {
        "kind": "lower",
        "witness": wname,
        "label": data["label"],
        "n_preparations": n_x,
        "status": best_status,
        "lower_bound": total,
        "classical_bound": data["classical_bound"],
        "violation_lower_minus_classical": total - data["classical_bound"],
        "restarts": restarts,
        "maxiter": maxiter,
        "seed": seed,
    }

    for x, value in enumerate(x_values):
        row[f"x{x}_lambda_max"] = value

    return row


def run_lower(witnesses: Sequence[str], restarts: int, maxiter: int, seed: int, out_file: str) -> List[dict]:
    rows: List[dict] = []

    for k, witness in enumerate(witnesses):
        print(f"\nLower optimization: {witness}")

        row = optimize_lower_for_witness(
            witness,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed + 1000 * k,
        )

        print(
            f"  lower={row['lower_bound']:.10f}, classical={row['classical_bound']:.10f}, "
            f"violation={row['violation_lower_minus_classical']:.10f}"
        )

        rows.append(row)
        save_csv(out_file, rows)

    return rows


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


def parse_witnesses(s: str) -> List[str]:
    if s.lower().strip() == "all":
        return ["CC1", "CC2", "CNS3", "CNS4"]

    out = []
    for item in s.split(","):
        item = item.strip()
        if item:
            out.append(canonical_witness_name(item))

    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["upper", "lower", "both"], default="both")
    parser.add_argument("--witnesses", default="all", help="CC1,CC2,CNS3,CNS4, or all.")
    parser.add_argument("--levels", default="1:1", help="Upper levels q:t, e.g. 1:1 or 1:1,1:2.")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--lower-restarts", type=int, default=20)
    parser.add_argument("--lower-maxiter", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-prefix", default="witness_bounds")
    return parser.parse_args()


def main():
    args = parse_args()

    witnesses = parse_witnesses(args.witnesses)
    levels = parse_levels(args.levels)

    upper_file = f"{args.out_prefix}_upper.csv"
    lower_file = f"{args.out_prefix}_lower.csv"

    if args.kind in ("upper", "both"):
        run_upper(
            witnesses=witnesses,
            levels=levels,
            eps=args.eps,
            max_iters=args.max_iters,
            out_file=upper_file,
        )

    if args.kind in ("lower", "both"):
        run_lower(
            witnesses=witnesses,
            restarts=args.lower_restarts,
            maxiter=args.lower_maxiter,
            seed=args.seed,
            out_file=lower_file,
        )


if __name__ == "__main__":
    main()

```
