# PAB Classical Polytopes — Two-Body and Marginal Vertices (4 Preparations)

Generates two classical PAB polytopes simultaneously for the scenario
`|X|=4, |Y|=2, |Z|=2, |M|=2`, with binary outputs:

1. **Two-body polytope (16D):** for each preparation `x`, keeps the four
   two-body correlators `[<B0C0>, <B0C1>, <B1C0>, <B1C1>]`.

2. **Marginal polytope (16D):** for each preparation `x`, keeps the four
   single-party correlators `[<B0>, <B1>, <C0>, <C1>]`.

Both polytopes are built by enumerating all deterministic strategies:
- `f : X → M` — Alice's encoding (16 functions)
- `hB : Y×M → B` — Bob's response (16 functions)
- `hC : Z×M → C` — Charlie's response (16 functions)

Duplicate vertices are removed and outputs are saved as `lrs` V-representation
files (`.ext`) and CSV tables.

**Run `lrs` on the `.ext` files to obtain the facet inequalities.**

---

## Dependencies

```bash
pip install numpy pandas
```

`lrs` must be installed separately for facet enumeration:
[cgm.cs.mcgill.ca/~avis/C/lrs.html](http://cgm.cs.mcgill.ca/~avis/C/lrs.html)

---

## Outputs

| File | Description |
|------|-------------|
| `pab_classical_X4_twobody_vertices.csv` | Two-body vertices as CSV |
| `pab_classical_X4_twobody_vertices.ext` | Two-body V-representation for `lrs` |
| `pab_classical_X4_marginal_vertices.csv` | Marginal vertices as CSV |
| `pab_classical_X4_marginal_vertices.ext` | Marginal V-representation for `lrs` |
| `pab_classical_X4_deterministic_strategies.csv` | All strategies with vertex assignments |

## Facet enumeration (after running the script)

```bash
lrs pab_classical_X4_twobody_vertices.ext > pab_classical_X4_twobody_facets.ine
lrs pab_classical_X4_marginal_vertices.ext > pab_classical_X4_marginal_facets.ine
```

---

## Full Source

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

# ============================================================
# PAB classical polytopes with 4 preparations
# Generates separately:
#   1) two-body polytope
#   2) marginal polytope
# ============================================================

X_SIZE = 4
Y_SIZE = 2
Z_SIZE = 2
M_SIZE = 2
B_SIZE = 2
C_SIZE = 2

OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Deterministic response functions
# ============================================================
#
# Classical model:
#
#   m  = f(x)
#   b  = hB(y, m)
#   c  = hC(z, m)
#
# with:
#
#   f  : X -> M
#   hB : Y x M -> B
#   hC : Z x M -> C
#
# ============================================================

def hB_value(hB, y, m):
    """
    hB : Y x M -> B

    Index order:
        hB[0] = hB(y=0, m=0)
        hB[1] = hB(y=0, m=1)
        hB[2] = hB(y=1, m=0)
        hB[3] = hB(y=1, m=1)
    """
    return hB[y * M_SIZE + m]


def hC_value(hC, z, m):
    """
    hC : Z x M -> C

    Index order:
        hC[0] = hC(z=0, m=0)
        hC[1] = hC(z=0, m=1)
        hC[2] = hC(z=1, m=0)
        hC[3] = hC(z=1, m=1)
    """
    return hC[z * M_SIZE + m]


# ============================================================
# Coordinate names
# ============================================================

TWOBODY_COORD_NAMES = [
    f"B{y}C{z}_x{x}"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

TWOBODY_COORD_PRETTY = [
    rf"<B_{y}C_{z}>^[x={x}]"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

MARGINAL_COORD_NAMES = []
for x in range(X_SIZE):
    MARGINAL_COORD_NAMES.extend([
        f"B0_x{x}", f"B1_x{x}", f"C0_x{x}", f"C1_x{x}",
    ])

MARGINAL_COORD_PRETTY = []
for x in range(X_SIZE):
    MARGINAL_COORD_PRETTY.extend([
        rf"<B_0>^[x={x}]", rf"<B_1>^[x={x}]",
        rf"<C_0>^[x={x}]", rf"<C_1>^[x={x}]",
    ])


# ============================================================
# Two-body vertex
# ============================================================

def twobody_vertex(f, hB, hC):
    """
    Builds a two-body vertex.

    For each preparation x:
        m = f(x)

    For each pair (y, z):
        <B_y C_z>^[x] = (-1)^{b+c}

    where:
        b = hB(y, m)
        c = hC(z, m)

    Coordinate order per preparation x:
        [<B0C0>, <B0C1>, <B1C0>, <B1C1>]
    """
    v = []
    for x in range(X_SIZE):
        m = f[x]
        for y in range(Y_SIZE):
            b_sign = (-1) ** hB_value(hB, y, m)
            for z in range(Z_SIZE):
                c_sign = (-1) ** hC_value(hC, z, m)
                v.append(b_sign * c_sign)
    return tuple(v)


# ============================================================
# Marginal vertex
# ============================================================

def marginal_vertex(f, hB, hC):
    """
    Builds a marginal vertex.

    For each preparation x:
        m = f(x)

        <B_y>^[x] = (-1)^b,   b = hB(y, m)
        <C_z>^[x] = (-1)^c,   c = hC(z, m)

    Coordinate order per preparation x:
        [<B0>, <B1>, <C0>, <C1>]
    """
    v = []
    for x in range(X_SIZE):
        m = f[x]
        for y in range(Y_SIZE):
            v.append((-1) ** hB_value(hB, y, m))
        for z in range(Z_SIZE):
            v.append((-1) ** hC_value(hC, z, m))
    return tuple(v)


# ============================================================
# Save V-representation for lrs
# ============================================================

def save_lrs_ext(vertices, filename):
    """
    Saves vertices in lrs V-representation format.

    Each line has the form:
        1 v_0 v_1 ... v_{d-1}

    The leading 1 denotes an affine point.
    """
    filename = Path(filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{len(vertices)} {vertices.shape[1] + 1} integer\n")
        for v in vertices:
            f.write("1 " + " ".join(str(int(x)) for x in v) + "\n")
        f.write("end\n")


# ============================================================
# Enumerate all deterministic strategies
# ============================================================

def generate_polytopes():
    """
    Enumerates all deterministic strategies:

        f  : X -> M
        hB : Y x M -> B
        hC : Z x M -> C

    and generates both the two-body and marginal vertex sets.
    """
    all_f  = list(product(range(M_SIZE), repeat=X_SIZE))
    all_hB = list(product(range(B_SIZE), repeat=Y_SIZE * M_SIZE))
    all_hC = list(product(range(C_SIZE), repeat=Z_SIZE * M_SIZE))

    total_raw = len(all_f) * len(all_hB) * len(all_hC)

    print("=" * 100)
    print("Deterministic strategy enumeration")
    print("=" * 100)
    print(f"f  : X -> M       = {len(all_f)}")
    print(f"hB : Y x M -> B   = {len(all_hB)}")
    print(f"hC : Z x M -> C   = {len(all_hC)}")
    print(f"Total (raw)       = {total_raw}")
    print()

    twobody_set  = set()
    marginal_set = set()
    strategies   = []

    for f in all_f:
        for hB in all_hB:
            for hC in all_hC:
                v2 = twobody_vertex(f, hB, hC)
                vm = marginal_vertex(f, hB, hC)
                twobody_set.add(v2)
                marginal_set.add(vm)
                strategies.append({
                    "f_x_to_m":    f,
                    "hB_y_m_to_b": hB,
                    "hC_z_m_to_c": hC,
                    "twobody_vertex":  v2,
                    "marginal_vertex": vm,
                })

    twobody_vertices  = np.array(sorted(twobody_set),  dtype=int)
    marginal_vertices = np.array(sorted(marginal_set), dtype=int)

    return twobody_vertices, marginal_vertices, strategies


# ============================================================
# Main
# ============================================================

print("=" * 100)
print("Two-body coordinate order")
print("=" * 100)
for i, name in enumerate(TWOBODY_COORD_PRETTY):
    print(f"{i:2d}: {name}")

print()
print("=" * 100)
print("Marginal coordinate order")
print("=" * 100)
for i, name in enumerate(MARGINAL_COORD_PRETTY):
    print(f"{i:2d}: {name}")

twobody_vertices, marginal_vertices, strategies = generate_polytopes()

# --- Two-body summary ---
print()
print("=" * 100)
print("Classical TWO-BODY polytope — 4 preparations")
print("=" * 100)
print(f"Ambient dimension : {twobody_vertices.shape[1]}")
print(f"Unique vertices   : {len(twobody_vertices)}")
print(f"Affine dimension  : {np.linalg.matrix_rank(twobody_vertices - twobody_vertices[0])}")

df_twobody = pd.DataFrame(twobody_vertices, columns=TWOBODY_COORD_NAMES)
df_twobody.insert(0, "vertex_id", range(1, len(df_twobody) + 1))

twobody_csv = OUT_DIR / "pab_classical_X4_twobody_vertices.csv"
twobody_ext = OUT_DIR / "pab_classical_X4_twobody_vertices.ext"

df_twobody.to_csv(twobody_csv, index=False)
save_lrs_ext(twobody_vertices, twobody_ext)

print(f"\nTwo-body files saved:\n  {twobody_csv}\n  {twobody_ext}")

# --- Marginal summary ---
print()
print("=" * 100)
print("Classical MARGINAL polytope — 4 preparations")
print("=" * 100)
print(f"Ambient dimension : {marginal_vertices.shape[1]}")
print(f"Unique vertices   : {len(marginal_vertices)}")
print(f"Affine dimension  : {np.linalg.matrix_rank(marginal_vertices - marginal_vertices[0])}")

df_marginal = pd.DataFrame(marginal_vertices, columns=MARGINAL_COORD_NAMES)
df_marginal.insert(0, "vertex_id", range(1, len(df_marginal) + 1))

marginal_csv = OUT_DIR / "pab_classical_X4_marginal_vertices.csv"
marginal_ext = OUT_DIR / "pab_classical_X4_marginal_vertices.ext"

df_marginal.to_csv(marginal_csv, index=False)
save_lrs_ext(marginal_vertices, marginal_ext)

print(f"\nMarginal files saved:\n  {marginal_csv}\n  {marginal_ext}")

# --- Deterministic strategies ---
twobody_to_id  = {tuple(v): i+1 for i, v in enumerate(twobody_vertices)}
marginal_to_id = {tuple(v): i+1 for i, v in enumerate(marginal_vertices)}

for s in strategies:
    s["twobody_vertex_id"]  = twobody_to_id[s["twobody_vertex"]]
    s["marginal_vertex_id"] = marginal_to_id[s["marginal_vertex"]]

df_strategies = pd.DataFrame(strategies)[[
    "twobody_vertex_id", "marginal_vertex_id",
    "f_x_to_m", "hB_y_m_to_b", "hC_z_m_to_c",
    "twobody_vertex", "marginal_vertex",
]]

strategies_csv = OUT_DIR / "pab_classical_X4_deterministic_strategies.csv"
df_strategies.to_csv(strategies_csv, index=False)

print()
print("=" * 100)
print("Deterministic strategies")
print("=" * 100)
print(f"File saved:\n  {strategies_csv}")

# --- lrs commands ---
print()
print("=" * 100)
print("Commands to generate facets with lrs")
print("=" * 100)
print(f"lrs {twobody_ext.name} > pab_classical_X4_twobody_facets.ine")
print(f"lrs {marginal_ext.name} > pab_classical_X4_marginal_facets.ine")
```
