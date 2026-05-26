# PAB Classical Marginal Polytope — 4 Preparations (16D)

Generates the marginal vertex set of the fully classical PAB polytope  
for `|X|=4` preparations and `|M|=2`.

For each preparation `x`, the 4-dimensional block  
`[<B0>, <B1>, <C0>, <C1>]` is computed from deterministic strategies, giving  
a 16-dimensional ambient vector.

Outputs saved:
- CSV of unique vertices (`pab_marginal_classical_X4_vertices.csv`)
- CSV of all deterministic strategies with vertex assignments
- `lrs` V-representation file (`.ext`) for facet enumeration

Run `lrs pab_marginal_classical_X4_vertices.ext` to obtain the facets.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product

# ============================================================
# PAB marginal todo clássico com 4 preparações
# ============================================================

X_SIZE = 4      # quatro preparações
Y_SIZE = 2
Z_SIZE = 2
M_SIZE = 2      # mensagem clássica binária
B_SIZE = 2
C_SIZE = 2

BLOCK = Y_SIZE + Z_SIZE
DIM = X_SIZE * BLOCK


# ============================================================
# Ordem das coordenadas marginais
# ============================================================

COORD_NAMES = []

for x in range(X_SIZE):
    COORD_NAMES.extend([
        f"B0_x{x}",
        f"B1_x{x}",
        f"C0_x{x}",
        f"C1_x{x}",
    ])

COORD_PRETTY = []

for x in range(X_SIZE):
    COORD_PRETTY.extend([
        rf"<B_0>^[x={x}]",
        rf"<B_1>^[x={x}]",
        rf"<C_0>^[x={x}]",
        rf"<C_1>^[x={x}]",
    ])

print("=" * 90)
print("Ordem das coordenadas marginais")
print("=" * 90)

for i, name in enumerate(COORD_PRETTY):
    print(f"{i:2d}: {name}")


# ============================================================
# Acessores das funções determinísticas
# ============================================================

def hB_value(hB, y, m):
    """
    hB : Y x M -> B

    Ordem:
        hB[0] = hB(y=0,m=0)
        hB[1] = hB(y=0,m=1)
        hB[2] = hB(y=1,m=0)
        hB[3] = hB(y=1,m=1)
    """
    return hB[y * M_SIZE + m]


def hC_value(hC, z, m):
    """
    hC : Z x M -> C

    Ordem:
        hC[0] = hC(z=0,m=0)
        hC[1] = hC(z=0,m=1)
        hC[2] = hC(z=1,m=0)
        hC[3] = hC(z=1,m=1)
    """
    return hC[z * M_SIZE + m]


# ============================================================
# Vértice marginal
# ============================================================

def marginal_vertex(F, hB, hC):
    """
    Constrói um vértice marginal.

    Modelo clássico:

        m = F(x)
        b = hB(y,m)
        c = hC(z,m)

    Marginais:

        <B_y>^[x] = (-1)^b
        <C_z>^[x] = (-1)^c

    Para cada x, a ordem é:

        [<B0>, <B1>, <C0>, <C1>]
    """

    v = []

    for x in range(X_SIZE):
        m = F[x]

        # Marginais de Bob: <B0>, <B1>
        for y in range(Y_SIZE):
            b = hB_value(hB, y, m)
            v.append((-1) ** b)

        # Marginais de Charlie: <C0>, <C1>
        for z in range(Z_SIZE):
            c = hC_value(hC, z, m)
            v.append((-1) ** c)

    return tuple(v)


# ============================================================
# Gerar vértices e eliminar duplicatas
# ============================================================

def generate_marginal_vertices():
    """
    Enumera todas as estratégias determinísticas:

        F  : X -> M
        hB : Y x M -> B
        hC : Z x M -> C

    e retorna os vértices marginais únicos em R^16.
    """

    all_F = list(product(range(M_SIZE), repeat=X_SIZE))
    all_hB = list(product(range(B_SIZE), repeat=Y_SIZE * M_SIZE))
    all_hC = list(product(range(C_SIZE), repeat=Z_SIZE * M_SIZE))

    total_raw = len(all_F) * len(all_hB) * len(all_hC)

    print()
    print("=" * 90)
    print("Enumeração das estratégias determinísticas")
    print("=" * 90)
    print(f"F  : X -> M       = {len(all_F)}")
    print(f"hB : Y x M -> B   = {len(all_hB)}")
    print(f"hC : Z x M -> C   = {len(all_hC)}")
    print(f"Total bruto       = {total_raw}")
    print()

    vertices_set = set()
    strategies = []

    for F in all_F:
        for hB in all_hB:
            for hC in all_hC:

                v = marginal_vertex(F, hB, hC)

                vertices_set.add(v)

                strategies.append({
                    "F_x_to_m": F,
                    "hB_y_m_to_b": hB,
                    "hC_z_m_to_c": hC,
                    "marginal_vertex": v,
                })

    vertices = np.array(sorted(vertices_set), dtype=int)

    return vertices, strategies


vertices, strategies = generate_marginal_vertices()


# ============================================================
# Resumo do politopo
# ============================================================

print("=" * 90)
print("Politopo PAB marginal todo clássico com 4 preparações")
print("=" * 90)
print(f"Vértices únicos após eliminar duplicatas: {len(vertices)}")
print(f"Dimensão ambiente: {vertices.shape[1]}")
print(f"Dimensão afim: {np.linalg.matrix_rank(vertices - vertices[0])}")
print()


# ============================================================
# Salvar CSV dos vértices
# ============================================================

df_vertices = pd.DataFrame(vertices, columns=COORD_NAMES)
df_vertices.insert(0, "vertex_id", range(1, len(df_vertices) + 1))

csv_vertices = "pab_marginal_classical_X4_vertices.csv"
df_vertices.to_csv(csv_vertices, index=False)

print("CSV dos vértices salvo:")
print(f"  {csv_vertices}")


# ============================================================
# Salvar estratégias determinísticas
# ============================================================

vertex_to_id = {
    tuple(v): i + 1
    for i, v in enumerate(vertices)
}

for s in strategies:
    s["vertex_id"] = vertex_to_id[s["marginal_vertex"]]

df_strategies = pd.DataFrame(strategies)

df_strategies = df_strategies[
    [
        "vertex_id",
        "F_x_to_m",
        "hB_y_m_to_b",
        "hC_z_m_to_c",
        "marginal_vertex",
    ]
]

csv_strategies = "pab_marginal_classical_X4_strategies.csv"
df_strategies.to_csv(csv_strategies, index=False)

print("CSV das estratégias salvo:")
print(f"  {csv_strategies}")


# ============================================================
# Salvar V-representation do lrs
# ============================================================

def save_lrs_ext(vertices, filename):
    """
    Salva os vértices em formato V-representation do lrs.

    Cada linha é:

        1 v_0 v_1 ... v_15

    O primeiro 1 indica ponto afim.
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{len(vertices)} {vertices.shape[1] + 1} integer\n")

        for v in vertices:
            f.write("1 " + " ".join(str(int(x)) for x in v) + "\n")

        f.write("end\n")


ext_file = "pab_marginal_classical_X4_vertices.ext"
save_lrs_ext(vertices, ext_file)

print("V-representation do lrs salva:")
print(f"  {ext_file}")


# ============================================================
# Resumo final
# ============================================================

print()
print("=" * 90)
print("Resumo")
print("=" * 90)
print(f"|X| = {X_SIZE}")
print(f"|M| = {M_SIZE}")
print(f"Dimensão do vetor marginal: {DIM}")
print(f"Número de vértices únicos: {len(vertices)}")
print()
print("Comando para gerar facetas com lrs:")
print(f"lrs {ext_file} > pab_marginal_classical_X4_facets.ine")
```

