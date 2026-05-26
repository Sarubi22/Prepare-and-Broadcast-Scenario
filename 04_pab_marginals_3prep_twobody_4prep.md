# PAB Marginal Vertices (3 Prep) and Two-Body Vertices (4 Prep)

Two computations in a single notebook:

1. **3-preparation marginal polytope** (`|X|=3`): enumerates all deterministic  
   strategies and projects onto the 12-dimensional marginal space  
   `[<By>^[x], <Cz>^[x]]` for `x=0,1,2`.

2. **4-preparation two-body polytope** (`|X|=4`): enumerates strategies for  
   four preparations and keeps only the two-body correlators `<ByCz>^[x]`  
   for each `x`, producing a 16-dimensional projected polytope.

Both outputs are saved as `lrs` V-representation `.ext` files and as CSV  
tables for inspection.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
import numpy as np
import pandas as pd
from itertools import product

# ============================================================
# Parâmetros do cenário
# ============================================================

X_SIZE = 3

A_SIZE = 2      # gargalo clássico: a in {0,1}

Y_SIZE = 2
Z_SIZE = 2

MB_SIZE = 2     # mensagem para Bob
MC_SIZE = 2     # mensagem para Charlie

B_SIZE = 2
C_SIZE = 2


# ============================================================
# Ordem das 12 coordenadas marginais
# ============================================================

MARGINAL_NAMES = []
MARGINAL_PRETTY = []

for x in range(X_SIZE):
    MARGINAL_NAMES.extend([
        f"B0_x{x}",
        f"B1_x{x}",
        f"C0_x{x}",
        f"C1_x{x}",
    ])

    MARGINAL_PRETTY.extend([
        rf"<B_0>^[x={x}]",
        rf"<B_1>^[x={x}]",
        rf"<C_0>^[x={x}]",
        rf"<C_1>^[x={x}]",
    ])

print("Ordem das coordenadas marginais:")
for i, name in enumerate(MARGINAL_PRETTY):
    print(f"{i:2d}: {name}")


# ============================================================
# Acessores das funções h_B e h_C
# ============================================================

def hB_value(hB, y, mB):
    """
    hB : Y x MB -> B

    Ordem:
        hB[0] = hB(y=0,mB=0)
        hB[1] = hB(y=0,mB=1)
        hB[2] = hB(y=1,mB=0)
        hB[3] = hB(y=1,mB=1)
    """
    return hB[y * MB_SIZE + mB]


def hC_value(hC, z, mC):
    """
    hC : Z x MC -> C

    Ordem:
        hC[0] = hC(z=0,mC=0)
        hC[1] = hC(z=0,mC=1)
        hC[2] = hC(z=1,mC=0)
        hC[3] = hC(z=1,mC=1)
    """
    return hC[z * MC_SIZE + mC]


# ============================================================
# Construção do vértice marginal
# ============================================================

def marginal_vertex(fA, gB, gC, hB, hC):
    """
    Constrói o vértice marginal de 12 coordenadas.

    Para cada x, a ordem é:

        [<B0>, <B1>, <C0>, <C1>]

    com:

        a  = fA(x)
        mB = gB(a)
        mC = gC(a)
        b  = hB(y,mB)
        c  = hC(z,mC)

    Portanto:

        <B_y> = (-1)^b
        <C_z> = (-1)^c.
    """

    v = []

    for x in range(X_SIZE):
        a = fA[x]

        mB = gB[a]
        mC = gC[a]

        # <B0>, <B1>
        for y in range(Y_SIZE):
            b = hB_value(hB, y, mB)
            v.append((-1) ** b)

        # <C0>, <C1>
        for z in range(Z_SIZE):
            c = hC_value(hC, z, mC)
            v.append((-1) ** c)

    return tuple(v)


# ============================================================
# Gerar vértices e eliminar duplicatas
# ============================================================

def generate_marginal_vertices():
    all_fA = list(product(range(A_SIZE), repeat=X_SIZE))
    all_gB = list(product(range(MB_SIZE), repeat=A_SIZE))
    all_gC = list(product(range(MC_SIZE), repeat=A_SIZE))
    all_hB = list(product(range(B_SIZE), repeat=Y_SIZE * MB_SIZE))
    all_hC = list(product(range(C_SIZE), repeat=Z_SIZE * MC_SIZE))

    total_raw = (
        len(all_fA)
        * len(all_gB)
        * len(all_gC)
        * len(all_hB)
        * len(all_hC)
    )

    print()
    print("=" * 90)
    print("Enumeração das estratégias determinísticas")
    print("=" * 90)
    print(f"fA : X -> A        = {len(all_fA)}")
    print(f"gB : A -> MB       = {len(all_gB)}")
    print(f"gC : A -> MC       = {len(all_gC)}")
    print(f"hB : Y x MB -> B   = {len(all_hB)}")
    print(f"hC : Z x MC -> C   = {len(all_hC)}")
    print(f"Total bruto        = {total_raw}")
    print()

    vertices_set = set()
    strategies = []

    for fA in all_fA:
        for gB in all_gB:
            for gC in all_gC:
                for hB in all_hB:
                    for hC in all_hC:

                        v = marginal_vertex(fA, gB, gC, hB, hC)

                        vertices_set.add(v)

                        strategies.append({
                            "fA_x_to_a": fA,
                            "gB_a_to_mB": gB,
                            "gC_a_to_mC": gC,
                            "hB_y_mB_to_b": hB,
                            "hC_z_mC_to_c": hC,
                            "marginal_vertex": v,
                        })

    vertices = np.array(sorted(vertices_set), dtype=int)

    return vertices, strategies


vertices_marginal, strategies_marginal = generate_marginal_vertices()

print("=" * 90)
print("Politopo marginal")
print("=" * 90)
print(f"Vértices únicos após eliminar duplicatas: {len(vertices_marginal)}")
print(f"Dimensão ambiente: {vertices_marginal.shape[1]}")
print(f"Dimensão afim: {np.linalg.matrix_rank(vertices_marginal - vertices_marginal[0])}")
print()


# ============================================================
# Salvar CSV dos vértices
# ============================================================

df_vertices = pd.DataFrame(vertices_marginal, columns=MARGINAL_NAMES)
df_vertices.insert(0, "vertex_id", range(1, len(df_vertices) + 1))

df_vertices.to_csv(
    "pab_explicit_dag_marginals_12_vertices.csv",
    index=False,
)

print("CSV salvo:")
print("  pab_explicit_dag_marginals_12_vertices.csv")


# ============================================================
# Salvar estratégias explícitas
# ============================================================

df_strategies = pd.DataFrame(strategies_marginal)

df_strategies.to_csv(
    "pab_explicit_dag_marginals_12_strategies.csv",
    index=False,
)

print("Estratégias salvas:")
print("  pab_explicit_dag_marginals_12_strategies.csv")


# ============================================================
# Salvar V-representation do lrs
# ============================================================

def save_lrs_ext(vertices, filename):
    """
    Salva em formato V-representation do lrs.

    Cada linha é:

        1 v_0 v_1 ... v_11

    O primeiro 1 indica ponto afim.
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{len(vertices)} {vertices.shape[1] + 1} integer\n")

        for v in vertices:
            f.write("1 " + " ".join(str(int(x)) for x in v) + "\n")

        f.write("end\n")


save_lrs_ext(
    vertices_marginal,
    "pab_explicit_dag_marginals_12_vertices.ext",
)

print("V-representation do lrs salva:")
print("  pab_explicit_dag_marginals_12_vertices.ext")


print()
print("=" * 90)
print("Comando para gerar as facetas com lrs")
print("=" * 90)
print("lrs pab_explicit_dag_marginals_12_vertices.ext > pab_explicit_dag_marginals_12_facets.ine")
```


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

# ============================================================
# PAB two-body todo clássico com 4 preparações
# ============================================================

X_SIZE = 4      # agora são 4 preparações
Y_SIZE = 2
Z_SIZE = 2
M_SIZE = 2      # mensagem clássica binária
B_SIZE = 2
C_SIZE = 2

BLOCK = Y_SIZE * Z_SIZE
DIM = X_SIZE * BLOCK


# ============================================================
# Ordem dos correladores
# ============================================================

COORD_NAMES = [
    f"B{y}C{z}_x{x}"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

COORD_PRETTY = [
    rf"<B_{y}C_{z}>^[x={x}]"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

print("=" * 90)
print("Ordem dos correladores")
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
# Vértice two-body
# ============================================================

def twobody_vertex(F, hB, hC):
    """
    Constrói um vértice two-body.

    Modelo:

        m = F(x)
        b = hB(y,m)
        c = hC(z,m)

    Correlador:

        E_yz^x = (-1)^{b+c}.
    """

    v = []

    for x in range(X_SIZE):
        m = F[x]

        for y in range(Y_SIZE):
            b = hB_value(hB, y, m)
            bsign = (-1) ** b

            for z in range(Z_SIZE):
                c = hC_value(hC, z, m)
                csign = (-1) ** c

                v.append(bsign * csign)

    return tuple(v)


# ============================================================
# Gerar vértices e eliminar duplicatas
# ============================================================

def generate_twobody_vertices():
    """
    Enumera todas as estratégias determinísticas:

        F  : X -> M
        hB : Y x M -> B
        hC : Z x M -> C

    e retorna os vértices únicos em R^16.
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

                v = twobody_vertex(F, hB, hC)

                vertices_set.add(v)

                strategies.append({
                    "F_x_to_m": F,
                    "hB_y_m_to_b": hB,
                    "hC_z_m_to_c": hC,
                    "twobody_vertex": v,
                })

    vertices = np.array(sorted(vertices_set), dtype=int)

    return vertices, strategies


vertices, strategies = generate_twobody_vertices()

print("=" * 90)
print("Politopo PAB two-body todo clássico com 4 preparações")
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

csv_vertices = "pab_twobody_classical_X4_vertices.csv"
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
    s["vertex_id"] = vertex_to_id[s["twobody_vertex"]]

df_strategies = pd.DataFrame(strategies)

df_strategies = df_strategies[
    [
        "vertex_id",
        "F_x_to_m",
        "hB_y_m_to_b",
        "hC_z_m_to_c",
        "twobody_vertex",
    ]
]

csv_strategies = "pab_twobody_classical_X4_strategies.csv"
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


ext_file = "pab_twobody_classical_X4_vertices.ext"
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
print(f"Dimensão do vetor two-body: {DIM}")
print(f"Número de vértices únicos: {len(vertices)}")
print()
print("Comando para gerar facetas com lrs:")
print(f"lrs {ext_file} > pab_twobody_classical_X4_facets.ine")
```

