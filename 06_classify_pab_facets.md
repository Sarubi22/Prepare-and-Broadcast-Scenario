# Classification of PAB Two-Body Facets from `.ine` File

Reads a facet file (`.ine` format from `lrs`) for the PAB two-body polytope  
and classifies the facets into equivalence classes under the symmetry group  
of the scenario.

**Scenario:** `|X|=3, |Y|=|Z|=2, |M|=2`, binary outputs.

Symmetries applied:
- Permutation of preparations (`x` relabelling)
- Relabelling of Bob's and Charlie's measurement settings
- Exchange of Bob and Charlie
- Flipping output bits

Each facet is reduced to a canonical form (primitive integer row, lexicographically  
smallest representative in its orbit). The distinct classes are counted and  
representative inequalities are exported to CSV.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

# Classificação de facetas PAB two-body a partir de arquivo `.ine`

Este notebook classifica facetas de um politopo PAB two-body geradas pelo `lrs`.

Cenário assumido:

\[
|X|=3,\qquad |Y|=|Z|=2,\qquad |M|=2.
\]

A ordem dos correladores é

\[
[
\langle B_0C_0\rangle^{[0]},
\langle B_0C_1\rangle^{[0]},
\langle B_1C_0\rangle^{[0]},
\langle B_1C_1\rangle^{[0]},
\dots,
\langle B_1C_1\rangle^{[2]}
].
\]

Cada linha do arquivo `.ine` é interpretada na convenção do `lrs`:

\[
b+a\cdot v\ge 0.
\]

O notebook permite classificar as facetas por dois grupos:

- `physical`: relabelings físicos do cenário.
- `full-block`: automorfismos maiores da projeção two-body.


```python
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction
from functools import reduce
from itertools import permutations, product
from math import gcd, lcm
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
```


## 1. Parâmetros do cenário e ordem dos correladores


```python
# ============================================================
# Parâmetros do cenário
# ============================================================

X_SIZE = 3
Y_SIZE = 2
Z_SIZE = 2
M_SIZE = 2

BLOCK = Y_SIZE * Z_SIZE
DIM = X_SIZE * BLOCK

COORD_NAMES = [
    rf"<B_{y}C_{z}>^[x={x}]"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

Row = Tuple[int, ...]
Transform = Tuple[Tuple[int, ...], Tuple[int, ...]]

print(f"DIM = {DIM}")
for i, name in enumerate(COORD_NAMES):
    print(f"{i:2d}: {name}")
```


## 2. Leitura do arquivo `.ine` do `lrs`

A função abaixo lê apenas as linhas entre `begin` e `end`.

Cada linha é convertida para uma linha inteira primitiva. O sinal da desigualdade não é invertido. Portanto, preservamos a orientação

\[
b+a\cdot v\ge 0.
\]


```python
# ============================================================
# Leitura do .ine do lrs
# ============================================================

def primitive_integer_row(values: Sequence[Fraction]) -> Row:
    """Converte uma linha racional para uma linha inteira primitiva.

    Importante: o sinal da desigualdade NÃO é invertido.
    Apenas removemos denominadores e divisor comum global.
    Isso preserva a orientação b + a.v >= 0 do lrs.
    """
    den = 1
    for q in values:
        den = lcm(den, q.denominator)

    ints = [int(q * den) for q in values]

    g = 0
    for val in ints:
        g = gcd(g, abs(val))

    if g == 0:
        return tuple(ints)

    return tuple(val // g for val in ints)


def parse_lrs_ine(path: str | Path) -> List[Row]:
    """Lê as facetas de um arquivo .ine gerado pelo lrs."""
    path = Path(path)
    rows: List[Row] = []
    inside = False

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()

        if not line or line.startswith("*"):
            continue

        low = line.lower()

        if low == "begin":
            inside = True
            continue

        if low == "end":
            break

        if not inside:
            continue

        parts = line.split()

        # Linha do tipo: "8400 13 rational" ou "***** 13 rational".
        if any(tok.lower() in {"rational", "integer", "real"} for tok in parts):
            continue

        values = [Fraction(tok) for tok in parts]
        row = primitive_integer_row(values)
        rows.append(row)

    if not rows:
        raise ValueError(f"Nenhuma faceta foi lida de {path}.")

    lengths = sorted(set(len(r) for r in rows))

    if lengths != [DIM + 1]:
        raise ValueError(
            f"O arquivo tem linhas com tamanhos {lengths}, mas eu esperava {DIM + 1} "
            f"colunas: 1 constante + {DIM} coeficientes."
        )

    return rows
```


## 3. Geração dos vértices clássicos-produto two-body

O canal clássico é

\[
x\mapsto m=F(x),\qquad m\in\{0,1\},
\]

e as respostas determinísticas são

\[
b=h_B(y,m),\qquad c=h_C(z,m).
\]

Na projeção two-body,

\[
\langle B_yC_z\rangle^{[x]}
=
(-1)^{h_B(y,F(x))}
(-1)^{h_C(z,F(x))}.
\]


```python
# ============================================================
# Vértices do modelo clássico-produto two-body
# ============================================================

def idx(x: int, y: int, z: int) -> int:
    return x * BLOCK + y * Z_SIZE + z


def idx_block(y: int, z: int) -> int:
    return y * Z_SIZE + z


def generate_twobody_vertices() -> np.ndarray:
    """Gera os vértices do modelo clássico-produto projetado em two-body."""
    vertices = set()

    all_F = list(product(range(M_SIZE), repeat=X_SIZE))
    all_hB = list(product(range(2), repeat=Y_SIZE * M_SIZE))
    all_hC = list(product(range(2), repeat=Z_SIZE * M_SIZE))

    for F in all_F:
        for hB in all_hB:
            for hC in all_hC:
                v: List[int] = []

                for x in range(X_SIZE):
                    m = F[x]
                    block = [0] * BLOCK

                    for y in range(Y_SIZE):
                        bsign = (-1) ** hB[y * M_SIZE + m]

                        for z in range(Z_SIZE):
                            csign = (-1) ** hC[z * M_SIZE + m]
                            block[idx_block(y, z)] = bsign * csign

                    v.extend(block)

                vertices.add(tuple(v))

    return np.array(sorted(vertices), dtype=int)


vertices = generate_twobody_vertices()

print(f"Número de vértices únicos na projeção two-body: {len(vertices)}")
print(f"Dimensão afim do politopo: {np.linalg.matrix_rank(vertices - vertices[0])}")
```


## 4. Grupos de simetria

### Grupo `physical`

Usa apenas relabelings físicos:

\[
x\text{-permutations},\quad
y\text{-permutations},\quad
z\text{-permutations},\quad
B\leftrightarrow C,\quad
\text{flips de saída}.
\]

### Grupo `full-block`

Usa um grupo maior de automorfismos da projeção two-body. Ele pode juntar facetas que não são equivalentes por relabelings físicos de Bob e Charlie.


```python
# ============================================================
# Grupos de simetria / automorfismo
# ============================================================

def physical_transforms() -> List[Transform]:
    """Relabelings físicos: x, y, z, Bob<->Charlie e flips de saída."""
    transforms: Dict[Transform, Transform] = {}

    for px in permutations(range(X_SIZE)):
        for pB in permutations(range(Y_SIZE)):
            for pC in permutations(range(Z_SIZE)):
                for swap_BC in (False, True):
                    for sB in product((1, -1), repeat=Y_SIZE):
                        for sC in product((1, -1), repeat=Z_SIZE):
                            old_indices: List[int] = []
                            signs: List[int] = []

                            for x in range(X_SIZE):
                                for y in range(Y_SIZE):
                                    for z in range(Z_SIZE):
                                        if not swap_BC:
                                            old_x = px[x]
                                            old_y = pB[y]
                                            old_z = pC[z]
                                        else:
                                            old_x = px[x]
                                            # Depois da troca B<->C,
                                            # o novo Bob vem do antigo Charlie
                                            # e o novo Charlie vem do antigo Bob.
                                            old_y = pB[z]
                                            old_z = pC[y]

                                        old_indices.append(idx(old_x, old_y, old_z))
                                        signs.append(sB[y] * sC[z])

                            tr = (tuple(old_indices), tuple(signs))
                            transforms[tr] = tr

    return list(transforms.values())


def full_block_transforms() -> List[Transform]:
    """Automorfismos maiores da projeção two-body.

    Cada bloco x é um politopo com vértices de paridade par:
        {s in {+-1}^4 : prod_i s_i = +1}.

    Seus automorfismos lineares incluem qualquer permutação das quatro
    coordenadas e qualquer padrão de sinais com produto +1.

    Aplicamos o mesmo automorfismo de bloco a todos os x,
    além de permutar as preparações.
    """
    transforms: List[Transform] = []

    block_perms = list(permutations(range(BLOCK)))
    even_signs = [
        sgn for sgn in product((1, -1), repeat=BLOCK)
        if reduce(lambda a, b: a * b, sgn) == 1
    ]

    for px in permutations(range(X_SIZE)):
        for pb in block_perms:
            for sg in even_signs:
                old_indices: List[int] = []
                signs: List[int] = []

                for x in range(X_SIZE):
                    for k in range(BLOCK):
                        old_indices.append(px[x] * BLOCK + pb[k])
                        signs.append(sg[k])

                transforms.append((tuple(old_indices), tuple(signs)))

    return transforms


def apply_transform_to_row(row: Row, transform: Transform) -> Row:
    """Aplica um relabeling à desigualdade b + a.v >= 0.

    A transformação das coordenadas é
        w_i = s_i v_{old_i}.

    Logo, a desigualdade transformada tem coeficientes
        a'_i = s_i a_{old_i}.
    """
    b = row[0]
    a = row[1:]
    old_indices, signs = transform

    return (b,) + tuple(signs[i] * a[old_indices[i]] for i in range(DIM))


def canonical_row(row: Row, transforms: Sequence[Transform]) -> Row:
    """Representante canônico lexicográfico da órbita da faceta."""
    b = row[0]
    a = row[1:]

    best = None

    for old_indices, signs in transforms:
        candidate = (b,) + tuple(signs[i] * a[old_indices[i]] for i in range(DIM))

        if best is None or candidate < best:
            best = candidate

    assert best is not None
    return best
```


## 5. Validação, classificação e saída


```python
# ============================================================
# Validação, classificação e impressão
# ============================================================

def evaluate_row_on_vertices(row: Row, vertices: np.ndarray) -> np.ndarray:
    b = row[0]
    a = np.array(row[1:], dtype=int)

    return b + vertices @ a


def validate_rows(rows: Sequence[Row], vertices: np.ndarray) -> Dict[str, int]:
    dim_affine = np.linalg.matrix_rank(vertices - vertices[0])

    invalid = 0
    nonfacets = 0
    min_global = None

    for row in rows:
        vals = evaluate_row_on_vertices(row, vertices)
        mn = int(vals.min())
        min_global = mn if min_global is None else min(min_global, mn)

        if mn < 0:
            invalid += 1
            continue

        sat = vertices[vals == 0]

        if len(sat) == 0:
            nonfacets += 1
            continue

        rank_sat = np.linalg.matrix_rank(sat - sat[0])

        if rank_sat != dim_affine - 1:
            nonfacets += 1

    return {
        "polytope_dimension": int(dim_affine),
        "invalid_inequalities": int(invalid),
        "nonfacets_by_rank_test": int(nonfacets),
        "minimum_value_seen": int(min_global),
    }


def pretty_inequality(row: Row) -> str:
    b = row[0]
    terms = [str(b)]

    for coef, name in zip(row[1:], COORD_NAMES):
        if coef == 0:
            continue

        if coef > 0:
            terms.append(f"+ {coef} {name}")
        else:
            terms.append(f"- {abs(coef)} {name}")

    return " ".join(terms) + " >= 0"


def classify_facets(rows: Sequence[Row], transforms: Sequence[Transform]) -> Dict[Row, List[int]]:
    classes: Dict[Row, List[int]] = defaultdict(list)
    cache: Dict[Row, Row] = {}

    for i, row in enumerate(rows, start=1):
        if row not in cache:
            cache[row] = canonical_row(row, transforms)

        classes[cache[row]].append(i)

    return dict(classes)


def classes_to_dataframe(
    classes: Dict[Row, List[int]],
    vertices: np.ndarray,
) -> pd.DataFrame:
    ordered = sorted(classes.items(), key=lambda item: item[1][0])
    data = []

    for class_id, (rep, indices) in enumerate(ordered, start=1):
        vals = evaluate_row_on_vertices(rep, vertices)
        sat = vertices[vals == 0]
        rank_sat = int(np.linalg.matrix_rank(sat - sat[0])) if len(sat) else -1

        data.append({
            "class_id": class_id,
            "size": len(indices),
            "first_facet_in_file": indices[0],
            "representative_b": rep[0],
            "representative_coefficients": list(rep[1:]),
            "representative_row": list(rep),
            "saturated_vertices_of_representative": int(len(sat)),
            "rank_of_saturated_set": rank_sat,
            "facet_indices": indices,
            "inequality": pretty_inequality(rep),
        })

    return pd.DataFrame(data)


def write_outputs(
    classes: Dict[Row, List[int]],
    rows: Sequence[Row],
    vertices: np.ndarray,
    out_prefix: str,
    group_name: str,
) -> None:
    ordered = sorted(classes.items(), key=lambda item: item[1][0])

    csv_path = Path(f"{out_prefix}_{group_name}.csv")
    txt_path = Path(f"{out_prefix}_{group_name}.txt")
    json_path = Path(f"{out_prefix}_{group_name}.json")

    df = classes_to_dataframe(classes, vertices)
    df.to_csv(csv_path, index=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Classificação de facetas PAB two-body\n")
        f.write(f"Grupo usado: {group_name}\n")
        f.write(f"Total de facetas lidas: {len(rows)}\n")
        f.write(f"Total de classes: {len(ordered)}\n")
        f.write(f"Distribuição dos tamanhos das classes: {dict(Counter(len(v) for v in classes.values()))}\n")
        f.write("\n")

        f.write("Ordem das coordenadas:\n")
        for i, name in enumerate(COORD_NAMES):
            f.write(f"  {i:2d}: {name}\n")
        f.write("\n")

        for class_id, (rep, indices) in enumerate(ordered, start=1):
            vals = evaluate_row_on_vertices(rep, vertices)
            sat = vertices[vals == 0]
            rank_sat = int(np.linalg.matrix_rank(sat - sat[0])) if len(sat) else -1

            f.write(f"=== Classe {class_id} ===\n")
            f.write(f"Número de facetas na classe: {len(indices)}\n")
            f.write(f"Primeira faceta no arquivo: {indices[0]}\n")
            f.write(f"Facetas do arquivo nessa classe: {indices}\n")
            f.write(f"Representante canônico: {list(rep)}\n")
            f.write(f"Inequação: {pretty_inequality(rep)}\n")
            f.write(f"Vértices saturados pelo representante: {len(sat)}\n")
            f.write(f"Rank do conjunto saturado: {rank_sat}\n")
            f.write("\n")

    json_data = df.to_dict(orient="records")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("Arquivos salvos:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")
```


## 6. Escolha do arquivo `.ine` e do grupo

Coloque o arquivo `.ine` na mesma pasta do notebook ou use o caminho completo.

Exemplos:

```python
INE_PATH = "resultado.ine"
GROUP = "physical"
```

ou

```python
GROUP = "full-block"
```


```python
# ============================================================
# Configuração principal
# ============================================================

from pathlib import Path

# Caminho do arquivo .ine na pasta Downloads
INE_PATH = Path.home() / "Downloads" / "resultado.ine"

GROUP = "physical"          # opções: "physical" ou "full-block"
OUT_PREFIX = "facet_classes_pab_twobody"
RUN_VALIDATION = True

if not INE_PATH.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {INE_PATH}")

print(f"Arquivo .ine: {INE_PATH}")
print(f"Grupo escolhido: {GROUP}")
```


## 7. Rodar a classificação


```python
# ============================================================
# Execução
# ============================================================

rows = parse_lrs_ine(INE_PATH)
vertices = generate_twobody_vertices()

print("=" * 88)
print("PAB two-body — classificação de facetas")
print("=" * 88)
print(f"Facetas lidas: {len(rows)}")
print(f"Dimensão das linhas: {len(rows[0])} = 1 + {DIM}")
print(f"Vértices gerados para validação: {len(vertices)}")
print()

if GROUP == "physical":
    transforms = physical_transforms()
elif GROUP == "full-block":
    transforms = full_block_transforms()
else:
    raise ValueError("GROUP deve ser 'physical' ou 'full-block'.")

print(f"Número de transformações distintas: {len(transforms)}")
print()

if RUN_VALIDATION:
    stats = validate_rows(rows, vertices)
    print("Validação contra os vértices:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

classes = classify_facets(rows, transforms)
class_df = classes_to_dataframe(classes, vertices)

print("Resumo da classificação:")
print(f"  total de classes: {len(classes)}")
print(f"  distribuição dos tamanhos: {dict(Counter(len(v) for v in classes.values()))}")
print()

class_df[[
    "class_id",
    "size",
    "first_facet_in_file",
    "representative_row",
    "saturated_vertices_of_representative",
    "rank_of_saturated_set",
]]
```


## 8. Salvar os resultados

Esta célula salva três arquivos:

- `.csv`: resumo em tabela;
- `.txt`: versão legível com as inequações;
- `.json`: dados completos.


```python
write_outputs(classes, rows, vertices, OUT_PREFIX, GROUP)
```


## 9. Inspecionar uma classe específica

Altere `CLASS_ID` para visualizar o representante, a inequação e os índices das facetas equivalentes.


```python
CLASS_ID = 1

row = class_df.loc[class_df["class_id"] == CLASS_ID].iloc[0]

print(f"Classe {CLASS_ID}")
print(f"Tamanho da classe: {row['size']}")
print(f"Primeira faceta no arquivo: {row['first_facet_in_file']}")
print()
print("Representante:")
print(row["representative_row"])
print()
print("Inequação:")
print(row["inequality"])
print()
print("Facetas do arquivo nessa classe:")
print(row["facet_indices"])
```


## 10. Rodar também o grupo `full-block` sem alterar a configuração principal

Use esta célula opcional se quiser comparar a classificação física com o grupo maior.


```python
# Célula opcional

RUN_FULL_BLOCK_TOO = False

if RUN_FULL_BLOCK_TOO:
    full_transforms = full_block_transforms()
    full_classes = classify_facets(rows, full_transforms)
    full_df = classes_to_dataframe(full_classes, vertices)

    print(f"Total de classes full-block: {len(full_classes)}")
    print(f"Distribuição dos tamanhos: {dict(Counter(len(v) for v in full_classes.values()))}")

    display(full_df[[
        "class_id",
        "size",
        "first_facet_in_file",
        "representative_row",
        "saturated_vertices_of_representative",
        "rank_of_saturated_set",
    ]])

    write_outputs(full_classes, rows, vertices, OUT_PREFIX, "full-block")
```


```python
# ============================================================
# Busca de violação quântica das facetas PAB two-body
# com canal de broadcasting ruidoso
#
# ALTERADO:
#   - Canal Bell-beta -> canal geral U(p)=exp[-iH(p)]
#   - H(p)=sum_k p_k Lambda_k, k=1,...,15
#   - V(p)=duas primeiras colunas de U(p)
#   - Impressão mostra W_clean completo, não apenas o gap
# ============================================================

import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

# ============================================================
# Configurações
# ============================================================

CLASSES_CSV = "facet_classes_pab_twobody_physical.csv"

N_GLOBAL_RUNS = 5

DE_MAXITER = 700
DE_POPSIZE = 18

VIOL_TOL = 1e-7

SELECT_CLASSES = None

# gamma = 1: canal limpo
# gamma = 0: saída totalmente branca I_4/4
GAMMA_TEST = 1.0

# Range dos parâmetros p_k do canal SU(4)
CHANNEL_P_BOUND = np.pi


# ============================================================
# Matrizes de Pauli e funções básicas
# ============================================================

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = np.array([SX, SY, SZ], dtype=complex)


def bloch_vec(theta, phi):
    """Vetor unitário na esfera de Bloch."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=float)


def qubit_state(theta, phi):
    """Estado puro de qubit rho = (I + r.sigma)/2."""
    r = bloch_vec(theta, phi)
    rho = 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)
    return rho


def observable(theta, phi):
    """Observável binário projetivo O = n.sigma, com autovalores +-1."""
    n = bloch_vec(theta, phi)
    return n[0] * SX + n[1] * SY + n[2] * SZ


# ============================================================
# Canal geral das Eqs. (51)-(52)
# ============================================================

def su_generators(d=4):
    """
    Gera uma base hermitiana e traceless de su(d).

    Para d=4, retorna 15 geradores:
      - 6 simétricos off-diagonal;
      - 6 antissimétricos off-diagonal;
      - 3 diagonais.
    """
    gens = []

    # Off-diagonal: simétricos e antissimétricos
    for j in range(d):
        for k in range(j + 1, d):
            S = np.zeros((d, d), dtype=complex)
            S[j, k] = 1.0
            S[k, j] = 1.0
            gens.append(S)

            A = np.zeros((d, d), dtype=complex)
            A[j, k] = -1j
            A[k, j] = 1j
            gens.append(A)

    # Diagonais
    for ell in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for j in range(ell):
            D[j, j] = 1.0
        D[ell, ell] = -ell
        D *= np.sqrt(2.0 / (ell * (ell + 1.0)))
        gens.append(D)

    return gens


SU4 = su_generators(4)

if len(SU4) != 15:
    raise RuntimeError("A base su(4) não tem 15 geradores.")


def general_su4_isometry(p):
    """
    Canal geral:

        U(p) = exp[-i H(p)]
        H(p) = sum_k p_k Lambda_k

    A isometria V: C^2 -> C^2 tensor C^2 é obtida tomando
    as duas primeiras colunas de U(p):

        V = U[:, :2]

    Como U é unitária, V^dagger V = I_2.
    """
    H = np.zeros((4, 4), dtype=complex)

    for pk, Gk in zip(p, SU4):
        H += pk * Gk

    U = expm(-1j * H)
    V = U[:, :2]

    return V


def broadcast_state(rho, p, gamma=1.0):
    """
    Estado bipartido depois do canal ruidoso:

        Phi_gamma(rho)
        =
        gamma V(p) rho V(p)^dagger
        +
        (1-gamma) I_4/4.
    """
    V = general_su4_isometry(p)
    tau_clean = V @ rho @ V.conj().T
    tau = gamma * tau_clean + (1.0 - gamma) * I4 / 4.0
    return tau


# ============================================================
# Parametrização do comportamento quântico
# ============================================================

def unpack_params(z):
    """
    Parâmetros:

    z[0:15] = parâmetros p_k do canal geral U(p).

    Para cada preparação x=0,1,2:
        theta_x, phi_x

    Para Bob:
        theta_B0, phi_B0, theta_B1, phi_B1

    Para Charlie:
        theta_C0, phi_C0, theta_C1, phi_C1

    Total:
        15 + 3*2 + 2*2 + 2*2 = 29 parâmetros.
    """
    pos = 0

    p = np.array(z[pos:pos + 15], dtype=float)
    pos += 15

    prep_angles = []
    for _ in range(3):
        prep_angles.append((z[pos], z[pos + 1]))
        pos += 2

    B_angles = []
    for _ in range(2):
        B_angles.append((z[pos], z[pos + 1]))
        pos += 2

    C_angles = []
    for _ in range(2):
        C_angles.append((z[pos], z[pos + 1]))
        pos += 2

    return p, prep_angles, B_angles, C_angles


def quantum_twobody_vector(z, gamma=1.0):
    """
    Retorna o vetor v em R^12 na ordem:

        x=0: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=1: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=2: <B0C0>, <B0C1>, <B1C0>, <B1C1>
    """
    p, prep_angles, B_angles, C_angles = unpack_params(z)

    B_obs = [observable(*ang) for ang in B_angles]
    C_obs = [observable(*ang) for ang in C_angles]

    BC_obs = [
        [np.kron(B_obs[y], C_obs[z_]) for z_ in range(2)]
        for y in range(2)
    ]

    vals = []

    for x in range(3):
        rho_x = qubit_state(*prep_angles[x])
        tau_x = broadcast_state(rho_x, p, gamma=gamma)

        for y in range(2):
            for z_ in range(2):
                val = np.trace(tau_x @ BC_obs[y][z_]).real
                vals.append(float(val))

    return np.array(vals, dtype=float)


# ============================================================
# Facetas
# ============================================================

def load_class_representatives():
    """
    Carrega os representantes de classe.

    Prioridade:
        1. Usa class_df já existente no notebook.
        2. Se não existir, lê o CSV facet_classes_pab_twobody_physical.csv.
    """
    if "class_df" in globals():
        df = class_df.copy()
    else:
        path = Path(CLASSES_CSV)

        if not path.exists():
            raise FileNotFoundError(
                f"Não encontrei {CLASSES_CSV}. Rode antes a classificação "
                "ou ajuste CLASSES_CSV para o caminho correto."
            )

        df = pd.read_csv(path)

        for col in ["representative_row", "representative_coefficients", "facet_indices"]:
            if col in df.columns:
                df[col] = df[col].apply(ast.literal_eval)

    if SELECT_CLASSES is not None:
        df = df[df["class_id"].isin(SELECT_CLASSES)].copy()

    rows = []

    for _, r in df.iterrows():
        rep = r["representative_row"]

        if isinstance(rep, str):
            rep = ast.literal_eval(rep)

        rep = np.array(rep, dtype=float)

        rows.append({
            "class_id": int(r["class_id"]),
            "size": int(r["size"]),
            "first_facet_in_file": int(r["first_facet_in_file"]),
            "row": rep,
        })

    return rows


def facet_value(row, v):
    """
    Avalia a desigualdade no formato LRS:

        F(v) = b + a.v.

    A faceta é satisfeita quando F(v) >= 0.
    Violação clássica se F(v) < 0.
    """
    b = row[0]
    a = row[1:]
    return float(b + np.dot(a, v))


def witness_value_from_F(row, F):
    """
    Para LRS:
        F = b + a.v.

    Definimos o witness positivo:
        W = -a.v.

    Como F = b - W, temos:
        W = b - F.
    """
    b = float(row[0])
    return b - F


# ============================================================
# Otimização
# ============================================================

# Bounds:
# 15 parâmetros p_k em [-pi, pi]
# todos os theta em [0, pi]
# todos os phi em [0, 2pi]
BOUNDS = []

# Canal SU(4): p_1,...,p_15
for _ in range(15):
    BOUNDS.append((-CHANNEL_P_BOUND, CHANNEL_P_BOUND))

# 3 preparações: theta, phi
for _ in range(3):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2 * np.pi))

# 2 medições de Bob: theta, phi
for _ in range(2):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2 * np.pi))

# 2 medições de Charlie: theta, phi
for _ in range(2):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2 * np.pi))


def optimize_facet(row, seed0=1234):
    """
    Minimiza F_clean = b + a.v_clean.

    Também calcula:
        W_clean = b - F_min_clean

    onde W_clean é o valor completo do witness positivo.

    Para o ruído na saída:
        Phi_gamma(rho)
        =
        gamma V rho V^dagger + (1-gamma) I_4/4,

    os correladores two-body escalam com gamma, e

        F_gamma = b + gamma(F_min_clean - b).

    Assim:
        gamma_crit = b / W_clean.
    """
    row = np.array(row, dtype=float)
    b = float(row[0])

    def obj_clean(z):
        v = quantum_twobody_vector(z, gamma=1.0)
        return facet_value(row, v)

    best = None

    for run in range(N_GLOBAL_RUNS):
        seed = seed0 + 1000 * run

        res_de = differential_evolution(
            obj_clean,
            bounds=BOUNDS,
            seed=seed,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            polish=False,
            tol=1e-8,
            updating="immediate",
            workers=1,
        )

        res_loc = minimize(
            obj_clean,
            res_de.x,
            method="Nelder-Mead",
            options={
                "maxiter": 5000,
                "xatol": 1e-10,
                "fatol": 1e-10,
                "disp": False,
            },
        )

        candidates = [res_de]

        if res_loc.success or np.isfinite(res_loc.fun):
            candidates.append(res_loc)

        for res in candidates:
            val = float(res.fun)

            if best is None or val < best["F_min_clean"]:
                best = {
                    "F_min_clean": val,
                    "params": np.array(res.x, dtype=float),
                    "success": bool(getattr(res, "success", False)),
                    "message": str(getattr(res, "message", "")),
                    "seed": seed,
                }

    F_min = best["F_min_clean"]

    W_clean = witness_value_from_F(row, F_min)
    violation_gap_clean = max(0.0, W_clean - b)

    # Como v_gamma = gamma v_clean no modelo de ruído na saída:
    # F_gamma = b + gamma(F_min - b)
    F_at_gamma_test = b + GAMMA_TEST * (F_min - b)
    W_at_gamma_test = witness_value_from_F(row, F_at_gamma_test)

    if W_clean > b + VIOL_TOL and b > 0:
        gamma_crit = b / W_clean
    elif W_clean > b + VIOL_TOL and abs(b) < 1e-12:
        gamma_crit = 0.0
    else:
        gamma_crit = np.nan

    best.update({
        "b": b,
        "W_clean": W_clean,
        "violation_gap_clean": violation_gap_clean,
        "gamma_crit": gamma_crit,
        "F_at_gamma_test": F_at_gamma_test,
        "W_at_gamma_test": W_at_gamma_test,
        "violates_at_gamma_test": F_at_gamma_test < -VIOL_TOL,
    })

    return best


# ============================================================
# Rodar busca em todas as classes
# ============================================================

facet_reps = load_class_representatives()

print("=" * 100)
print("Busca de violação quântica das facetas PAB two-body")
print("Canal geral U(p)=exp[-iH(p)] das Eqs. (51)-(52)")
print("=" * 100)
print(f"Número de facetas representantes testadas: {len(facet_reps)}")
print(f"Ruído de teste GAMMA_TEST = {GAMMA_TEST}")
print(f"N_GLOBAL_RUNS = {N_GLOBAL_RUNS}")
print(f"Número total de parâmetros otimizados = {len(BOUNDS)}")
print()

results = []

for k, item in enumerate(facet_reps, start=1):
    class_id = item["class_id"]
    row = item["row"]

    print(f"[{k:03d}/{len(facet_reps):03d}] Otimizando classe {class_id} ...")

    opt = optimize_facet(row, seed0=10_000 + 97 * class_id)

    zbest = opt["params"]
    vclean = quantum_twobody_vector(zbest, gamma=1.0)
    vgamma = quantum_twobody_vector(zbest, gamma=GAMMA_TEST)

    results.append({
        "class_id": class_id,
        "size": item["size"],
        "first_facet_in_file": item["first_facet_in_file"],
        "b": opt["b"],
        "F_min_clean": opt["F_min_clean"],

        # Novo:
        "W_clean": opt["W_clean"],
        "violation_gap_clean": opt["violation_gap_clean"],

        "gamma_crit": opt["gamma_crit"],
        "F_at_gamma_test": opt["F_at_gamma_test"],
        "W_at_gamma_test": opt["W_at_gamma_test"],
        "violates_at_gamma_test": opt["violates_at_gamma_test"],

        # Não há mais beta.
        # Guardamos os 15 parâmetros do canal separadamente.
        "channel_params": zbest[:15].tolist(),

        "params": zbest.tolist(),
        "v_clean": vclean.tolist(),
        "v_gamma_test": vgamma.tolist(),
        "representative_row": row.tolist(),
        "seed": opt["seed"],
    })

    print(
        f"    F_clean = {opt['F_min_clean']:.12f} | "
        f"W_clean = {opt['W_clean']:.12f} | "
        f"gap = {opt['violation_gap_clean']:.12f} | "
        f"gamma_c = {opt['gamma_crit']}"
    )

results_df = pd.DataFrame(results)

# Ordena pelos maiores valores completos do witness.
results_df = results_df.sort_values(
    by=["W_clean", "gamma_crit"],
    ascending=[False, True],
).reset_index(drop=True)

print()
print("=" * 100)
print("Resumo das melhores violações")
print("=" * 100)

display_cols = [
    "class_id",
    "size",
    "first_facet_in_file",
    "b",
    "F_min_clean",
    "W_clean",
    "violation_gap_clean",
    "gamma_crit",
    "violates_at_gamma_test",
]

display(results_df[display_cols].head(20))

# Salva resultados.
out_csv = "quantum_violations_pab_twobody_general_channel_noise.csv"
out_json = "quantum_violations_pab_twobody_general_channel_noise.json"

results_df.to_csv(out_csv, index=False)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print()
print("Arquivos salvos:")
print(f"  {out_csv}")
print(f"  {out_json}")
```


```python
# ============================================================
# VIOLAÇÃO QUÂNTICA APENAS DA CLASSE 3
# PAB two-body com canal de broadcasting ruidoso
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from itertools import product

# ============================================================
# 1. Escolha da faceta
# ============================================================

CLASS_TO_CHECK = 3

# Se class_df existir no notebook, ele usa automaticamente a classe 3.
# Caso contrário, usa manualmente o representante da classe 3 que apareceu no seu print.
MANUAL_ROW_CLASS_3 = [
    8, -3, -1, -1, 1, 0, 2, 2, 0, 3, -1, -1, -1
]

if "class_df" in globals():
    row_info = class_df.loc[class_df["class_id"] == CLASS_TO_CHECK].iloc[0]
    row = np.array(row_info["representative_row"], dtype=float)
else:
    row = np.array(MANUAL_ROW_CLASS_3, dtype=float)

b = row[0]
a = row[1:]

print("=" * 90)
print(f"Classe analisada: {CLASS_TO_CHECK}")
print("=" * 90)
print(f"Faceta:")
print(row.astype(int).tolist())
print()
print(f"b = {b}")
print(f"a = {a.astype(int).tolist()}")
print()

# Ordem dos correladores
COORD_NAMES = [
    "<B0C0>^[x=0]",
    "<B0C1>^[x=0]",
    "<B1C0>^[x=0]",
    "<B1C1>^[x=0]",
    "<B0C0>^[x=1]",
    "<B0C1>^[x=1]",
    "<B1C0>^[x=1]",
    "<B1C1>^[x=1]",
    "<B0C0>^[x=2]",
    "<B0C1>^[x=2]",
    "<B1C0>^[x=2]",
    "<B1C1>^[x=2]",
]

print("Ordem dos correladores:")
for i, name in enumerate(COORD_NAMES):
    print(f"{i:2d}: {name}")
print()


# ============================================================
# 2. Matrizes de Pauli
# ============================================================

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ============================================================
# 3. Estados, observáveis e canal
# ============================================================

def bloch_vec(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=float)


def qubit_state(theta, phi):
    """
    Estado puro de qubit:

        rho = (I + r.sigma)/2.
    """
    r = bloch_vec(theta, phi)
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def observable(theta, phi):
    """
    Observável projetivo binário:

        O = n.sigma,

    com autovalores +-1.
    """
    n = bloch_vec(theta, phi)
    return n[0] * SX + n[1] * SY + n[2] * SZ


def bell_beta_isometry(beta):
    """
    Isometria Bell-beta:

        V: C^2 -> C^2 tensor C^2.

    |0> -> |psi0(beta)>
    |1> -> |psi1(beta)>

    com

    |psi0> = sin(beta)|phi-> + cos(beta)|psi+>
    |psi1> = -cos(beta)|phi-> + sin(beta)|psi+>.

    Base computacional:

        |00>, |01>, |10>, |11>.
    """
    ket00 = np.array([1, 0, 0, 0], dtype=complex)
    ket01 = np.array([0, 1, 0, 0], dtype=complex)
    ket10 = np.array([0, 0, 1, 0], dtype=complex)
    ket11 = np.array([0, 0, 0, 1], dtype=complex)

    phi_minus = (ket00 - ket11) / np.sqrt(2)
    psi_plus  = (ket01 + ket10) / np.sqrt(2)

    psi0 = np.sin(beta) * phi_minus + np.cos(beta) * psi_plus
    psi1 = -np.cos(beta) * phi_minus + np.sin(beta) * psi_plus

    V = np.column_stack([psi0, psi1])

    return V


def broadcast_state(rho, beta, gamma=1.0):
    """
    Canal de broadcasting ruidoso:

        Phi_gamma(rho)
        =
        gamma V rho V^dagger
        +
        (1-gamma) I_4/4.

    gamma = 1: canal limpo.
    gamma = 0: saída totalmente branca.
    """
    V = bell_beta_isometry(beta)
    tau_clean = V @ rho @ V.conj().T
    tau_noisy = gamma * tau_clean + (1.0 - gamma) * I4 / 4.0
    return tau_noisy


# ============================================================
# 4. Parametrização do comportamento quântico
# ============================================================

def unpack_params(z):
    """
    Parâmetros otimizados:

    z[0] = beta do canal.

    Preparações:
        x=0: theta_x0, phi_x0
        x=1: theta_x1, phi_x1
        x=2: theta_x2, phi_x2

    Bob:
        y=0: theta_B0, phi_B0
        y=1: theta_B1, phi_B1

    Charlie:
        z=0: theta_C0, phi_C0
        z=1: theta_C1, phi_C1

    Total:
        1 + 3*2 + 2*2 + 2*2 = 15 parâmetros.
    """
    beta = z[0]
    pos = 1

    prep_angles = []
    for _ in range(3):
        prep_angles.append((z[pos], z[pos + 1]))
        pos += 2

    B_angles = []
    for _ in range(2):
        B_angles.append((z[pos], z[pos + 1]))
        pos += 2

    C_angles = []
    for _ in range(2):
        C_angles.append((z[pos], z[pos + 1]))
        pos += 2

    return beta, prep_angles, B_angles, C_angles


def quantum_twobody_vector(z, gamma=1.0):
    """
    Vetor de correladores two-body em R^12.

    Ordem:

        x=0: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=1: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=2: <B0C0>, <B0C1>, <B1C0>, <B1C1>.
    """
    beta, prep_angles, B_angles, C_angles = unpack_params(z)

    B_obs = [observable(*ang) for ang in B_angles]
    C_obs = [observable(*ang) for ang in C_angles]

    BC_obs = [
        [np.kron(B_obs[y], C_obs[zc]) for zc in range(2)]
        for y in range(2)
    ]

    vals = []

    for x in range(3):
        rho_x = qubit_state(*prep_angles[x])
        tau_x = broadcast_state(rho_x, beta, gamma=gamma)

        for y in range(2):
            for zc in range(2):
                val = np.trace(tau_x @ BC_obs[y][zc]).real
                vals.append(float(val))

    return np.array(vals, dtype=float)


def facet_value(row, v):
    """
    Avalia a faceta:

        F(v) = b + a.v.
    """
    return float(row[0] + np.dot(row[1:], v))


# ============================================================
# 5. Vértices clássicos para validação da faceta
# ============================================================

def generate_classical_twobody_vertices():
    """
    Gera os vértices clássicos-produto do modelo PAB two-body:

        x -> m = F(x),
        b = h_B(y,m),
        c = h_C(z,m).

    Na projeção two-body:

        <B_y C_z>^[x]
        =
        (-1)^h_B(y,F(x)) (-1)^h_C(z,F(x)).
    """
    X_SIZE = 3
    Y_SIZE = 2
    Z_SIZE = 2
    M_SIZE = 2

    vertices = set()

    all_F = list(product(range(M_SIZE), repeat=X_SIZE))
    all_hB = list(product(range(2), repeat=Y_SIZE * M_SIZE))
    all_hC = list(product(range(2), repeat=Z_SIZE * M_SIZE))

    for F in all_F:
        for hB in all_hB:
            for hC in all_hC:
                v = []

                for x in range(X_SIZE):
                    m = F[x]

                    for y in range(Y_SIZE):
                        bsign = (-1) ** hB[y * M_SIZE + m]

                        for zc in range(Z_SIZE):
                            csign = (-1) ** hC[zc * M_SIZE + m]
                            v.append(bsign * csign)

                vertices.add(tuple(v))

    return np.array(sorted(vertices), dtype=float)


vertices_check = generate_classical_twobody_vertices()
vals_classical = b + vertices_check @ a

print("=" * 90)
print("Validação clássica da faceta")
print("=" * 90)
print(f"Número de vértices clássicos two-body: {len(vertices_check)}")
print(f"min F(v_classical) = {vals_classical.min():.12f}")
print(f"max F(v_classical) = {vals_classical.max():.12f}")
print(f"número de vértices saturados = {np.sum(np.isclose(vals_classical, 0.0))}")
print(f"todos satisfazem F >= 0? {np.all(vals_classical >= -1e-9)}")
print()


# ============================================================
# 6. Otimização apenas dessa faceta
# ============================================================

# Bounds:
# beta em [0, pi]
# theta em [0, pi]
# phi em [0, 2pi]

BOUNDS = []

# beta
BOUNDS.append((0.0, np.pi))

# 3 preparações
for _ in range(3):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2.0 * np.pi))

# 2 medições de Bob
for _ in range(2):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2.0 * np.pi))

# 2 medições de Charlie
for _ in range(2):
    BOUNDS.append((0.0, np.pi))
    BOUNDS.append((0.0, 2.0 * np.pi))


def objective_clean(z):
    """
    Objetivo para canal limpo:

        minimizar F_clean = b + a.v_clean.
    """
    v = quantum_twobody_vector(z, gamma=1.0)
    return facet_value(row, v)


# Parâmetros da busca.
# Aumente N_RUNS, DE_MAXITER e DE_POPSIZE se quiser uma busca mais pesada.
N_RUNS = 8
DE_MAXITER = 900
DE_POPSIZE = 20

best = None

print("=" * 90)
print("Busca de violação quântica da classe 3")
print("=" * 90)

for run in range(N_RUNS):
    seed = 12345 + 1000 * run

    print(f"Run {run + 1}/{N_RUNS}, seed={seed}")

    res_de = differential_evolution(
        objective_clean,
        bounds=BOUNDS,
        seed=seed,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=1e-8,
        polish=False,
        updating="immediate",
        workers=1,
    )

    # Refinamento local com bounds.
    res_loc = minimize(
        objective_clean,
        res_de.x,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={
            "maxiter": 5000,
            "ftol": 1e-14,
            "gtol": 1e-10,
        },
    )

    for label, res in [("DE", res_de), ("LOCAL", res_loc)]:
        val = float(res.fun)

        if best is None or val < best["F_min_clean"]:
            best = {
                "F_min_clean": val,
                "params": np.array(res.x, dtype=float),
                "method": label,
                "seed": seed,
                "success": bool(res.success),
                "message": str(res.message),
            }

    print(f"  melhor parcial: F_clean = {best['F_min_clean']:.12f}")

print()

zbest = best["params"]
F_min_clean = best["F_min_clean"]
v_clean = quantum_twobody_vector(zbest, gamma=1.0)

# Limiar de ruído branco no canal.
# Como v_gamma = gamma v_clean, temos:
#
# F_gamma = b + gamma (F_clean - b).
#
# A violação some em F_gamma = 0.
if F_min_clean < 0 and b > 0:
    gamma_crit = b / (b - F_min_clean)
elif F_min_clean < 0 and abs(b) < 1e-12:
    gamma_crit = 0.0
else:
    gamma_crit = np.nan

print("=" * 90)
print("Resultado ótimo encontrado")
print("=" * 90)
print(f"F_min_clean = {F_min_clean:.15f}")
print(f"violação limpa = {max(0.0, -F_min_clean):.15f}")
print(f"gamma_crit = {gamma_crit:.15f}")
print(f"método do melhor ponto = {best['method']}")
print(f"seed do melhor ponto = {best['seed']}")
print(f"beta ótimo = {zbest[0]:.15f} rad")
print(f"beta ótimo = {180.0 * zbest[0] / np.pi:.12f} graus")
print()


# ============================================================
# 7. Tabela dos correladores e contribuições da faceta
# ============================================================

contribs = a * v_clean

corr_df = pd.DataFrame({
    "index": np.arange(12),
    "correlator": COORD_NAMES,
    "coefficient": a.astype(int),
    "value_clean": v_clean,
    "term_a_times_value": contribs,
})

print("=" * 90)
print("Correladores ótimos para gamma=1")
print("=" * 90)
display(corr_df)

print(f"Soma dos termos a.v = {np.dot(a, v_clean):.15f}")
print(f"b + a.v = {b + np.dot(a, v_clean):.15f}")
print()

print("Checagem dos correladores:")
print(f"min correlador = {v_clean.min():.15f}")
print(f"max correlador = {v_clean.max():.15f}")
print(f"todos em [-1,1]? {np.all(np.abs(v_clean) <= 1 + 1e-9)}")
print()


# ============================================================
# 8. Checagem explícita do ruído branco no canal
# ============================================================

print("=" * 90)
print("Checagem do ruído branco no canal")
print("=" * 90)

gammas_to_check = [
    1.0,
    0.9,
    0.8,
    float(gamma_crit) if np.isfinite(gamma_crit) else 0.5,
    0.6,
    0.5,
    0.0,
]

noise_rows = []

for gamma in gammas_to_check:
    v_gamma = quantum_twobody_vector(zbest, gamma=gamma)

    F_direct = facet_value(row, v_gamma)
    F_scaling = b + gamma * (F_min_clean - b)

    noise_rows.append({
        "gamma": gamma,
        "F_direct_from_noisy_channel": F_direct,
        "F_from_scaling_formula": F_scaling,
        "difference": F_direct - F_scaling,
        "violates": F_direct < -1e-8,
    })

noise_df = pd.DataFrame(noise_rows)
display(noise_df)

print()
print("No ponto gamma = gamma_crit, F deve ficar aproximadamente zero.")
print()


# ============================================================
# 9. Salvar resultado da classe 3
# ============================================================

result_single = {
    "class_id": CLASS_TO_CHECK,
    "representative_row": row.astype(int).tolist(),
    "F_min_clean": float(F_min_clean),
    "violation_clean": float(max(0.0, -F_min_clean)),
    "gamma_crit": float(gamma_crit) if np.isfinite(gamma_crit) else None,
    "beta_opt_rad": float(zbest[0]),
    "beta_opt_deg": float(180.0 * zbest[0] / np.pi),
    "params": zbest.tolist(),
    "v_clean": v_clean.tolist(),
}

pd.DataFrame([result_single]).to_csv(
    "violacao_classe_3_pab_canal_ruidoso.csv",
    index=False,
)

corr_df.to_csv(
    "correladores_classe_3_pab_canal_ruidoso.csv",
    index=False,
)

noise_df.to_csv(
    "ruido_classe_3_pab_canal_ruidoso.csv",
    index=False,
)

print("Arquivos salvos:")
print("  violacao_classe_3_pab_canal_ruidoso.csv")
print("  correladores_classe_3_pab_canal_ruidoso.csv")
print("  ruido_classe_3_pab_canal_ruidoso.csv")
```

