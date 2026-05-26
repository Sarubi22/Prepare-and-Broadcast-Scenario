# 



---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera os vertices do subpolitopo two-body do modelo classico produto
PAM-broadcasting para:

    |X| = 3
    |Y| = 2
    |Z| = 2
    |M| = 2
    b,c in {0,1}

Modelo deterministico:
    F   : X -> {0,1}
    h_B : Y x {0,1} -> {0,1}
    h_C : Z x {0,1} -> {0,1}

Aqui projetamos apenas nas coordenadas two-body:
    para cada x in {0,1,2}:
        bloco_x = [ <B0C0>[x], <B0C1>[x], <B1C0>[x], <B1C1>[x] ]

Logo, o vetor esta em R^12.

Saida:
    arquivo .ext no formato V-representation do lrs
"""

import os
from itertools import product

# ------------------------------------------------------------
# Parametros do cenario
# ------------------------------------------------------------
X_SIZE = 3
Y_SIZE = 2
Z_SIZE = 2
M_SIZE = 2

BLOCK = Y_SIZE * Z_SIZE   # 4 coordenadas por x
DIM = X_SIZE * BLOCK      # 12

out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
out_path = os.path.join(out_dir, "pab_classical_twobody_vertices_X3Y2Z2M2.ext")


# ------------------------------------------------------------
# Indice dentro do bloco de cada preparacao x
# Ordem:
#   <B0C0>, <B0C1>, <B1C0>, <B1C1>
# ------------------------------------------------------------
def idx_ByCz(y, z):
    return y * Z_SIZE + z   # 0,1,2,3


# ------------------------------------------------------------
# Gera vertices unicos do subpolitopo two-body
# ------------------------------------------------------------
def generate_vertices():
    vertices = set()

    all_F = product(range(M_SIZE), repeat=X_SIZE)                 # 2^3 = 8
    all_hB = product(range(2), repeat=Y_SIZE * M_SIZE)           # 2^4 = 16
    all_hC = product(range(2), repeat=Z_SIZE * M_SIZE)           # 2^4 = 16

    for F in all_F:
        for h_B in all_hB:
            for h_C in all_hC:

                v = []

                for x in range(X_SIZE):
                    m = F[x]
                    bloco = [0] * BLOCK

                    for y in range(Y_SIZE):
                        b_sign = (-1) ** h_B[y * M_SIZE + m]
                        for z in range(Z_SIZE):
                            c_sign = (-1) ** h_C[z * M_SIZE + m]
                            bloco[idx_ByCz(y, z)] = b_sign * c_sign

                    v.extend(bloco)

                vertices.add(tuple(v))

    return sorted(vertices)


# ------------------------------------------------------------
# Escreve V-representation para o lrs
# ------------------------------------------------------------
def write_vrep(filepath, vertices):
    m = len(vertices)
    n = DIM + 1   # coordenada homogenia + 12 coordenadas

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="ascii", newline="\n") as f:
        f.write("* PAB classical product model - two-body projection\n")
        f.write(f"* |X|={X_SIZE}, |Y|={Y_SIZE}, |Z|={Z_SIZE}, |M|={M_SIZE}\n")
        f.write(f"* Dimension = {DIM}\n")
        f.write("* Coordinates per preparation x:\n")
        f.write("*   <B0C0>[x], <B0C1>[x], <B1C0>[x], <B1C1>[x]\n")
        f.write("* Blocks concatenated for x = 0,1,2\n")
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{m} {n} rational\n")

        for v in vertices:
            row = ["1"] + [str(int(c)) for c in v]
            f.write(" ".join(row) + "\n")

        f.write("end\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Gerando vertices do subpolitopo two-body...")
    print(f"  |X|={X_SIZE}, |Y|={Y_SIZE}, |Z|={Z_SIZE}, |M|={M_SIZE}")

    total_candidates = (M_SIZE ** X_SIZE) * (2 ** (Y_SIZE * M_SIZE)) * (2 ** (Z_SIZE * M_SIZE))
    print(f"  Candidatos totais: {total_candidates}")

    vertices = generate_vertices()

    print(f"  Vertices unicos apos remocao de duplicatas: {len(vertices)}")
    print(f"  Dimensao do vetor: {DIM}")

    write_vrep(out_path, vertices)

    print(f"\nArquivo salvo em:\n  {out_path}")
    print("\nFormato: V-representation do lrs")
    print(f"Linhas: 1 coordenada homogenia + {DIM} coordenadas two-body")


if __name__ == "__main__":
    main()
```


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Busca violacoes quanticas das facetas do politopo PAB classico produto.
 
Cenario: |X|=3, |Y|=2, |Z|=2, |M|=2, saidas binarias.
 
Ordem dos correladores no vetor (24 coordenadas), por bloco x:
    x=0: <B0>[0], <B1>[0], <C0>[0], <C1>[0],
         <B0C0>[0], <B0C1>[0], <B1C0>[0], <B1C1>[0]
    x=1: idem
    x=2: idem
 
Canal quantico: isometria geral V(p): H_A -> H_B x H_C
    U(p) = exp(-i H(p)),  H(p) = sum_{k=1}^{15} p_k Lambda_k
    V(p) = U(p)[:, 0:2]   (primeiras 2 colunas = embedding do qubit)
 
Parametros de otimizacao (29 no total):
    p in R^15          canal (geradores SU(4))
    (theta, phi) x 3   vetores de Bloch de Alice r_x
    (theta, phi) x 2   medicoes de Bob b_y
    (theta, phi) x 2   medicoes de Charlie c_z
 
Para cada faceta lrs (formato: RHS a1 a2 ... a24):
    W = sum_j a_j * v_j   (produto escalar com vetor de correladores)
    Bound classico: W <= RHS
    Busca quantica: maximiza W sobre os 29 parametros
 
Saida:
    - Tabela impressa no terminal/Jupyter
    - Arquivo CSV com resultados em C:/Users/tayla/Downloads/
"""
 
import os
import math
import csv
import numpy as np
from scipy.optimize import minimize
 
# ==============================================================
# Caminhos de entrada e saida (fixos para Jupyter)
# ==============================================================
FACETS_FILE = r"C:\Users\tayla\Downloads\pab_classical_facets_X3Y2Z2M2.ine"
OUT_CSV     = r"C:\Users\tayla\Downloads\pab_violation_results.csv"
 
# ==============================================================
# Parametros do cenario
# ==============================================================
X_SIZE = 3
Y_SIZE = 2
Z_SIZE = 2
BLOCK  = Y_SIZE + Z_SIZE + Y_SIZE * Z_SIZE  # 8 por preparacao
DIM    = BLOCK * X_SIZE                      # 24
 
def idx_By(y):      return y
def idx_Cz(z):      return Y_SIZE + z
def idx_ByCz(y, z): return Y_SIZE + Z_SIZE + y * Z_SIZE + z
 
# ==============================================================
# Geradores de SU(4) - base de Gell-Mann generalizada (15 matrizes)
# ==============================================================
def gell_mann_su4():
    d = 4
    mats = []
    for j in range(d):
        for k in range(j + 1, d):
            M = np.zeros((d, d), complex)
            M[j, k] = 1.0
            M[k, j] = 1.0
            mats.append(M)
    for j in range(d):
        for k in range(j + 1, d):
            M = np.zeros((d, d), complex)
            M[j, k] = -1j
            M[k, j] = +1j
            mats.append(M)
    for l in range(1, d):
        M = np.zeros((d, d), complex)
        for j in range(l):
            M[j, j] = 1.0
        M[l, l] = -float(l)
        norm = math.sqrt(2.0 / (l * (l + 1)))
        mats.append(M * norm)
    assert len(mats) == 15
    return mats
 
GENERATORS = gell_mann_su4()
 
# ==============================================================
# Operadores de Pauli
# ==============================================================
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,  1 ], [1,   0]], complex)
sy = np.array([[0, -1j], [1j,  0]], complex)
sz = np.array([[1,  0 ], [0,  -1]], complex)
PAULI = [sx, sy, sz]
 
# ==============================================================
# Canal de broadcasting: isometria geral V(p): C^2 -> C^4
# ==============================================================
def build_isometry(p):
    H = np.zeros((4, 4), complex)
    for k in range(15):
        H += float(p[k]) * GENERATORS[k]
    eigvals, eigvecs = np.linalg.eigh(H)
    U = eigvecs @ np.diag(np.exp(-1j * eigvals)) @ eigvecs.conj().T
    return U[:, 0:2]
 
# ==============================================================
# Vetor de Bloch -> vetor unitario em R^3
# ==============================================================
def sph_to_vec(theta, phi):
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta)
    ], dtype=float)
 
# ==============================================================
# Calculo dos correladores quanticos
# ==============================================================
def compute_correlators(V, r_list, bB, bC):
    B_ops = [sum(bB[y][k] * PAULI[k] for k in range(3)) for y in range(Y_SIZE)]
    C_ops = [sum(bC[z][k] * PAULI[k] for k in range(3)) for z in range(Z_SIZE)]
    corr  = np.zeros(DIM, float)
    for x in range(X_SIZE):
        r      = r_list[x]
        rho    = 0.5 * (I2 + sum(r[k] * PAULI[k] for k in range(3)))
        tau    = V @ rho @ V.conj().T
        offset = x * BLOCK
        for y in range(Y_SIZE):
            op = np.kron(B_ops[y], I2)
            corr[offset + idx_By(y)] = float(np.trace(tau @ op).real)
        for z in range(Z_SIZE):
            op = np.kron(I2, C_ops[z])
            corr[offset + idx_Cz(z)] = float(np.trace(tau @ op).real)
        for y in range(Y_SIZE):
            for z in range(Z_SIZE):
                op = np.kron(B_ops[y], C_ops[z])
                corr[offset + idx_ByCz(y, z)] = float(np.trace(tau @ op).real)
    return corr
 
# ==============================================================
# Desempacotamento dos 29 parametros
# ==============================================================
def unpack_params(z):
    p   = z[0:15]
    r0  = sph_to_vec(z[15], z[16])
    r1  = sph_to_vec(z[17], z[18])
    r2  = sph_to_vec(z[19], z[20])
    bB0 = sph_to_vec(z[21], z[22])
    bB1 = sph_to_vec(z[23], z[24])
    bC0 = sph_to_vec(z[25], z[26])
    bC1 = sph_to_vec(z[27], z[28])
    return p, [r0, r1, r2], [bB0, bB1], [bC0, bC1]
 
# ==============================================================
# Ponto de partida aleatorio
# ==============================================================
def random_start(rng):
    z = np.zeros(29, float)
    z[0:15] = rng.uniform(-math.pi, math.pi, 15)
    idx = 15
    for _ in range(7):
        z[idx]     = rng.uniform(0.0, math.pi)
        z[idx + 1] = rng.uniform(0.0, 2.0 * math.pi)
        idx += 2
    return z
 
# ==============================================================
# Bounds para SLSQP
# ==============================================================
BOUNDS = (
    [(-math.pi, math.pi)] * 15 +
    [(0.0, math.pi), (0.0, 2 * math.pi)] * 7
)
 
# ==============================================================
# Funcao objetivo
# ==============================================================
def make_objective(coeffs):
    def objective(z):
        try:
            p, r_list, bB, bC = unpack_params(z)
            V    = build_isometry(p)
            corr = compute_correlators(V, r_list, bB, bC)
            return -float(np.dot(coeffs, corr))
        except Exception:
            return 0.0
    return objective
 
# ==============================================================
# Otimizacao para uma faceta
# ==============================================================
def optimize_facet(coeffs, rhs, restarts=100, maxiter=1000, seed=0, verbose=False):
    rng      = np.random.default_rng(seed)
    obj      = make_objective(coeffs)
    best_val = -1e300
    best_z   = None
    for k in range(restarts):
        z0  = random_start(rng)
        res = minimize(obj, z0, method='SLSQP', bounds=BOUNDS,
                       options={'maxiter': maxiter, 'ftol': 1e-11, 'disp': False})
        if not res.success:
            res = minimize(obj, res.x, method='SLSQP', bounds=BOUNDS,
                           options={'maxiter': maxiter, 'ftol': 1e-11, 'disp': False})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_z   = res.x.copy()
        if verbose and (k + 1) % 10 == 0:
            print(f"    restart {k+1:3d}/{restarts}  best W = {best_val:.8f}")
    violation = best_val - rhs
    return {
        'rhs'       : rhs,
        'W_quantum' : best_val,
        'violation' : violation,
        'violates'  : violation > 1e-6,
        'z_opt'     : best_z
    }
 
# ==============================================================
# Leitura das facetas no formato lrs
# ==============================================================
def parse_facets(filepath):
    facets = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('*'):
                continue
            low = s.lower()
            if low in ('h-representation', 'v-representation',
                       'begin', 'end', 'linearity'):
                continue
            parts = s.split()
            if len(parts) != DIM + 1:
                continue
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            rhs    = nums[0]
            coeffs = np.array(nums[1:], dtype=float)
            facets.append((rhs, coeffs))
    return facets
 
# ==============================================================
# Relatorio
# ==============================================================
def report_facet(idx, result):
    status = "*** VIOLA ***" if result['violates'] else "ok"
    print(f"  Faceta {idx:3d} | "
          f"RHS = {result['rhs']:6.2f} | "
          f"W_Q = {result['W_quantum']:10.6f} | "
          f"margem = {result['violation']:+10.6f} | "
          f"{status}")
 
def print_optimal_details(idx, result):
    if not result['violates']:
        return
    z = result['z_opt']
    p, r_list, bB, bC = unpack_params(z)
    print(f"\n  --- Detalhes da violacao (Faceta {idx}) ---")
    print(f"  W_quantum = {result['W_quantum']:.9f}  >  RHS = {result['rhs']:.2f}")
    for i, r in enumerate(r_list):
        print(f"  r_{i} = [{r[0]:+.6f}, {r[1]:+.6f}, {r[2]:+.6f}]")
    for y, b in enumerate(bB):
        print(f"  b_{y} = [{b[0]:+.6f}, {b[1]:+.6f}, {b[2]:+.6f}]")
    for zi, c in enumerate(bC):
        print(f"  c_{zi} = [{c[0]:+.6f}, {c[1]:+.6f}, {c[2]:+.6f}]")
    print(f"  p (canal) = {np.round(p, 6).tolist()}")
 
# ==============================================================
# Main
# ==============================================================
def main():
 
    RESTARTS = 100
    MAXITER  = 1000
    SEED     = 42
    VERBOSE  = False   # mude para True para ver progresso por restart
 
    print(f"Lendo facetas de:\n  {FACETS_FILE}\n")
    facets = parse_facets(FACETS_FILE)
    print(f"  {len(facets)} facetas encontradas.")
    print(f"  Dimensao esperada: {DIM} correladores por faceta.\n")
 
    if not facets:
        print("Nenhuma faceta encontrada. Verifique o arquivo.")
        return
 
    results = []
    print(f"Otimizando {len(facets)} facetas | "
          f"{RESTARTS} restarts | "
          f"canal SU(4) 15 parametros | "
          f"29 parametros totais")
    print("-" * 78)
 
    for i, (rhs, coeffs) in enumerate(facets):
        print(f"Faceta {i+1:3d}/{len(facets)} ...", end='', flush=True)
        result = optimize_facet(
            coeffs   = coeffs,
            rhs      = rhs,
            restarts = RESTARTS,
            maxiter  = MAXITER,
            seed     = SEED + i,
            verbose  = VERBOSE
        )
        results.append(result)
        print(f"\r", end='')
        report_facet(i + 1, result)
 
    print("-" * 78)
    n_viol = sum(1 for r in results if r['violates'])
    print(f"\nSUMARIO: {n_viol}/{len(facets)} facetas violadas quanticamente.\n")
 
    if n_viol > 0:
        print("Detalhes das facetas violadas:")
        for i, r in enumerate(results):
            if r['violates']:
                print_optimal_details(i + 1, r)
 
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['faceta', 'rhs', 'W_quantum', 'violacao', 'viola'])
        for i, r in enumerate(results):
            writer.writerow([
                i + 1,
                f"{r['rhs']:.4f}",
                f"{r['W_quantum']:.9f}",
                f"{r['violation']:.9f}",
                'SIM' if r['violates'] else 'NAO'
            ])
 
    print(f"\nResultados salvos em:\n  {OUT_CSV}")
 
 
if __name__ == "__main__":
    main()
```

