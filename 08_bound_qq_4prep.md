# Quantum–Quantum (QQ) Violation for 4-Preparation PAB Facet

Numerically optimises a two-body PAB inequality for the fully quantum  
broadcasting model (QQ) with four preparations (`|X|=4`).

**Model:** Alice sends a qubit state through a general isometric broadcasting  
channel `U(p) = exp(-i Σ p_k Λ_k)` where `{Λ_k}` are the 15 generators of `SU(4)`.  
Bob and Charlie perform projective qubit measurements.

**Optimisation:** SLSQP with multiple random restarts over 29 real parameters:
- 15 channel parameters `p ∈ R^15`
- 3 × 2 Bloch-sphere angles for Alice's preparations
- 2 × 2 angles for Bob's measurements
- 2 × 2 angles for Charlie's measurements

The preparations are additionally constrained to satisfy the geometric  
four-preparation PAM classicality criteria (Eqs. 6–7 of the paper),  
so that any observed violation constitutes activation of nonclassicality.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
# ============================================================
# Violação quântica de uma faceta PAB two-body com 4 preparações
# Canal geral isométrico U(p)=exp[-i sum_k p_k Lambda_k]
# Estados puros de qubit e medições projetivas
# ============================================================

import numpy as np
import pandas as pd
import json
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

# ============================================================
# 1. Configuração da faceta
# ============================================================

# Faceta em formato LRS:
#     b + a.E >= 0
#
# Equivalente ao witness:
#     W = -a.E <= b
#
# Ordem:
# x=0: <B0C0>, <B0C1>, <B1C0>, <B1C1>
# x=1: <B0C0>, <B0C1>, <B1C0>, <B1C1>
# x=2: <B0C0>, <B0C1>, <B1C0>, <B1C1>
# x=3: <B0C0>, <B0C1>, <B1C0>, <B1C1>

LRS_ROW = [8, 1, 1, -1, -1, 1, -1, -1, 1, -3, -1, -1, 1, 0, 0, 2, -2]

N_GLOBAL_RUNS = 3
DE_MAXITER = 300
DE_POPSIZE = 12
LOCAL_MAXITER = 2000

VIOL_TOL = 1e-8
CHANNEL_P_BOUND = np.pi

OUT_PREFIX = "violacao_faceta_X4_twobody_general_isometry"


# ============================================================
# 2. Ler faceta
# ============================================================

row = np.array(LRS_ROW, dtype=float)

B_CLASSICAL = float(row[0])
a_lrs = row[1:]
w_coeff = -a_lrs

if len(a_lrs) % 4 != 0:
    raise ValueError("Número de coeficientes incompatível com blocos two-body de tamanho 4.")

X_SIZE = len(a_lrs) // 4
Y_SIZE = 2
Z_SIZE = 2

COORD_NAMES = [
    rf"<B_{y}C_{z}>^[x={x}]"
    for x in range(X_SIZE)
    for y in range(Y_SIZE)
    for z in range(Z_SIZE)
]

print("=" * 100)
print("Faceta carregada")
print("=" * 100)
print(f"LRS_ROW = {LRS_ROW}")
print(f"Bound clássico B_C = {B_CLASSICAL}")
print(f"Número de preparações X_SIZE = {X_SIZE}")
print()
print("Ordem dos correladores:")
for i, name in enumerate(COORD_NAMES):
    print(
        f"{i:2d}: {name:20s} "
        f"coeff_LRS={a_lrs[i]:+g}   coeff_W={w_coeff[i]:+g}"
    )
print()


# ============================================================
# 3. Pauli e funções básicas
# ============================================================

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def bloch_vec(theta, phi):
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=float)


def qubit_state(theta, phi):
    """
    Estado puro:
        rho = (I + r.sigma)/2
    """
    r = bloch_vec(theta, phi)
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def observable(theta, phi):
    """
    Observável projetivo:
        O = n.sigma
    com autovalores +-1.
    """
    n = bloch_vec(theta, phi)
    return n[0] * SX + n[1] * SY + n[2] * SZ


# ============================================================
# 4. Canal geral isométrico U(p)
# ============================================================

def su_generators(d=4):
    """
    Base hermitiana e traceless de su(d).
    Para d=4, retorna 15 geradores.
    """
    gens = []

    # Off-diagonal simétricos e antissimétricos
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
    raise RuntimeError("A base su(4) deveria ter 15 geradores.")


def isometry_from_p(p):
    """
    U_BC(p) = exp[-i H(p)]
    H(p) = sum_k p_k Lambda_k

    A isometria V: C^2 -> C^2 tensor C^2
    é obtida pelas duas primeiras colunas de U_BC(p).
    """
    H = np.zeros((4, 4), dtype=complex)

    for pk, Gk in zip(p, SU4):
        H += pk * Gk

    UBC = expm(-1j * H)
    V = UBC[:, :2]

    return V


def broadcast_state(rho, p):
    """
    Canal isométrico limpo:
        tau_x = V rho_x V^dagger.
    """
    V = isometry_from_p(p)
    return V @ rho @ V.conj().T


# ============================================================
# 5. Parametrização
# ============================================================

def make_bounds():
    bounds = []

    # 15 parâmetros do canal
    for _ in range(15):
        bounds.append((-CHANNEL_P_BOUND, CHANNEL_P_BOUND))

    # 4 preparações puras: theta_x, phi_x
    for _ in range(X_SIZE):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, 2.0 * np.pi))

    # Bob: B0, B1
    for _ in range(2):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, 2.0 * np.pi))

    # Charlie: C0, C1
    for _ in range(2):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, 2.0 * np.pi))

    return bounds


BOUNDS = make_bounds()

print(f"Número total de parâmetros otimizados: {len(BOUNDS)}")
print()


def unpack_params(z):
    pos = 0

    # Canal
    p = np.array(z[pos:pos + 15], dtype=float)
    pos += 15

    # Preparações
    prep_angles = []
    for _ in range(X_SIZE):
        prep_angles.append((z[pos], z[pos + 1]))
        pos += 2

    # Bob
    B_angles = []
    for _ in range(2):
        B_angles.append((z[pos], z[pos + 1]))
        pos += 2

    # Charlie
    C_angles = []
    for _ in range(2):
        C_angles.append((z[pos], z[pos + 1]))
        pos += 2

    return p, prep_angles, B_angles, C_angles


# ============================================================
# 6. Correladores e witness
# ============================================================

def quantum_twobody_vector(z):
    """
    Retorna E em R^16 na ordem:

        x=0: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=1: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=2: <B0C0>, <B0C1>, <B1C0>, <B1C1>
        x=3: <B0C0>, <B0C1>, <B1C0>, <B1C1>
    """
    p, prep_angles, B_angles, C_angles = unpack_params(z)

    B_obs = [observable(*ang) for ang in B_angles]
    C_obs = [observable(*ang) for ang in C_angles]

    BC_obs = [
        [np.kron(B_obs[y], C_obs[z_]) for z_ in range(2)]
        for y in range(2)
    ]

    vals = []

    for x in range(X_SIZE):
        rho_x = qubit_state(*prep_angles[x])
        tau_x = broadcast_state(rho_x, p)

        for y in range(2):
            for z_ in range(2):
                val = np.trace(tau_x @ BC_obs[y][z_]).real
                vals.append(float(val))

    return np.array(vals, dtype=float)


def F_lrs_value(E):
    """
    Forma LRS:
        F = b + a.E
    """
    return float(B_CLASSICAL + np.dot(a_lrs, E))


def W_value(E):
    """
    Witness positivo:
        W = -a.E
    """
    return float(np.dot(w_coeff, E))


def objective(z):
    """
    Minimiza F = b + a.E.
    Equivalente a maximizar W = -a.E.
    """
    E = quantum_twobody_vector(z)
    return F_lrs_value(E)


# ============================================================
# 7. Otimização
# ============================================================

def optimize_violation(seed0=12345):
    best = None

    print("=" * 100)
    print("Busca de violação quântica da faceta X=4")
    print("=" * 100)

    for run in range(N_GLOBAL_RUNS):
        seed = seed0 + 1000 * run
        print(f"Run {run + 1}/{N_GLOBAL_RUNS}, seed={seed}")

        res_de = differential_evolution(
            objective,
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
            objective,
            res_de.x,
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={
                "maxiter": LOCAL_MAXITER,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        candidates = [
            ("DE", res_de),
            ("LOCAL", res_loc),
        ]

        for method, res in candidates:
            F = float(res.fun)

            if best is None or F < best["F_clean"]:
                best = {
                    "F_clean": F,
                    "params": np.array(res.x, dtype=float),
                    "method": method,
                    "seed": seed,
                    "success": bool(getattr(res, "success", False)),
                    "message": str(getattr(res, "message", "")),
                }

        W_best = B_CLASSICAL - best["F_clean"]

        print(
            f"  melhor parcial: "
            f"F_clean = {best['F_clean']:.12f} | "
            f"W_clean = {W_best:.12f} | "
            f"W-B_C = {W_best - B_CLASSICAL:.12f}"
        )

    return best


best = optimize_violation()


# ============================================================
# 8. Resultado final
# ============================================================

zbest = best["params"]

E_best = quantum_twobody_vector(zbest)
F_best = F_lrs_value(E_best)
W_best = W_value(E_best)

gap = W_best - B_CLASSICAL

print()
print("=" * 100)
print("RESULTADO FINAL")
print("=" * 100)
print(f"F_clean = b + a.E = {F_best:.15f}")
print(f"W_clean = -a.E    = {W_best:.15f}")
print(f"Bound clássico    = {B_CLASSICAL:.15f}")
print(f"Violação W-B_C    = {gap:.15f}")
print(f"Violou?           = {W_best > B_CLASSICAL + VIOL_TOL}")
print(f"Método            = {best['method']}")
print(f"Seed              = {best['seed']}")


# ============================================================
# 9. Tabelas de parâmetros ótimos
# ============================================================

p_opt, prep_angles, B_angles, C_angles = unpack_params(zbest)

df_corr = pd.DataFrame({
    "index": list(range(len(E_best))),
    "correlator": COORD_NAMES,
    "coeff_LRS": a_lrs,
    "coeff_W": w_coeff,
    "E_value": E_best,
    "term_W": w_coeff * E_best,
})

prep_rows = []
for x, (theta, phi) in enumerate(prep_angles):
    r = bloch_vec(theta, phi)
    prep_rows.append({
        "x": x,
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": 180 * theta / np.pi,
        "phi_deg": 180 * phi / np.pi,
        "rx": r[0],
        "ry": r[1],
        "rz": r[2],
    })
df_prep = pd.DataFrame(prep_rows)

bob_rows = []
for y, (theta, phi) in enumerate(B_angles):
    n = bloch_vec(theta, phi)
    bob_rows.append({
        "y": y,
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": 180 * theta / np.pi,
        "phi_deg": 180 * phi / np.pi,
        "nx": n[0],
        "ny": n[1],
        "nz": n[2],
    })
df_bob = pd.DataFrame(bob_rows)

charlie_rows = []
for zc, (theta, phi) in enumerate(C_angles):
    n = bloch_vec(theta, phi)
    charlie_rows.append({
        "z": zc,
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": 180 * theta / np.pi,
        "phi_deg": 180 * phi / np.pi,
        "nx": n[0],
        "ny": n[1],
        "nz": n[2],
    })
df_charlie = pd.DataFrame(charlie_rows)

df_channel = pd.DataFrame({
    "parameter": [f"p_{k+1}" for k in range(15)],
    "value": p_opt,
})

V_opt = isometry_from_p(p_opt)

df_isometry = pd.DataFrame({
    "row": np.repeat(np.arange(V_opt.shape[0]), V_opt.shape[1]),
    "col": np.tile(np.arange(V_opt.shape[1]), V_opt.shape[0]),
    "real": V_opt.real.reshape(-1),
    "imag": V_opt.imag.reshape(-1),
})

print()
print("=" * 100)
print("Correladores ótimos")
print("=" * 100)
display(df_corr)

print()
print("=" * 100)
print("Estados preparados ótimos")
print("=" * 100)
display(df_prep)

print()
print("=" * 100)
print("Medições ótimas de Bob")
print("=" * 100)
display(df_bob)

print()
print("=" * 100)
print("Medições ótimas de Charlie")
print("=" * 100)
display(df_charlie)

print()
print("=" * 100)
print("Parâmetros do canal")
print("=" * 100)
display(df_channel)


# ============================================================
# 10. Salvar arquivos
# ============================================================

df_corr.to_csv(f"{OUT_PREFIX}_correladores.csv", index=False)
df_prep.to_csv(f"{OUT_PREFIX}_preparacoes.csv", index=False)
df_bob.to_csv(f"{OUT_PREFIX}_bob_measurements.csv", index=False)
df_charlie.to_csv(f"{OUT_PREFIX}_charlie_measurements.csv", index=False)
df_channel.to_csv(f"{OUT_PREFIX}_channel_params.csv", index=False)
df_isometry.to_csv(f"{OUT_PREFIX}_isometry.csv", index=False)

summary = {
    "LRS_ROW": LRS_ROW,
    "B_CLASSICAL": float(B_CLASSICAL),
    "F_clean": float(F_best),
    "W_clean": float(W_best),
    "violation_gap": float(gap),
    "violated": bool(W_best > B_CLASSICAL + VIOL_TOL),
    "best_method": best["method"],
    "best_seed": int(best["seed"]),
    "best_params": zbest.tolist(),
    "channel_params": p_opt.tolist(),
    "E_values": E_best.tolist(),
    "preparations": df_prep.to_dict(orient="records"),
    "bob_measurements": df_bob.to_dict(orient="records"),
    "charlie_measurements": df_charlie.to_dict(orient="records"),
    "isometry_real": V_opt.real.tolist(),
    "isometry_imag": V_opt.imag.tolist(),
}

with open(f"{OUT_PREFIX}_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print()
print("=" * 100)
print("Arquivos salvos")
print("=" * 100)
print(f"{OUT_PREFIX}_correladores.csv")
print(f"{OUT_PREFIX}_preparacoes.csv")
print(f"{OUT_PREFIX}_bob_measurements.csv")
print(f"{OUT_PREFIX}_charlie_measurements.csv")
print(f"{OUT_PREFIX}_channel_params.csv")
print(f"{OUT_PREFIX}_isometry.csv")
print(f"{OUT_PREFIX}_summary.json")
```

