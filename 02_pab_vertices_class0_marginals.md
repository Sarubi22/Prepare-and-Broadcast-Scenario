# PAB Classical Vertices — Class-0 Projection with Marginals (18D)

Builds a projected vertex set for the PAB classical polytope, retaining only  
the marginal correlators and a specific subset of two-body correlators relevant to  
the Class-0 PAB inequality.

The 18-dimensional output vector keeps, for each preparation `x`, the four  
marginals `[B0, B1, C0, C1]` plus two selected two-body correlators:  
- `x=0`: `[E00, E01]`  
- `x=1`: `[E01, E11]`  
- `x=2`: `[E00, E11]`

**Input:** 16D NS-vertices file (`pam_broadcasting_verticesNS.ine.out`).  
**Output:** 18D V-representation file for `lrs`.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from fractions import Fraction

# ------------------------------------------------------------
# Caminhos: mantém o mesmo arquivo de entrada.
# (Mudei apenas o nome do arquivo de saída para não sobrescrever o anterior.)
# ------------------------------------------------------------
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
in_path  = os.path.join(downloads_path, "pam_broadcasting_verticesNS.ine.out")
out_path = os.path.join(os.path.expanduser("~"), "pam_vertices_class0_with_marginals.ext")

# ------------------------------------------------------------
# Utilitários de parsing (idênticos ao seu script)
# ------------------------------------------------------------
def parse_vrep_vertices_16(filepath):
    """
    Lê V-representation (lrs) com vértices em R^16.
    Considera apenas linhas que começam com '1' (homogêneo).
    Espera blocos por x no formato:
      [B0, B1, C0, C1, E00, E01, E10, E11]  |  m=0
      [B0, B1, C0, C1, E00, E01, E10, E11]  |  m=1
    concatenados -> total 16 coords por vértice.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    verts, in_body = [], False
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            low = line.lower()
            if low in ("v-representation", "h-representation"):
                continue
            if low == "begin":
                in_body = True
                continue
            if low == "end":
                break
            if not in_body:
                continue

            parts = line.split()
            # Só processa linhas de vértice homogêneo
            if parts[0] not in ("1", "1/1"):
                continue

            nums = []
            ok = True
            for p in parts:
                p = p.replace(",", "")
                try:
                    nums.append(float(Fraction(p))) if "/" in p else nums.append(float(p))
                except Exception:
                    ok = False
                    break
            if not ok or len(nums) < 17:
                continue

            vec16 = nums[1:17]  # remove coordenada homogênea
            if len(vec16) == 16:
                verts.append(vec16)

    if not verts:
        raise RuntimeError(f"Nenhum vértice 16D encontrado em '{filepath}'.")
    return verts
```


```python
# ------------------------------------------------------------
# Construção do comportamento: CLASSE 0 + MARGINAIS (dimensão 18)
# ------------------------------------------------------------
def build_vertices_class0_with_marginals(verticesNS16, X_size=3):
    """
    A entrada 'verticesNS16' traz, para cada x, dois blocos (m=0 e m=1), no total 16 coords:
        Bloco m=0 = v[0:8]   com ordem [B0,B1,C0,C1,E00,E01,E10,E11]
        Bloco m=1 = v[8:16]  com ordem [B0,B1,C0,C1,E00,E01,E10,E11]

    Mantemos, para cada x, as marginais (B0,B1,C0,C1) e APENAS os correladores
    usados pela faceta da classe 0:

        x=0: E00, E01
        x=1: E01, E11
        x=2: E00, E11

    Ordem final (18 coords):
        [ B0,B1,C0,C1,E00,E01 |  B0,B1,C0,C1,E01,E11 |  B0,B1,C0,C1,E00,E11 ]

    Para cada vértice NS 16D e para cada f ∈ {0,1}^X:
        - escolhe o bloco m = f[x] para cada x;
        - extrai (B0,B1,C0,C1) e os E's relevantes daquele bloco.

    Retorna lista de vetores 18D.
    """
    finals = []

    # índices dos elementos dentro de um bloco de 8
    idx_B0, idx_B1, idx_C0, idx_C1 = 0, 1, 2, 3
    idx_E00, idx_E01, idx_E10, idx_E11 = 4, 5, 6, 7

    alice_fs = list(product([0, 1], repeat=X_size))

    for v in verticesNS16:
        # dois blocos 8D
        blk0 = v[0:8]     # m=0
        blk1 = v[8:16]    # m=1

        for f in alice_fs:
            out = []

            # x = 0: [B0,B1,C0,C1,E00,E01]
            blk = blk0 if f[0] == 0 else blk1
            out.extend([blk[idx_B0], blk[idx_B1], blk[idx_C0], blk[idx_C1],
                        blk[idx_E00], blk[idx_E01]])

            # x = 1: [B0,B1,C0,C1,E01,E11]
            blk = blk0 if f[1] == 0 else blk1
            out.extend([blk[idx_B0], blk[idx_B1], blk[idx_C0], blk[idx_C1],
                        blk[idx_E01], blk[idx_E11]])

            # x = 2: [B0,B1,C0,C1,E00,E11]
            blk = blk0 if f[2] == 0 else blk1
            out.extend([blk[idx_B0], blk[idx_B1], blk[idx_C0], blk[idx_C1],
                        blk[idx_E00], blk[idx_E11]])

            finals.append(out)

    return finals  # cada 'out' tem 18 coords

# ------------------------------------------------------------
# Escrita em V-representation (18D)
# ------------------------------------------------------------
def write_vrep_18(filepath, vertices18):
    m = len(vertices18)
    n = 19  # 1 (homogêneo) + 18 coords
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write("* PAM broadcasting — CLASS0 correlators + marginals (18D, |X|=3)\n")
        f.write("* Ordem (por bloco x):\n")
        f.write("*   x=0: [B0, B1, C0, C1, E00, E01]\n")
        f.write("*   x=1: [B0, B1, C0, C1, E01, E11]\n")
        f.write("*   x=2: [B0, B1, C0, C1, E00, E11]\n")
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{m} {n} rational\n")
        for v in vertices18:
            row = ["1"] + [str(int(round(x))) if abs(x - round(x)) < 1e-12 else str(x) for x in v]
            f.write(" ".join(row) + "\n")
        f.write("end\n")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"Lendo NS vertices (16D) de:\n  {in_path}")
    ns16 = parse_vrep_vertices_16(in_path)
    print(f"- lidos {len(ns16)} vértices NS (esperado: ~576).")

    finals18 = build_vertices_class0_with_marginals(ns16, X_size=3)
    print(f"- gerados {len(finals18)} vértices (esperado: 576 * 8 = 4608).")

    # Remover duplicatas (por segurança)
    unique18 = list(set(map(tuple, finals18)))
    print(f"- após remover duplicatas: {len(unique18)} vértices únicos.")

    write_vrep_18(out_path, unique18)
    print(f"Arquivo salvo:\n  {out_path}")
    print("Formato: V-representation, cada linha = [1  + 18 componentes].")
    print("Ordem (18 coords):")
    print("  x=0: [B0,B1,C0,C1,E00,E01] | x=1: [B0,B1,C0,C1,E01,E11] | x=2: [B0,B1,C0,C1,E00,E11]")

if __name__ == "__main__":
    main()
```

