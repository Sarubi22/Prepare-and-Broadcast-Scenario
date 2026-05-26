# PAB Classical Vertices — Full Correlator Space (24D)

Generates the vertices of the classical prepare-and-broadcast (PAB) polytope  
in the full 24-dimensional correlator space for the scenario `|X|=3, |Y|=2, |Z|=2, |M|=2`.

Each vertex is built by combining a deterministic encoding `f: X → M` with a pair  
of nonsignalling blocks (one per message value). The output is a V-representation  
file readable by `lrs`.

**Scenario parameters:** `|X|=3` preparations, `|M|=2` classical message, binary inputs and outputs.  
**Output vector per preparation** `x`: `[B0, B1, C0, C1, E00, E01, E10, E11]` — 8 correlators.  
**Total dimension:** 24.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

```python
import os
import numpy as np
from itertools import product
from fractions import Fraction
```


```python
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
in_path  = os.path.join(downloads_path, "pam_broadcasting_verticesNS.ine.out")
out_path = os.path.join(os.path.expanduser("~"), "pam_vertices_final_correlators.ext")
```


```python
def parse_vrep_vertices_16(filepath):
    """
    Lê V-representation (lrs) com vértices em R^16.
    Considera apenas linhas que começam com '1' (homogêneo).
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
            if low == "v-representation" or low == "h-representation":
                continue
            if low == "begin":
                in_body = True
                continue
            if low == "end":
                break
            if not in_body:
                # ignora cabeçalhos, inclusive 'm n rational'
                continue

            parts = line.split()
            # Pegue só linhas de vértice: começam com '1' (ou '1/1')
            if parts[0] not in ("1", "1/1"):
                continue

            nums = []
            ok = True
            for p in parts:
                p = p.replace(",", "")
                try:
                    if "/" in p:
                        nums.append(float(Fraction(p)))
                    else:
                        nums.append(float(p))
                except Exception:
                    ok = False
                    break
            if not ok or len(nums) < 17:
                continue

            vec16 = nums[1:17]  # remove homogêneo
            if len(vec16) == 16:
                verts.append(vec16)

    if not verts:
        raise RuntimeError(f"Nenhum vértice 16D encontrado em '{filepath}'.")
    return verts

def build_final_vertices_24(verticesNS16, X_size=3):
    """
    Para cada vértice 16D (m=0|m=1) e cada f ∈ {0,1}^X, concatena os blocos.
    """
    finals = []
    alice_fs = list(product([0, 1], repeat=X_size))
    for v in verticesNS16:
        blk0 = v[0:8]    # m=0
        blk1 = v[8:16]   # m=1
        for f in alice_fs:
            out = []
            for x in range(X_size):
                out.extend(blk0 if f[x] == 0 else blk1)
            finals.append(out)
    return finals
```


```python
def write_vrep_24(filepath, vertices24):
    m = len(vertices24)
    n = 25  # 1 (homogêneo) + 24 coords
    with open(filepath, "w", encoding="ascii", newline="\n") as f:
        f.write("* PAM broadcasting final vertices (correlators, 24D, |X|=3)\n")
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{m} {n} rational\n")
        for v in vertices24:
            row = ["1"] + [str(int(round(x))) if abs(x - round(x)) < 1e-12 else str(x) for x in v]
            f.write(" ".join(row) + "\n")
        f.write("end\n")

def main():
    print(f"Lendo NS vertices (16D) de:\n  {in_path}")
    ns16 = parse_vrep_vertices_16(in_path)
    print(f"- lidos {len(ns16)} vértices NS (esperado: ~576).")

    finals24 = build_final_vertices_24(ns16, X_size=3)
    print(f"- gerados {len(finals24)} vértices finais (esperado: 576 * 8 = 4608).")

    # Remover duplicatas
    unique24 = list(set(map(tuple, finals24)))
    print(f"- após remover duplicatas: {len(unique24)} vértices únicos.")

    write_vrep_24(out_path, unique24)
    print(f"Arquivo salvo:\n  {out_path}")
    print("Formato: V-representation, cada linha = [1  + 24 correladores].")
    print("Ordem (24 coords): blocos x=0, depois x=1, depois x=2; em cada bloco: [B0, B1, C0, C1, E00, E01, E10, E11].")

if __name__ == "__main__":
    main()
```

