# Classical–Quantum Bound via NPA Hierarchy (Levels 1, 2, 3)

Computes the classical–quantum (CQ) upper bound for a fixed PAB facet  
using the NPA (Navascues–Pironio–Acin) semidefinite programming hierarchy  
at levels 1, 2, and 3.

**Setup:**
- Alice's message is classical and dimension-bounded (`|M|=2`).
- For each deterministic encoding `f: X → M`, the broadcasting device  
  distributes a bipartite quantum state to Bob and Charlie.
- The CQ bound is the maximum of the NPA bound over all deterministic encodings.

**NPA levels:**
- **Level 1:** moment matrix built from `{I, B0, B1, C0, C1}`.
- **Level 2:** adds all length-2 products.
- **Level 3:** adds length-3 products.

The SDP is solved with `cvxpy`. The bound is stable across levels,  
confirming convergence at level 1 for the witnesses studied in the paper.


---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

---

# Bound clássico-quântico $B_{CQ}$ para facetas PAB — NPA níveis 1, 2 e 3

Este notebook calcula, para uma faceta fixa do politopo clássico, o bound do modelo clássico-quântico (CQ). A ideia é:

1. a faceta vem no formato `lrs`, isto é, `b + a.v >= 0`;
2. reescrevemos como um witness `W <= B_C`, com `W = -a.v`;
3. fixamos uma estratégia clássica determinística `f: X -> M`;
4. agrupamos os coeficientes da faceta por mensagem clássica `m`;
5. para cada mensagem `m`, calculamos o bound quântico do bloco bipartido `BC` por NPA;
6. somamos os blocos e maximizamos sobre todas as estratégias determinísticas `f`.

O modelo CQ usado é

$$p(b,c|x,y,z)=\sum_{m,\lambda}p(\lambda)p(m|x,\lambda)p^Q_{BC}(b,c|y,z,m,\lambda).$$

Assim, a mensagem enviada por Alice continua clássica e limitada, mas Bob e Charlie podem compartilhar um bloco quântico condicionado a `m`.

Neste notebook, a NPA é rodada nos níveis `1`, `2` e `3`. O nível 1 usa as palavras `{I,B0,B1,C0,C1}`. Os níveis 2 e 3 incluem todas as palavras reduzidas até comprimento 2 e 3, respectivamente, com as relações `B_y^2 = C_z^2 = I` e `[B_y,C_z]=0`.


## 1. Imports e Faceta


```python
import numpy as np
import cvxpy as cp
from itertools import product as iprod
import warnings
warnings.filterwarnings('ignore')

# ── Cenário ───────────────────────────────────────────────────────────────────
NX, NY, NZ, NM = 3, 2, 2, 2

# ── Faceta (formato lrs: b + a.v >= 0) ───────────────────────────────────────
# Ordem das 12 coords: bloco x=0: [<B0C0>,<B0C1>,<B1C0>,<B1C1>], x=1, x=2
LRS_ROW  = [6, -2, 0, 0, -2, -1, -1, 1, 1, 1, 1, -1, 1]
B_C      = float(LRS_ROW[0])                           # bound clássico
ALPHA    = -np.array(LRS_ROW[1:], dtype=float).reshape(NX, NY, NZ)  # coef. W = ALPHA.v

COORD_NAMES = [f'<B{y}C{z}>[{x}]'
               for x in range(NX) for y in range(NY) for z in range(NZ)]

print(f'Bound clássico B_C = {B_C}')
print('Coeficientes alpha[x,y,z] não-nulos:')
for x in range(NX):
    for y in range(NY):
        for z in range(NZ):
            if abs(ALPHA[x,y,z]) > 1e-12:
                print(f'  alpha[{x},{y},{z}] = {ALPHA[x,y,z]:+.0f}  '
                      f'({COORD_NAMES[x*NY*NZ + y*NZ + z]})')
```


## 2. Verificação do Bound Clássico


```python
def check_classical_bound():
    max_W = -1e9
    for F in iprod(range(NM), repeat=NX):
        for hB in iprod(range(2), repeat=NY*NM):
            for hC in iprod(range(2), repeat=NZ*NM):
                W = 0.0
                for x in range(NX):
                    m = F[x]
                    for y in range(NY):
                        for z in range(NZ):
                            bs = (-1)**hB[y*NM + m]
                            cs = (-1)**hC[z*NM + m]
                            W += ALPHA[x,y,z] * bs * cs
                if W > max_W:
                    max_W = W
    return max_W

W_cl = check_classical_bound()
print(f'Max W sobre vértices clássicos = {W_cl}')
print(f'Bound lrs  = {B_C}')
print(f'Consistente? {abs(W_cl - B_C) < 1e-9}')
```


## 3. Matriz de Blocos $A^{(m)}(f)$

Para uma função $f$ fixada e uma mensagem $m$, a matriz de bloco é:

$$A^{(m)}_{yz}(f) = \sum_{x:\, f(x)=m} \alpha_{xyz}$$

Ela agrega os coeficientes de todas as preparações $x$ que enviam a mensagem $m$.


```python
def block_matrix(f_map, m):
    """
    Calcula A^(m)(f): matriz (NY x NZ) dos coeficientes agregados
    para as preparações x tais que f(x) = m.
    """
    A = np.zeros((NY, NZ), dtype=float)
    for x in range(NX):
        if f_map[x] == m:
            A += ALPHA[x]   # ALPHA[x] tem shape (NY, NZ)
    return A

# Exemplo: f=(0,0,1)
f_ex = (0, 0, 1)
print(f'Exemplo f={f_ex}:')
for m in range(NM):
    A = block_matrix(f_ex, m)
    xs = [x for x in range(NX) if f_ex[x] == m]
    print(f'  A^(m={m}) — preps x={xs}:')
    print(f'    {A}')
```


## 4. NPA por bloco até o nível 3

Para cada bloco de mensagem `m`, resolvemos um SDP da hierarquia NPA para maximizar o funcional bipartido de correlatores.

- **Nível 1:** palavras até comprimento 1, isto é, `{I,B0,B1,C0,C1}`.
- **Nível 2:** palavras reduzidas até comprimento 2.
- **Nível 3:** palavras reduzidas até comprimento 3.

As regras usadas são:

- `B_y^2 = I` e `C_z^2 = I`;
- operadores de Bob e Charlie comutam entre si;
- produtos são reduzidos usando essas relações;
- entradas da matriz de momento que representam o mesmo produto reduzido são identificadas.

Com isso, podemos verificar se o bound diminui ao passar de nível 1 para níveis 2 e 3, como sugerido.


```python
def reduce_party_word(seq):
    """
    Reduz uma palavra de uma única parte usando A_i^2 = I.
    Como os operadores da mesma parte não comutam, apenas letras iguais
    adjacentes se cancelam.
    """
    out = []
    for a in seq:
        if out and out[-1] == a:
            out.pop()
        else:
            out.append(a)
    return tuple(out)


def multiply_words(w1, w2):
    """
    Produto de palavras canônicas.

    Uma palavra canônica é um par (bob_word, charlie_word), onde os
    operadores de Bob ficam à esquerda e os de Charlie à direita.
    Isso usa [B_y, C_z] = 0, preservando a ordem interna de cada parte.
    """
    b1, c1 = w1
    b2, c2 = w2
    return (reduce_party_word(b1 + b2), reduce_party_word(c1 + c2))


def dagger_word(w):
    """
    Adjunto de uma palavra. As letras são hermitianas, mas a ordem inverte.
    """
    b, c = w
    return (tuple(reversed(b)), tuple(reversed(c)))


def reduced_sequences(max_len):
    """
    Todas as sequências reduzidas em dois geradores {0,1} até max_len.
    Não há letras iguais adjacentes.
    """
    seqs = [tuple()]
    for L in range(1, max_len + 1):
        for start in range(2):
            seq = [start]
            for _ in range(1, L):
                seq.append(1 - seq[-1])
            seqs.append(tuple(seq))
    return seqs


def generate_words(level):
    """
    Gera palavras canônicas (bob_word, charlie_word) com comprimento total
    até o nível escolhido.
    """
    seqs = reduced_sequences(level)
    words = []
    seen = set()
    for b in seqs:
        for c in seqs:
            if len(b) + len(c) <= level:
                w = (b, c)
                if w not in seen:
                    seen.add(w)
                    words.append(w)
    words.sort(key=lambda w: (len(w[0]) + len(w[1]), len(w[0]), w[0], w[1]))
    return words


def word_label(w):
    b, c = w
    if len(b) == 0 and len(c) == 0:
        return 'I'
    parts = []
    parts += [f'B{y}' for y in b]
    parts += [f'C{z}' for z in c]
    return ''.join(parts)


# Cache para não resolver o mesmo bloco várias vezes
_NPA_CACHE = {}


def npa_block_bound(A, level=1, solver=cp.CLARABEL, verbose=False):
    """
    Bound NPA para um bloco correlator-only:

        max sum_yz A[y,z] <B_y C_z>

    usando palavras até o comprimento `level`.
    """
    A = np.asarray(A, dtype=float).reshape(NY, NZ)

    if np.allclose(A, 0):
        return 0.0

    key = (int(level), tuple(np.round(A.ravel(), 12)))
    if key in _NPA_CACHE:
        return _NPA_CACHE[key]

    words = generate_words(level)
    n = len(words)
    idx = {w: i for i, w in enumerate(words)}

    # Matriz de momento hermitiana complexa
    G = cp.Variable((n, n), hermitian=True)

    constraints = [G >> 0]

    # Normalização
    id_word = (tuple(), tuple())
    id_idx = idx[id_word]
    constraints.append(G[id_idx, id_idx] == 1.0)

    # Diagonal igual a 1: para qualquer palavra w, w^† w = I pelas relações unitárias
    for i in range(n):
        constraints.append(G[i, i] == 1.0)

    # Igualdade de entradas que representam o mesmo produto reduzido u^† v
    product_classes = {}
    for i, u in enumerate(words):
        du = dagger_word(u)
        for j, v in enumerate(words):
            prod = multiply_words(du, v)
            product_classes.setdefault(prod, []).append((i, j))

    for entries in product_classes.values():
        ref_i, ref_j = entries[0]
        ref = G[ref_i, ref_j]
        for i, j in entries[1:]:
            constraints.append(G[i, j] == ref)

    # Objetivo. Usamos <B_y C_z> = Gamma[B_y, C_z].
    obj = 0.0
    for y in range(NY):
        By = ((y,), tuple())
        iB = idx[By]
        for z in range(NZ):
            Cz = (tuple(), (z,))
            iC = idx[Cz]
            obj += float(A[y, z]) * cp.real(G[iB, iC])

    prob = cp.Problem(cp.Maximize(obj), constraints)

    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-8, max_iters=200000, verbose=verbose)

    if prob.status not in ('optimal', 'optimal_inaccurate') or prob.value is None:
        val = np.nan
    else:
        val = float(prob.value)

    _NPA_CACHE[key] = val
    return val


# Compatibilidade com o nome antigo usado no notebook
def tsirelson_block(A, level=1):
    return npa_block_bound(A, level=level)


print('Número de palavras por nível:')
for L in [1, 2, 3]:
    words_L = generate_words(L)
    print(f'  nível {L}: {len(words_L)} palavras')
    print('   ', [word_label(w) for w in words_L])

print('\nTeste com CHSH em níveis 1, 2 e 3:')
A_chsh = np.array([[1, 1], [1, -1]], dtype=float)
for L in [1, 2, 3]:
    beta_chsh = npa_block_bound(A_chsh, level=L)
    print(f'  nível {L}: beta^Q(CHSH) = {beta_chsh:.8f}  | esperado: {2*np.sqrt(2):.8f}')
```


## 5. Bound CQ — Loop sobre as 8 Funções $f$

$$B_{CQ} = \max_{f: X \to M} \sum_{m=0}^{1} \beta^Q\!\left(A^{(m)}(f)\right)$$


```python
def evaluate_cq_bound_at_level(level):
    """
    Calcula B_CQ no nível NPA especificado, maximizando sobre todas as
    estratégias determinísticas f: X -> M.
    """
    all_f = list(iprod(range(NM), repeat=NX))
    results_level = []

    for f_map in all_f:
        total = 0.0
        block_vals = []
        block_mats = []

        for m in range(NM):
            A = block_matrix(f_map, m)
            b = npa_block_bound(A, level=level)
            total += b
            block_vals.append(b)
            block_mats.append(A)

        results_level.append({
            'level': level,
            'f': f_map,
            'B_CQ': total,
            'blocks': block_vals,
            'block_mats': block_mats,
        })

    results_level.sort(key=lambda r: r['B_CQ'], reverse=True)
    return results_level


LEVELS_TO_TEST = [1, 2, 3]
results_by_level = {}

print('=' * 72)
print('Bound CQ por blocos de mensagem — NPA níveis 1, 2 e 3')
print('=' * 72)
print(f'Bound clássico B_C = {B_C}\n')

for level in LEVELS_TO_TEST:
    print('-' * 72)
    print(f'NÍVEL NPA {level}')
    print('-' * 72)

    res_level = evaluate_cq_bound_at_level(level)
    results_by_level[level] = res_level

    for r in res_level:
        beta_str = '  '.join([f'beta{m}={r["blocks"][m]:.6f}' for m in range(NM)])
        flag = '  *** ACIMA DE B_C ***' if r['B_CQ'] > B_C + 1e-4 else ''
        print(f'f={r["f"]}:  {beta_str}  B_CQ={r["B_CQ"]:.6f}{flag}')

    best = res_level[0]
    print(f'\nMelhor no nível {level}: f={best["f"]}, B_CQ={best["B_CQ"]:.8f}\n')
```


## 6. Resultado Final


```python
print('=' * 72)
print('RESULTADO FINAL POR NÍVEL')
print('=' * 72)
print(f'Bound clássico B_C = {B_C:.6f}\n')

for level in LEVELS_TO_TEST:
    best = results_by_level[level][0]
    print(f'Nível {level}:')
    print(f'  Bound CQ/NPA B_CQ = {best["B_CQ"]:.8f}')
    print(f'  Gap B_CQ - B_C    = {best["B_CQ"] - B_C:.8f}')
    print(f'  Melhor f          = {best["f"]}')
    print('  Blocos da melhor estratégia:')
    for m in range(NM):
        A = best['block_mats'][m]
        xs = [x for x in range(NX) if best['f'][x] == m]
        print(f'    m={m}, preps x={xs}: A={A.tolist()}, beta={best["blocks"][m]:.8f}')
    print()

# Checagem de monotonicidade: bounds NPA devem ser não-crescentes com o nível
vals = [results_by_level[L][0]['B_CQ'] for L in LEVELS_TO_TEST]
print('Checagem de monotonicidade esperada: B_CQ(Q1) >= B_CQ(Q2) >= B_CQ(Q3)')
for L, val in zip(LEVELS_TO_TEST, vals):
    print(f'  Q{L}: {val:.8f}')

ok_monotone = all(vals[i] + 1e-5 >= vals[i+1] for i in range(len(vals)-1))
print(f'Monotonicidade satisfeita? {ok_monotone}')

if abs(vals[0] - vals[-1]) < 1e-5:
    print('Os níveis 1, 2 e 3 coincidem dentro da precisão numérica.')
else:
    print('O bound diminuiu ao subir o nível da NPA; use o nível mais alto como melhor upper bound.')
```


## 7. Comparação com um lower bound numérico, se houver

O valor calculado pela NPA é um **upper bound** para o modelo clássico-quântico no nível escolhido. Ao aumentar o nível da NPA, o bound deve ficar igual ou diminuir.

Se você tiver um valor obtido por otimização explícita, por exemplo via SLSQP com uma estratégia quântica concreta, esse valor é um **lower bound**: ele mostra que existe uma realização que atinge aquele número.

A interpretação correta é:

- se `W_slsqp > B_C`, o modelo totalmente clássico é violado;
- se `W_slsqp > B_CQ` no nível mais alto testado, então o modelo clássico-quântico também é excluído nesse bound;
- se `B_C < W_slsqp <= B_CQ`, a violação clássica ainda pode ser explicada por mensagem clássica mais bloco quântico `BC`.


```python
# Cole aqui o W* obtido pelo SLSQP para esta faceta, se houver
W_slsqp = None   # ex: W_slsqp = 8.123456

best_highest = results_by_level[max(LEVELS_TO_TEST)][0]
B_CQ_highest = best_highest['B_CQ']

print('Comparação:')
print(f'  Bound clássico B_C              = {B_C:.6f}')
for level in LEVELS_TO_TEST:
    val = results_by_level[level][0]['B_CQ']
    print(f'  Upper bound NPA nível {level}     = {val:.6f}')

if W_slsqp is not None:
    print(f'  Lower bound explícito/SLSQP     = {W_slsqp:.6f}')
    print(f'  Violação sobre B_C              = {W_slsqp - B_C:.6f}')
    print(f'  Gap para NPA nível {max(LEVELS_TO_TEST)}         = {B_CQ_highest - W_slsqp:.6f}')

    if W_slsqp > B_CQ_highest + 1e-5:
        print('  => O lower bound ultrapassa o upper bound CQ: conferir implementação/tolerâncias.')
    elif W_slsqp > B_C + 1e-5:
        print('  => Viola o clássico, mas ainda não exclui necessariamente o modelo CQ.')
    else:
        print('  => Não há violação do bound clássico para este W_slsqp.')
else:
    print('  Nenhum lower bound explícito foi fornecido em W_slsqp.')
```

