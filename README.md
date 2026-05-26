# The Prepare-and-Broadcast Scenario — Numerical Codes

This repository contains the numerical codes for the paper  
**"The Prepare and Broadcast Scenario"**  
by T. S. Sarubi, M. Alves, S. Zamora, V. Alves, A. de Oliveira Junior, C. Roch i Carceller, A. Tavakoli, and R. Chaves.

---

## Overview

The prepare-and-broadcast (PAB) scenario extends the standard prepare-and-measure (PAM) framework by allowing the prepared message to be distributed to two spatially separated receivers (Bob and Charlie) through a broadcasting channel. This repository provides codes for:

- Enumerating vertices of the classical PAB polytope in several projected spaces
- Classifying facet inequalities under scenario symmetries
- Computing classical–quantum (CQ) upper bounds via the NPA hierarchy
- Optimising quantum violations of PAB inequalities (QQ model)
- Studying nonclassicality activation via critical visibility calculations

---

## File Index

| File | Description |
|------|-------------|
| `01_pab_vertices_24dim.md` | Full 24D classical vertex enumeration (`\|X\|=3`) |
| `02_pab_vertices_class0_marginals.md` | 18D projection: marginals + Class-0 two-body correlators |
| `03_pab_twobody_vertices.md` | 12D two-body projection, classical polytope (`\|X\|=3`) |
| `04_pab_marginals_3prep_twobody_4prep.md` | Marginal vertices (3 prep) and two-body vertices (4 prep) |
| `05_pab_marginals_4prep.md` | 16D marginal polytope, 4 preparations |
| `06_classify_pab_facets.md` | Symmetry classification of PAB two-body facets |
| `07_bcq_npa_levels.md` | CQ bound via NPA hierarchy at levels 1, 2, 3 |
| `08_bound_qq_4prep.md` | QQ violation for 4-preparation PAB facet (activation) |
| `09_visibility_500_measurements.md` | Critical PAM visibility vs `nY` (3 preparation sets, 500 seeds) |
| `10_min_visibility_vs_nY.md` | Minimum `v*` vs `nY` sweep (500 seeds, HiGHS LP) |

---

## Dependencies

```bash
pip install numpy scipy cvxpy pandas matplotlib
```

Facet enumeration requires [`lrs`](http://cgm.cs.mcgill.ca/~avis/C/lrs.html) installed separately.

---

## Usage

Each `.md` file is a self-contained documented script. Copy the code blocks  
into a `.py` file or a Jupyter notebook and run directly.  
Input/output paths may need to be adjusted to your local directory structure.

---

## Citation

If you use these codes, please cite the associated paper (reference to be added upon publication).
