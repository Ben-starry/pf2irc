# From a primordial fluctuation to an idealised rotation curve

Code, data and the interactive tool accompanying

> B. R. Parodi, *From a primordial fluctuation to an idealised rotation curve* (A&A, submitted)

## Grow a Dwarf Galaxy (interactive)

Open **https://USERNAME.github.io/pf2irc/** in any browser, or download `index.html` and open it locally (it needs no installation).
Set the flat speed *v*, the spin λ, the end of growth, the feedback ε and the profile exponent *n*, and grow any of the 79 SPARC dwarfs of the paper.

## Code (`code/`)

| file | purpose |
|---|---|
| `make_figures.py` | reproduces every figure, table and number of the paper (writes `results.txt`) |
| `grow.py` | the four-step recipe of Appendix G |
| `fit_exponent.py` | fits ρ = ρ_SIS [1−(1−x)e^(−x)]^n with v, r_c and n free |
| `shells/` | one-dimensional collisionless shell experiment (Sect. 7.1) and transport runs |
| `sparc_data/` | SPARC rotation curves and the fits used in the paper |

Run the scripts from the `code/` folder with Python 3 (numpy, scipy, matplotlib).

## Data

The rotation curves are from SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157; http://astroweb.cwru.edu/SPARC/); please cite that paper when using them.

## How to cite

See `CITATION.cff` or the Zenodo record (DOI badge on the repository page).

## Licence

Code: MIT licence (see `LICENSE`). Text, figures and derived tables: CC BY 4.0.
