# From a primordial fluctuation to an idealised rotation curve

Code, data, supplementary notes and the interactive tool accompanying

> B. R. Parodi, *From a primordial fluctuation to an idealised rotation curve — Collapse, energy conservation, and the cores of late-type dwarf galaxies* (A&A, submitted)

## Grow a Dwarf Galaxy (interactive)

Open **https://beroparodi.github.io/pf2irc/** in any browser, or download `index.html` and open it locally (no installation needed).
Set the flat speed *v*, the spin λ, the redshift at which growth ends, the feedback ε and the profile exponent *n*, and grow any of the 79 SPARC dwarfs of the paper.

## Code (`code/`)

Run the scripts from the `code/` folder with Python 3 (numpy, scipy, matplotlib).

| file | purpose |
|---|---|
| `make_figures.py` | reproduces the figures, tables and numbers of the paper; writes `results.txt` (included, from the run of 26 Oct 2026) |
| `robustness.py` | χ² of the fits, the slope of r_c ∝ v^b for all estimators and subsamples (Table F.1), the mock samples, the energy budget and the disc test (Sects. 5.6, 6, App. F); output in `robustness_results.txt` |
| `mock_b2.py` | the 1000 mock samples with a true b = 2 (App. F); logs `mockA.log`, `mockB.log`, results `mock_b2_*.npy` |
| `fit_exponent.py` | fits the generalised profile ρ = v²/(4πGR²)[1−(1−x)e^(−x)]^n with v, r_c and n free (Sect. 7, App. G); output in `fit_exponent_out.txt`, `fit_exponent_results.csv` |
| `fig_nfree.py`, `fig_alpha.py` | Fig. 6 and Fig. G.1 |
| `recipe.py` | the Python recipe of Appendix C (identical to the printed listing) |
| `make_table.py` | Table E.1 |
| `checks_sec6to8.py` | numbers of Sects. 6.4, 6.6, 7.4 and 8.1 not printed by `make_figures.py` (epoch ranges, n = 1.76, pericentres, radial acceleration relation, core growth after t_acc) |
| `spin_collapse.py` | effect of the spin on the collapse (Sect. 4.2) |
| `shells/` | one-dimensional collisionless shell runs (Sect. 8.1, App. H): `shells.py`, `driver2.py` (runs), `analyse2.py`, `fig.py` (Fig. H.1), `selfsim_check.py` with `selfsim_check_out.txt` (self-similarity check); `shells2.py`, `shells3.py`, `driver3.py`, `driver4.py`, `analyse3.py` are the transport experiments; `splashback_*` belong to the splashback note |
| `sparc_data/` | SPARC rotation curves, the fits used in the paper and the out-of-sample curves |

## Supplementary notes (`notes/`)

Each note has its PDF, LaTeX source and code.

| folder | topic |
|---|---|
| `tophat_note/` | shell and density pictures of a top hat, and the mean density of the seed (paper Sect. 3.2) |
| `splashback_note/` | splashback and the accretion front (Sect. 3.3) |
| `seed_disc/` | the disc that the seed's angular momentum would build (Sect. 6.5) |

## Data

The rotation curves are from SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157; http://astroweb.cwru.edu/SPARC/); please cite that paper when using them.

## How to cite

See `CITATION.cff` or the Zenodo record (DOI badge on the repository page).

## Licence

Code: MIT licence (see `LICENSE`). Text, figures and derived tables: CC BY 4.0.
