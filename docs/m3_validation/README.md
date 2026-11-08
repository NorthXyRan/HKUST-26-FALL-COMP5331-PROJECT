# M3 validation evidence

Executed on 2026-11-08 using Python 3.11.18 and `configs/m3_environment.txt`.

- `python -m src.m3 --output results/m3`: 18 frozen streams completed.
- `python -m src.m3_verify`: all input/source/output hashes checked; every statistics table independently recomputed from saved rows.
- `python -m unittest discover -s tests -v`: 34 tests, 33 passed, 1 optional pinned-upstream formula test skipped because TGF checkout is absent.
- Seven output figures were visually inspected.
- `git diff --check`: passed.

The tiny single-observation summary fixture can trigger a SciPy asymptotic p-value
warning; only the KS statistic is consumed, and no p-values are reported.

The saved M2/M4 input manifest, streams, features and results are unchanged.
The historical source-hash test validates the producing commit so M5's later
modeling refactor does not invalidate frozen provenance. Current behavioral
checks still execute against the current source tree.
