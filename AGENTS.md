# Project conventions

- Use the pinned versions in requirements.txt; run commands from the repository root.
- Keep saved upstream experiment inputs immutable. Write new analyses under results/<milestone>/.
- Reuse src/modeling.py for M4-compatible balancing, model fitting and metrics.
- Record inputs, configuration, seeds, dependency versions and source hashes for saved experiments.
- Separate injection-seed variation, classifier-seed variation and permutation repeats in reports.
- Use unittest for behavioral tests. Run `python -m unittest discover -s tests -v` before committing.
- Keep feature definitions tied to upstream extractor code. Do not infer feature semantics from a milestone label.
