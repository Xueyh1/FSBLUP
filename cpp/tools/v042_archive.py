"""Bundle original Slurm outputs for the 0.4.2 review, without copying the wide input cache."""

import argparse
import os
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path, help="root containing before/ and final/")
p.add_argument("wide_reference", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
final = a.run / "final"
if not (final / "wide-final" / "search.tsv").is_file():
    raise SystemExit("wide search has not published its trace")
with NamedTemporaryFile(dir=a.output.parent, delete=False) as tmp:
    stage = Path(tmp.name)
try:
    with tarfile.open(stage, "w:gz", compresslevel=5) as archive:

        def add(path, name):
            archive.add(
                path,
                arcname=name,
                filter=lambda t: None if "__pycache__" in Path(t.name).parts else t,
            )

        add(a.run / "before", "before")
        for name in (
            "after",
            "numeric",
            "face-predictions",
            "design",
            "ops",
            "units",
            "measured",
            "stochastic",
            "panel-8",
            "panel-16",
            "panel-32",
            "panel-64",
            "heap-C",
            "wide-final",
        ):
            add(final / name, "final/" + name)
        for name in (
            "massif-C-new",
            "massif-C-old",
            "lifetime.tsv",
            "ID-deltas.tsv.gz",
            "official.json",
        ):
            add(final / name, "final/" + name)
        for name in (
            "smoke.log",
            "official/runtime.tsv",
            "official/model.tsv",
            "official/pd.tsv",
            "official/predictions.tsv",
        ):
            add(final / "clean" / name, "final/clean/" + name)
        for path in sorted(final.glob("*.log")):
            add(path, "final/logs/" + path.name)
        for name in (
            "rc-tests.log",
            "shipped-asan.log",
            "shipped-tsan.log",
            "shipped-mem.log",
            "shipped-mem-lifetime.tsv",
            "shipped-mem-lifetime.log",
            "test-mem.log",
            "mem.log",
            "wide-31204.log",
        ):
            add(a.run / name, "debug/" + name)
        for path in sorted((a.run / "controls").iterdir()):
            if path.suffix in (".tsv", ".log"):
                add(path, "debug/controls/" + path.name)
        for name in (
            "search.tsv",
            "fold_counts.tsv",
            "tuning.tsv",
            "model.tsv",
            "runtime.tsv",
            "cache_usage.tsv",
            "pcg_summary.tsv",
            "cv_pcg_summary.tsv",
            "predictions.tsv",
        ):
            add(a.wide_reference / name, "wide-reference/" + name)
    os.replace(stage, a.output)
finally:
    stage.unlink(missing_ok=True)
