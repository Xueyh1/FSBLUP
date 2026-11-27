"""Archive raw scientific and detector outputs without generated build trees or wide inputs."""

import argparse, tarfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for key in ("run", "output"):
    p.add_argument(key, type=Path)
a = p.parse_args()
with tarfile.open(a.output, "w:gz", compresslevel=4) as ar:

    def add(path, name):
        ar.add(
            path,
            arcname=name,
            filter=lambda t: None if "__pycache__" in Path(t.name).parts else t,
        )

    add(a.run / "before", "before")
    for name in [
        "after",
        "comparison",
        "connected",
        "validation",
        "panel",
        "medium-1",
        "medium-2",
        "medium-3",
    ]:
        add(a.run / "release" / name, "after/" + name)
    for name in [
        "baseline-ctest.log",
        "release.log",
        "wide-attempt.log",
        "wide-exit",
        "node-at-planning.txt",
        "evidence.log",
        "medium.log",
        "clean.log",
        "rc-test.log",
        "ship-tests.log",
    ]:
        add(a.run / name, "logs/" + name)
    for batch in ["b1", "b2", "b3", "b4", "b5", "bfinal"]:
        root = a.run / batch
        for f in (
            list(root.glob("*.log"))
            + list(root.glob("*.tsv"))
            + list((root / "controls").glob("*.log"))
            + list((root / "controls").glob("*.tsv"))
        ):
            add(f, batch + "/" + str(f.relative_to(root)))
    for name in [
        "smoke.log",
        "official/pd.tsv",
        "official/model.tsv",
        "official/predictions.tsv",
        "official/runtime.tsv",
    ]:
        add(a.run / "release" / "clean" / name, "clean/" + name)
