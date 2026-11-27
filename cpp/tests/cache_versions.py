import hashlib, subprocess, sys, tempfile, shutil
from pathlib import Path

d = Path(__file__).parent / "data"


def call(args):
    p = subprocess.run([sys.argv[1], *args], capture_output=True, text=True)
    assert not p.returncode, p.stderr


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cache = root / "v3"
    call(
        [
            "prepare",
            "--ped",
            d / "ped.tsv",
            "--snp",
            d / "snp.tsv",
            "--omics",
            d / "omics.tsv",
            "--out",
            cache,
        ]
    )
    ref = None
    for version in (3, 2, 1):
        dest = root / ("v" + str(version))
        if version < 3:
            shutil.copytree(cache, dest)
            (dest / "topology.tsv").unlink()
            manifest = (cache / "manifest").read_text().splitlines()[1:-1]
            if version == 1:
                for name in ("G.shape", "O.shape"):
                    shape = dest / name
                    shape.write_text(" ".join(shape.read_text().split()[:3]) + "\n")
                    manifest = [
                        hashlib.sha256(shape.read_bytes()).hexdigest() + " " + name
                        if r.endswith(" " + name)
                        else r
                        for r in manifest
                    ]
            header = (
                "fsblup-cache 1 little-endian f64"
                if version == 1
                else "fsblup-cache 2 little-endian"
            )
            (dest / "manifest").write_text("\n".join([header, *manifest]) + "\n")
        out = root / str(version)
        call(
            [
                "fit",
                "--cache",
                dest,
                "--cache-backend",
                "operator",
                "--pheno",
                d / "pheno.tsv",
                "--trait",
                "trait",
                "--alpha",
                ".2",
                "--beta",
                ".4",
                "--out",
                out,
            ]
        )
        pred = (out / "predictions.tsv").read_bytes()
        if ref is not None:
            assert pred == ref
        ref = pred
