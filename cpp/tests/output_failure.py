"""Verify that buffered output failures are reported before publication on Linux."""
import csv
import resource
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def run_checked(binary, args, root, name, cap, expected_file):
    dest = root / name
    if name == "fit":
        dest.mkdir()  # --out also accepts an existing empty directory.

    def limit_file_size():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))

    result = subprocess.run([str(binary), *args, "--out", str(dest)],
                            preexec_fn=limit_file_size, capture_output=True, text=True)
    assert result.returncode != 0, "incomplete output reported as success"
    assert expected_file in result.stderr, result.stderr
    assert not dest.exists() or not list(dest.iterdir()), dest
    assert not list(root.glob(f"{name}.tmp.*"))


def test(binary, root):
    ids = [f"s{i:02}" for i in range(24)]
    ids[:3] = ['s"0', "s\t1", "s,2"]
    matrix = root / "cov.tsv"
    with matrix.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["ID", *ids])
        for i, sample in enumerate(ids):
            writer.writerow([sample, *[0.65 ** abs(i - j) for j in range(len(ids))]])
    pheno = root / "pheno.tsv"
    with pheno.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["ID", "y"])
        for i, sample in enumerate(ids):
            writer.writerow([sample, (7 * i) % 13 + i / 19])
    inputs = ["--A", str(matrix), "--G", str(matrix), "--O", str(matrix),
              "--alpha", "0", "--beta", "0"]
    control = root / "complete"
    subprocess.run([str(binary), "fuse", *inputs, "--out", str(control)], check=True)
    with (control / "K_raw.tsv").open(newline="") as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    assert table[0] == ["ID", *ids] and len(table) == len(ids) + 1
    for i, row in enumerate(table[1:]):
        expected = [format(0.65 ** abs(i - j), ".17g") for j in range(len(ids))]
        assert row == [ids[i], *expected], row
    run_checked(binary, ["fuse", *inputs], root, "metadata", 32, "memory.tsv")
    run_checked(binary, ["fuse", *inputs], root, "matrix", 512, "K_raw.tsv")
    run_checked(binary, ["fit", *inputs, "--pheno", str(pheno), "--trait", "y"],
                root, "fit", 512, "predictions.tsv")


if __name__ == "__main__":
    with TemporaryDirectory() as work:
        test(Path(sys.argv[1]).resolve(), Path(work))
