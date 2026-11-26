"""Wrap only the native executables in a generated CTest registry with Memcheck."""

import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    config = (build / "CTestTestfile.cmake").read_text()
    for name in (
        "fsblup",
        "core_test",
        "nearpd_core_test",
        "io_limits_test",
        "repeat_test","pcg_math_test","cache_math_test",
    ):
        exe = build / name
        wrapper = out / name
        # Keep Valgrind diagnostics on a pipe/stderr during RLIMIT_FSIZE tests.
        # A sentinel prevents an expected application rejection from masking a
        # Memcheck error even when the Python test only checks nonzero status.
        wrapper.write_text(
            "#!/bin/sh\nvalgrind --quiet --vgdb=no --tool=memcheck --leak-check=full "
            "--show-leak-kinds=all --errors-for-leak-kinds=definite,indirect "
            "--track-origins=yes --error-exitcode=99 "
            + shlex.quote(str(exe))
            + ' "$@"\nrc=$?\nif [ "$rc" = 99 ]; then touch '
            + shlex.quote(str(out / "memcheck-error"))
            + '; fi\nexit "$rc"\n'
        )
        wrapper.chmod(0o755)
        assert str(exe) in config, name
        config = config.replace('"' + str(exe) + '"', '"' + str(wrapper) + '"')
    (out / "CTestTestfile.cmake").write_text(config)
    subprocess.run(
        ["ctest", "--test-dir", str(out), "--parallel", "2", "--output-on-failure"],
        check=True,
    )
    if (out / "memcheck-error").exists():
        raise RuntimeError("Memcheck reported errors in a subprocess")


if __name__ == "__main__":
    main()
