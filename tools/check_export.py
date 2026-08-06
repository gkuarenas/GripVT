"""Verify a BSL Pro dynamometer export BEFORE the participant unmounts.

Both of participant 001's export failures were catchable in ten seconds, and
both would have been fixable by re-exporting while the participant was still
in the chair. This tool exists so that check is one command instead of ten
manual squints across a two-hour block.

The two real failure modes, from
docs/participants/participant001/data-quality-findings.md:

  session 4 -- header written, ",37555 samples," declared, ZERO data rows.
               File is 171 bytes. Obvious once you look; invisible if you
               don't.

  session 1 -- data rows written but NO HEADER AT ALL. This is the nastier
               one: the file is ~1 MB of perfectly good force samples, so it
               looks fine by every casual check. But the header is where the
               "Recording on:" wall-clock line lives, and that line is the
               ONLY anchor tying the Biopac clock to the camera clock. No
               header means no sync, which means the marker data has no
               usable force label. A full-looking file that is silently
               useless.

So "did the export work" is really two independent questions -- is the
ANCHOR there, and is the DATA there -- and p001 managed to fail each of them
separately, in different sessions.

Expected format (a good export):

    graph-template.gtl.acq
    Graph Created on: 2026-08-05 09:09:56.385
    Recording on: 2026-08-05 09:48:58.268      <- the sync anchor
    5 msec/sample
    1 channels
    Clench Force
    kg
    sec,CH2,
    ,48426 samples,
    0,0.26024,
    0.005,0.26024,
    ...

Stdlib only, deliberately: this runs on a collection day, and a tool whose
job is to catch problems must not be able to fail for its own reasons.

Usage:
    python tools/check_export.py <path>

<path> may be any of:
    a dynamometer.txt file
    a session directory      (session3_20260805_095947/)
    a participant directory  (recordings/participant001_.../) -- checks all
                             sessions and prints a summary table

Exit code is 0 only if every checked file passed.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPORT_NAME = "dynamometer.txt"

# A session runs ~2.5-4 min. p001's real exports were 242-260 s. Anything
# under a minute is not a short session, it is a broken one -- but it is
# reported as a warning rather than a failure, because a genuinely aborted
# session is a legitimate thing to have on disk.
MIN_PLAUSIBLE_DURATION_S = 60.0


@dataclass
class Report:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recording_on: str | None = None
    declared_samples: int | None = None
    actual_rows: int = 0
    sample_period_s: float | None = None
    duration_s: float | None = None
    force_min: float | None = None
    force_max: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_data_row(line: str) -> bool:
    """A data row is 'time,value,' -- two parseable floats. Used to detect a
    missing header: if line 1 already parses as data, the header is gone."""
    parts = [p for p in line.split(",") if p.strip()]
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def check_export(path: Path) -> Report:
    rep = Report(path=path)

    if not path.exists():
        rep.errors.append(f"file does not exist: {path}")
        return rep
    if path.stat().st_size == 0:
        rep.errors.append("file is empty (0 bytes)")
        return rep

    # errors="replace": BSL writes plain ASCII, but a decode crash here would
    # defeat the point of the tool.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        rep.errors.append("file has no lines")
        return rep

    # --- the anchor -------------------------------------------------------
    # Scan only the head: the header is always the first handful of lines,
    # and a stray "Recording on:" deeper in the file would not be a header.
    header_zone = lines[:20]
    if _is_data_row(lines[0]):
        rep.errors.append(
            "NO HEADER -- file starts with a data row. This is session 1's "
            "failure: the force samples are fine but 'Recording on:' is "
            "missing, so there is no way to sync this to the camera. "
            "Re-export WITH the header."
        )
    else:
        for line in header_zone:
            if line.strip().lower().startswith("recording on:"):
                rep.recording_on = line.split(":", 1)[1].strip()
                break
        if rep.recording_on is None:
            rep.errors.append(
                "header present but no 'Recording on:' line -- this is the "
                "only camera/Biopac sync anchor. Re-export."
            )

    for line in header_zone:
        stripped = line.strip()
        if "msec/sample" in stripped:
            try:
                rep.sample_period_s = float(stripped.split()[0]) / 1000.0
            except (ValueError, IndexError):
                rep.warnings.append(f"could not parse sample rate from {stripped!r}")
        if "samples," in stripped:
            for part in stripped.split(","):
                token = part.strip().split(" ")[0]
                if token.isdigit():
                    rep.declared_samples = int(token)
                    break

    # --- the data ---------------------------------------------------------
    for line in lines:
        if _is_data_row(line):
            rep.actual_rows += 1
            parts = [p for p in line.split(",") if p.strip()]
            try:
                value = float(parts[1])
            except (ValueError, IndexError):
                continue
            if rep.force_min is None or value < rep.force_min:
                rep.force_min = value
            if rep.force_max is None or value > rep.force_max:
                rep.force_max = value

    if rep.actual_rows == 0:
        rep.errors.append(
            "ZERO DATA ROWS -- this is session 4's failure. The header was "
            "written but no samples followed. Re-export NOW, while the "
            "participant is still mounted."
        )
    elif rep.declared_samples is not None and rep.declared_samples != rep.actual_rows:
        rep.errors.append(
            f"header declares {rep.declared_samples} samples but the file "
            f"contains {rep.actual_rows} data rows -- truncated export."
        )

    if rep.sample_period_s and rep.actual_rows:
        rep.duration_s = rep.actual_rows * rep.sample_period_s
        if rep.duration_s < MIN_PLAUSIBLE_DURATION_S:
            rep.warnings.append(
                f"only {rep.duration_s:.1f} s of data -- a full session is "
                f"~2.5-4 min. Was this session aborted?"
            )

    if rep.actual_rows and rep.force_min is not None and rep.force_max is not None:
        if rep.force_max == rep.force_min:
            rep.errors.append(
                f"force is constant at {rep.force_min} for every sample -- "
                "dead channel or unplugged transducer."
            )

    return rep


def print_report(rep: Report) -> None:
    status = "PASS" if rep.ok else "FAIL"
    print(f"\n[{status}] {rep.path}")
    print(f"  recording on : {rep.recording_on or '-- MISSING --'}")
    rows = f"{rep.actual_rows}"
    if rep.declared_samples is not None:
        rows += f" (header declares {rep.declared_samples})"
    print(f"  data rows    : {rows}")
    if rep.duration_s is not None:
        print(f"  duration     : {rep.duration_s:.1f} s")
    if rep.force_min is not None:
        print(f"  force range  : {rep.force_min:.4f} .. {rep.force_max:.4f} kg")
    for err in rep.errors:
        print(f"  ERROR   {err}")
    for warn in rep.warnings:
        print(f"  WARNING {warn}")


def _session_sort_key(path: Path) -> tuple[int, str]:
    """sessionN_<ts> sorts numerically, so session10 lands after session9."""
    name = path.name
    digits = ""
    for char in name[len("session"):]:
        if not char.isdigit():
            break
        digits += char
    return (int(digits) if digits else 0, name)


def resolve_targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / EXPORT_NAME
    if direct.exists():
        return [direct]
    sessions = sorted(
        (d for d in path.iterdir() if d.is_dir() and d.name.startswith("session")),
        key=_session_sort_key,
    )
    return [d / EXPORT_NAME for d in sessions]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    root = Path(argv[1])
    if not root.exists():
        print(f"path does not exist: {root}")
        return 2

    targets = resolve_targets(root)
    if not targets:
        print(f"no {EXPORT_NAME} found under {root}")
        return 2

    reports = [check_export(t) for t in targets]
    for rep in reports:
        print_report(rep)

    failed = [r for r in reports if not r.ok]
    print(f"\n{'=' * 60}")
    if len(reports) > 1:
        print(f"{len(reports) - len(failed)}/{len(reports)} passed")
    if failed:
        print("FAILED:")
        for rep in failed:
            print(f"  {rep.path}")
        print("\nRe-export before the participant unmounts.")
        return 1
    print("All exports OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
