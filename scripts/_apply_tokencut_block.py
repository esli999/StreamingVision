#!/usr/bin/env python
"""Idempotently re-apply the self-supervised TokenCut seed-augmentation block to a
freshly-emitted streaming_general.yaml (Step 2 of the end-to-end run).

The held-out STRUCTURAL gate (phase_validate) deliberately emits an UN-augmented
config (its chosen_infer has tokencut_seed_augment=False) so the structural
comparison is fair. The augmentation is a separate ADDITIVE seed layer whose knobs
are TRAIN-selected self-supervised (recipe via tokencut.select_knobs; n_points via
tokencut.select_n_points) — it is NOT re-fit per run. This script injects

    tracking.tokencut_seed_augment: true
    tracking.tokencut_knobs:        <runs/.../tokencut_knobs.json>

into the target config, preserving the leading header comments, and appends a
short sentinel-wrapped provenance note. Idempotent: re-running on a config that
already carries the block (e.g. a restored backup) reproduces the same result
(the note is replaced, not duplicated). Float hypers round-trip exactly through
yaml (repr-faithful), and sort_keys=False preserves key order.

Run:  python scripts/_apply_tokencut_block.py
        [--config configs/streaming_general.yaml]
        [--knobs  runs/calibrate_consistency/tokencut_knobs.json]
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_NOTE_BEGIN = "# >>> tokencut-aug-note >>>"
_NOTE_END = "# <<< tokencut-aug-note <<<"


def _split_header(text):
    """Return (header_lines, body_text): leading comment/blank lines vs YAML body.
    The body starts at the first non-blank, non-comment line (e.g. ``paths:``)."""
    lines = text.splitlines()
    cut = len(lines)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and not s.startswith("#"):
            cut = i
            break
    return lines[:cut], "\n".join(lines[cut:])


def _strip_prior_note(header):
    """Drop any existing sentinel-wrapped aug-note block so re-runs don't stack."""
    out, skip = [], False
    for ln in header:
        if ln.strip() == _NOTE_BEGIN:
            skip = True
            continue
        if ln.strip() == _NOTE_END:
            skip = False
            continue
        if not skip:
            out.append(ln)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(_REPO / "configs/streaming_general.yaml"))
    p.add_argument("--knobs",
                   default=str(_REPO / "runs/calibrate_consistency/tokencut_knobs.json"))
    args = p.parse_args(argv)
    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    knobs_path = Path(args.knobs)
    if not knobs_path.is_file():
        print(f"ERROR: knobs not found: {knobs_path}", file=sys.stderr)
        return 1
    knobs = json.loads(knobs_path.read_text())

    text = cfg_path.read_text()
    header, body = _split_header(text)
    cfg = yaml.safe_load(body)
    if not isinstance(cfg, dict) or "tracking" not in cfg or cfg["tracking"] is None:
        print(f"ERROR: {cfg_path} has no usable tracking: block", file=sys.stderr)
        return 1

    trk = cfg["tracking"]
    already = bool(trk.get("tokencut_seed_augment")) and trk.get("tokencut_knobs") == knobs
    trk["tokencut_seed_augment"] = True
    trk["tokencut_knobs"] = knobs

    note = [
        _NOTE_BEGIN,
        "# TokenCut self-supervised seed augmentation (additive seed layer; re-applied",
        "# post-emit by scripts/_apply_tokencut_block.py). ALL knobs TRAIN-selected, no",
        "# GT / no held-out: recipe via tokencut.select_knobs (SAM-agreement), n_points",
        "# via tokencut.select_n_points (reprompt-vs-q agreement).",
        f"#   n_points = {knobs.get('n_points')}  (SAM2 reprompt; base-invariant -> reused, not re-fit)",
        "# A pure-JAX normalized-cut object discovery locates objects SAM2 MISSES at the",
        "# seed frame; multi-point SAM2 re-prompts for a crisp mask, painted ADDITIVELY",
        "# only into SAM-uncovered cells -> working/demo videos are bit-identical no-ops.",
        "# Held-out increment measured ONLY inside the locked gate (phase_validate_",
        "# augmentation; cc._heldout_gate_enabled). Numbers + per-video table in",
        "# runs/calibrate_consistency/END_TO_END_REPORT.md.",
        "# Default-OFF in streaming_default.yaml (live path bit-exact).",
        _NOTE_END,
    ]
    new_header = _strip_prior_note(header) + note + [""]
    new_body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    cfg_path.write_text("\n".join(new_header) + new_body)
    status = "(idempotent no-op; already present) " if already else ""
    print(f"{status}applied tokencut block (n_points={knobs.get('n_points')}, "
          f"seed_augment=true) -> {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
