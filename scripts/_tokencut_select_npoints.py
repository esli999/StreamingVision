#!/usr/bin/env python
"""Self-supervised TRAIN selection of the TokenCut SAM2-reprompt point count
(``n_points``) — closes the last residual leak (n_points was fixed from held-out
observation, not selected on TRAIN like the rest of the recipe).

SELF-SUPERVISED, TRAIN-ONLY: scores each candidate by the fired reprompt's
agreement with the TokenCut posterior (NO GT, NO held-out — `select_n_points`
reads neither). Writes the winner into runs/.../tokencut_knobs.json. After this,
update configs/streaming_general.yaml's `n_points` to match and re-validate via
`--phase validate_augmentation` to get the honest held-out number under the
TRAIN-selected knob.

Run:  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python scripts/_tokencut_select_npoints.py
"""
import os, sys, json
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "genmatterpp"))
import calibrate_consistency as cc      # noqa: E402
import tokencut                         # noqa: E402

CANDIDATES = (2, 3, 5, 8)
KNOBS_PATH = cc.OUT_ROOT / "tokencut_knobs.json"


def main():
    cc._ensure_jax_setup()
    # Load the current self-supervised recipe (everything except n_points is already
    # TRAIN-selected by tokencut.select_knobs); we re-select ONLY n_points here.
    knobs = json.loads(KNOBS_PATH.read_text()) if KNOBS_PATH.is_file() else dict(tokencut.DEFAULT_KNOBS)
    print(f"current knobs: {knobs}\ncandidates: {CANDIDATES}\n")

    # TRAIN videos only (discipline). No-op videos simply never fire and contribute
    # nothing; the fire videos (breakdance/dance-twirl/india/mbike-trick) drive it.
    video_labels = {}
    for vid in cc.TRAIN_VIDEOS:
        assert vid in cc.TRAIN_VIDEOS
        if (cc.LABELS_DIR / f"{vid}.npz").is_file():
            try:
                video_labels[vid] = cc._load_labels(vid)
            except Exception as e:
                print(f"  skip {vid}: {e}")
    print(f"selecting over {len(video_labels)} TRAIN videos (self-supervised, no GT)\n")

    best, table = tokencut.select_n_points(video_labels, knobs, candidates=CANDIDATES)

    print("\n=== self-supervised n_points selection (TRAIN, reprompt-vs-q agreement) ===")
    for k in CANDIDATES:
        m = table["means"][str(k)]; nf = table["n_fired"][str(k)]
        ms = f"{m:.4f}" if m == m else "  nan"
        star = "  <== selected" if k == best else ""
        print(f"  n_points={k:>2d}: mean agreement={ms}  (fired on {nf} TRAIN videos){star}")
    print(f"\nSELECTED n_points = {best}  (was {knobs.get('n_points')})")

    knobs["n_points"] = int(best)
    KNOBS_PATH.write_text(json.dumps(knobs, indent=2))
    (cc.OUT_ROOT / "select_npoints.json").write_text(json.dumps(table, indent=2))
    print(f"wrote {KNOBS_PATH} (n_points={best}) + select_npoints.json")
    if best != 8:
        print("\nNOTE: selection differs from the held-out-informed n_points=8 — update "
              "configs/streaming_general.yaml + streaming_render_v2.yaml n_points to "
              f"{best} and re-run --phase validate_augmentation --force.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
