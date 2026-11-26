"""Regenerate every TrackLens artifact from the raw CSV.

    python pipeline/build.py                       # uses data/raw/dataset.csv
    python pipeline/build.py --csv /path/to/file.csv
    python pipeline/build.py --skip-local          # aggregate artifacts only

Writes
  web/public/data/*.json             aggregate-only, committed, published
  web/local-data/*.json              track-level, git-ignored, local full mode only
  web/src/lib/__fixtures__/parity.json   synthetic parity fixtures (no real records)
  docs/model-card.md, docs/data-notes.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracklens import aggregates, config, data, modeling  # noqa: E402
from tracklens.report import write_data_notes, write_model_card  # noqa: E402


def dump(obj, path: Path, indent: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, separators=None if indent else (",", ":"), ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return len(text.encode())


def log(msg: str) -> None:
    print(msg, flush=True)


def parity_fixtures(spec: dict, est, train: pd.DataFrame) -> dict:
    """Synthetic inputs only: medians, bound edges, seeded random, split thresholds."""
    rng = np.random.default_rng(config.SEED)
    feats = config.MODEL_FEATURES
    med = {}
    for f in feats:
        v = float(train[f].astype(float).median())
        med[f] = round(v) if config.VALIDATION[f]["integer"] else round(v, 4)
    cases = [{"name": "training medians", "x": dict(med)}]
    cases.append({"name": "all minimums", "x": {f: config.VALIDATION[f]["min"] for f in feats}})
    cases.append({"name": "all maximums", "x": {f: config.VALIDATION[f]["max"] for f in feats}})
    for f in feats:
        for edge in ("min", "max"):
            x = dict(med)
            x[f] = config.VALIDATION[f][edge]
            cases.append({"name": f"{f} at {edge}", "x": x})
    for i in range(40):
        x = {}
        for f in feats:
            r = config.VALIDATION[f]
            v = rng.uniform(r["min"], r["max"])
            x[f] = int(round(v)) if r["integer"] else round(float(v), 5)
        cases.append({"name": f"random {i + 1}", "x": x})
    if spec["kind"] == "trees":
        picked = 0
        for ti, t in enumerate(spec["trees"][:50]):
            for ni, fidx in enumerate(t["f"]):
                if fidx < 0:
                    continue
                f = feats[fidx]
                thr = t["t"][ni]
                if config.VALIDATION[f]["integer"]:
                    continue
                for label, val in (("at", thr), ("just above", math.nextafter(thr, math.inf))):
                    x = dict(med)
                    x[f] = val
                    cases.append({"name": f"tree {ti} node {ni}: {f} {label} split", "x": x})
                picked += 1
                break
            if picked >= 12:
                break
    X = pd.DataFrame([c["x"] for c in cases])[feats]
    sk = modeling.clip(est.predict(X))
    out = []
    for c, p in zip(cases, sk):
        vec = [float(c["x"][f]) for f in feats]
        mine = modeling.predict_exported(spec, vec)
        if abs(mine - p) > 1e-9:
            raise AssertionError(f"exported evaluator differs from scikit-learn on {c['name']}: {mine} vs {p}")
        out.append({"name": c["name"], "input": c["x"], "expected": float(p)})

    invalid = [
        {"name": "danceability above 1", "patch": {"danceability": 1.2}, "field": "danceability"},
        {"name": "negative tempo", "patch": {"tempo": -5}, "field": "tempo"},
        {"name": "fractional key", "patch": {"key": 2.5}, "field": "key"},
        {"name": "missing loudness", "patch": {"loudness": None}, "field": "loudness"},
        {"name": "duration zero", "patch": {"duration_ms": 0}, "field": "duration_ms"},
        {"name": "loudness too high", "patch": {"loudness": 12}, "field": "loudness"},
        {"name": "time signature 9", "patch": {"time_signature": 9}, "field": "time_signature"},
    ]
    for case in invalid:
        x = {**med, **case["patch"]}
        problems = modeling.validate_input(x)
        if not any(p.startswith(case["field"]) for p in problems):
            raise AssertionError(f"validation did not reject {case['name']}")
        case["base"] = med
    return {"schemaVersion": config.SCHEMA_VERSION, "tolerance": 1e-9, "valid": out, "invalid": invalid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=config.RAW_CSV)
    ap.add_argument("--skip-local", action="store_true", help="do not write track-level local artifacts")
    args = ap.parse_args()
    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}. Place the dataset at data/raw/dataset.csv or pass --csv.")

    t0 = time.time()
    digest = data.sha256(args.csv)
    log(f"Loading {args.csv} (sha256 {digest[:12]})")
    raw = data.load_raw(args.csv)
    tracks, memberships, rules = data.canonicalise(raw)
    tracks["dup_group"] = data.assign_duplicate_groups(tracks)
    prof = data.profile(raw, tracks, memberships, rules)
    mpop = memberships.merge(tracks[["track_id", "popularity"]], on="track_id")
    prof["zero_popularity_memberships"] = int((mpop["popularity"] == 0).sum())
    dup_sizes = tracks.groupby("dup_group").size()
    prof["duplicate_recording_groups"] = int((dup_sizes > 1).sum())
    prof["tracks_in_duplicate_recording_groups"] = int(dup_sizes[dup_sizes > 1].sum())
    gsize = tracks["dup_group"].map(dup_sizes)
    gmax = tracks.groupby("dup_group")["popularity"].transform("max")
    zero = tracks["popularity"] == 0
    prof["zero_tracks_in_duplicate_groups"] = int((zero & (gsize > 1)).sum())
    prof["zero_tracks_with_nonzero_duplicate"] = int((zero & (gmax > 0)).sum())
    log(f"  {prof['rows']} rows -> {len(tracks)} canonical tracks, {len(memberships)} memberships, {prof['genres']} genres")

    genres = sorted(memberships["track_genre"].unique().tolist())
    valid = tracks[tracks["model_valid"]].reset_index(drop=True)
    sim_X = valid[config.SIMILARITY_FEATURES].to_numpy(float)
    sim_mean, sim_scale = sim_X.mean(axis=0), sim_X.std(axis=0, ddof=0)

    version = {"schemaVersion": config.SCHEMA_VERSION, "dataSha256": digest}
    sizes = {}
    log("Building aggregate artifacts")
    overview = aggregates.build_overview(tracks, memberships, genres)
    sizes["overview.json"] = dump({**overview, **version}, config.PUBLIC_DATA / "overview.json")
    genre_doc = aggregates.build_genres(tracks, memberships, genres, sim_mean, sim_scale)
    sizes["genres.json"] = dump({**genre_doc, **version}, config.PUBLIC_DATA / "genres.json")
    sim_spec = aggregates.build_similarity_spec(sim_mean, sim_scale, len(valid))
    sizes["similarity.json"] = dump({**sim_spec, **version}, config.PUBLIC_DATA / "similarity.json")

    log("Splitting and training")
    valid["split"] = modeling.split(valid).to_numpy()
    frames = {k: valid[valid["split"] == k].reset_index(drop=True) for k in ("train", "validation", "test")}
    crossing = valid.groupby("dup_group")["split"].nunique()
    assert int((crossing > 1).sum()) == 0, "a duplicate group crosses splits"
    name_only = valid["track_name"].map(data.normalise_text)
    train_names = set(name_only[valid["split"] == "train"]) - {""}
    test_names_in_train = int(name_only[valid["split"] == "test"].isin(train_names).sum())
    split_info = {
        "rows": {k: int(len(v)) for k, v in frames.items()},
        "groupsCrossingSplits": 0,
        "excludedInvalidTracks": int((~tracks["model_valid"]).sum()),
        "testTracksWhoseTitleAlsoAppearsInTrainWithDifferentArtistText": test_names_in_train,
        "seed": config.SEED,
        "method": "GroupShuffleSplit 70/15/15 on duplicate-recording groups",
    }
    sel = modeling.train_and_select(frames, log=log)
    evaluation = modeling.evaluate(sel["selected"], sel["best_by_family"], frames, log=log)
    evaluation["split"] = split_info
    evaluation["trials"] = [
        {"family": t["family"], "label": t["label"], "validationMae": t["validation"]["mae"], "fitSeconds": t["fitSeconds"]}
        for t in sel["trials"]
    ]
    evaluation["features"] = config.MODEL_FEATURES
    spec = modeling.export_model(sel["selected"], len(frames["train"]))
    defaults = {}
    for f in config.MODEL_FEATURES:
        v = float(frames["train"][f].astype(float).median())
        defaults[f] = int(round(v)) if config.VALIDATION[f]["integer"] else round(v, 4)
    spec["defaults"] = defaults  # training-split medians, used as the UI starting point
    evaluation["baselineValue"] = float(sel["best_by_family"]["baseline"]["_est"].model.constant_.ravel()[0])
    sizes["model.json"] = dump({**spec, **version}, config.PUBLIC_DATA / "model.json")
    sizes["model-eval.json"] = dump({**evaluation, **version}, config.PUBLIC_DATA / "model-eval.json")

    log("Writing parity fixtures")
    fixtures = parity_fixtures(spec, sel["selected"]["_est"], frames["train"])
    dump({**fixtures, **version}, config.WEB_FIXTURES / "parity.json", indent=1)

    meta = {
        **version,
        "title": "TrackLens",
        "profile": prof,
        "artifacts": sizes,
        "publication": {
            "mode": "aggregate",
            "reason": (
                "The CSV is a third-party compilation of data originally retrieved from Spotify. Its mirrors list "
                "conflicting licences and Spotify's developer terms restrict redistributing Spotify content, so "
                "permission to republish track-level records could not be established. The public site therefore "
                "ships only aggregates, the model parameters and synthetic test fixtures."
            ),
        },
    }
    sizes["meta.json"] = dump(meta, config.PUBLIC_DATA / "meta.json")

    if not args.skip_local:
        log("Writing local track-level artifacts (git-ignored)")
        t = tracks.copy()
        gidx = {g: i for i, g in enumerate(genres)}
        tg = memberships.assign(g=memberships["track_genre"].map(gidx)).groupby("track_id")["g"].apply(lambda s: sorted(int(v) for v in s))
        split_map = dict(zip(valid["track_id"], valid["split"]))
        local = {
            **version,
            "count": int(len(t)),
            "genres": genres,
            "id": t["track_id"].tolist(),
            "name": [v or "" for v in t["track_name"]],
            "artists": [v or "" for v in t["artists"]],
            "album": [v or "" for v in t["album_name"]],
            "popularity": t["popularity"].astype(int).tolist(),
            "genreIdx": [tg[i] for i in t["track_id"]],
            "dupGroup": t["dup_group"].astype(int).tolist(),
            "split": [split_map.get(i, "excluded") for i in t["track_id"]],
            "valid": t["model_valid"].astype(int).tolist(),
            "explicit": t["explicit"].astype(int).tolist(),
        }
        for f in ["duration_ms", "key", "mode", "time_signature"]:
            local[f] = t[f].astype(int).tolist()
        for f in config.UNIT_FEATURES + ["loudness", "tempo"]:
            local[f] = [float(v) for v in t[f]]
        dump(local, config.LOCAL_DATA / "tracks.json")
        dump({"mode": "full", **version}, config.LOCAL_DATA / "manifest.json")

    write_data_notes(prof, split_info, genre_doc, config.DOCS / "data-notes.md")
    write_model_card(evaluation, spec, split_info, prof, config.DOCS / "model-card.md")
    log("Artifact sizes (bytes): " + ", ".join(f"{k}={v:,}" for k, v in sizes.items()))
    log(f"Selected {evaluation['selected']}: test MAE {evaluation['test']['mae']:.3f} vs baseline {evaluation['baselineTest']['mae']:.3f}")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
