"""MS_THRESHOLD under the Edenhofer dust_prior.

The ms+ms threshold of 59 (src/spbin/hrd.py) was set on the 154 training and
tested on the 197 independent RV-constant APOGEE dwarfs of
docs/validation-whitedwarf.md: fit_star(region="ms", feh=None,
fit_parallax=True) on XP61 + accepted 2MASS JHKs with free age, the training
controls under their Edenhofer moments with the width floored at 0.01, the
independent controls under a 0 +- 0.03 Gaussian E prior. This script refits
both sets with the dwarf and ms+ms hypotheses under the dust prior:

  floor01  Edenhofer moments, width at least 0.01 (training calibration)
  gauss    0 +- 0.03 Gaussian E prior (independent calibration)
  floor    Edenhofer moments, width at least 0.005
  map      Edenhofer moments at the map's own width, as fit_star gets them

Each control runs its calibration configuration plus floor and map.

With MS_SAMPLE=sb2 the script measures the completeness on the 755 Gaia DR3
SB2s that the calibration run placed in ms instead: their calibration
configuration is gauss, the dust configuration map, and summary gives the
fraction above 59, 76 and 102 overall and in bins of the RV mass ratio.

    python scripts/ms_threshold_dust.py inputs   # SEDs and training moments (local data)
    python scripts/ms_threshold_dust.py dust     # Edenhofer moments still missing (needs the map)
    python scripts/ms_threshold_dust.py fit      # WORKERS processes -> fits.jsonl, resumes
    python scripts/ms_threshold_dust.py reseed   # single-star refits of the exceedances
    python scripts/ms_threshold_dust.py summary  # -> summary.json

Paths come from MS_DATA (data/whitedwarf), MS_TRAIN_DUST (training moments),
MS_HOLDOUT_DUST (optional precomputed holdout moments), MS_CONTROLS (the
validation_ms_*.jsonl calibration fits), MS_SAMPLE (controls or sb2) and
MS_WORK.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import beta  # noqa: E402

from spbin.extinction import EdenhoferPrior  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MS_DATA", ROOT / "data/whitedwarf"))
TRAIN_DUST = Path(os.environ.get("MS_TRAIN_DUST", ROOT / "reports/wd_plan_20261010/fgk_dust_moments.json"))
HOLDOUT_DUST = os.environ.get("MS_HOLDOUT_DUST")
CONTROLS = Path(os.environ.get("MS_CONTROLS", DATA))
WORK = Path(os.environ.get("MS_WORK", ROOT / "reports/ms_threshold_dust/work"))
WORKERS = int(os.environ.get("WORKERS", 4))
LIMIT = int(os.environ.get("LIMIT", 0))
SAMPLE = os.environ.get("MS_SAMPLE", "controls")
if SAMPLE == "sb2":
    SAMPLES = dict(sb2="sb2")
    CALIBRATION = dict(sb2="gauss")
    DUST_CONFIGS = ("map",)
else:
    SAMPLES = dict(training="fgk", independent="fgk_holdout")
    CALIBRATION = dict(training="floor01", independent="gauss")
    DUST_CONFIGS = ("floor", "map")
FLOOR = dict(floor01=.01, floor=.005, map=0.)
HYPOTHESES = ("dwarf", "ms+ms")
THRESHOLD = 59.0
THRESHOLDS = (59.0, 76.0, 102.0)
Q_BINS = (("<0.8", 0., .8), ("0.8-0.9", .8, .9), ("0.9-0.95", .9, .95), (">0.95", .95, np.inf))
FALSE_POSITIVE = .02
DWARF_LOGG = 4.1
MAP_EDGE_PC = 70.


class CachedDust:
    """Edenhofer moments interpolated along one line of sight, width at least floor."""

    def __init__(self, record, floor):
        self.d, self.mean, self.sigma = (np.asarray(record[k], float) for k in ("d", "mean", "sigma"))
        self.floor = floor

    def moments(self, ra, dec, distance):
        return (float(np.interp(distance, self.d, self.mean)),
                max(self.floor, float(np.interp(distance, self.d, self.sigma))))

    penalty = staticmethod(EdenhoferPrior.penalty)


def build_sed(sid, gaia, flux, error, tmass):
    """XP61 + accepted 2MASS JHKs, as validate_ms.py builds them."""
    from spbin import SED
    from spbin.fetch import _photometry
    from spbin.model import wavelength_um
    f, e, mask = np.full(168, np.nan), np.full(168, np.nan), np.zeros(168, bool)
    f[:61], e[:61] = flux[6:], error[6:]
    mask[:61] = np.isfinite(f[:61]) & np.isfinite(e[:61]) & (e[:61] > 0)
    pf, pe, pm, _ = _photometry([("2MASS", tmass, range(3), ("j_m", "h_m", "ks_m"),
                                  ("j_msigcom", "h_msigcom", "ks_msigcom"))], wavelength_um()[61:66])
    f[61:64], e[61:64], mask[61:64] = pf[:3], pe[:3], pm[:3]
    metadata = {k: gaia[k] for k in ("ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag")}
    return SED(f, e, mask, gaia["parallax"], gaia["parallax_error"], str(sid), metadata)


def inputs():
    """The 154 / 197 controls (region ms, spectroscopic log g > 4.1 in the
    calibration run) or the 755 SB2s (region ms): SEDs, calibration fits and
    the training moments."""
    from astropy.table import Table
    WORK.mkdir(parents=True, exist_ok=True)
    seds, controls = [], []
    for sample, kind in SAMPLES.items():
        rows = [json.loads(line) for line in (CONTROLS / f"validation_ms_{kind}.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["location"]["region"] == "ms" and r.get("logg_spec", np.inf) > DWARF_LOGG]
        directory = DATA / f"xp_{kind}"
        gaia = {int(r["source_id"]): {k: np.nan if np.ma.is_masked(r[k]) else float(r[k]) for k in r.colnames}
                for r in Table.read(directory / "gaia.ecsv")}
        tmass = Table.read(directory / "tmass_ms.ecsv")
        tmass_ids = np.asarray(tmass["source_id"], np.int64)
        with np.load(directory / "calibrated.npz") as a:
            index = {int(v): i for i, v in enumerate(a["source_id"])}
            flux, error = a["flux"], a["error"]
        for r in rows:
            sid = int(r["source_id"])
            seds.append(build_sed(sid, gaia[sid], flux[index[sid]], error[index[sid]], tmass[tmass_ids == sid]))
            labels = {k: r[k] for k in ("logg_spec", "feh_spec", "q_rv", "period") if k in r}
            controls.append(dict(source_id=str(sid), sample=sample, **labels,
                                 calibration={h: r["free"][h] for h in HYPOTHESES}))
        n_fit = [int(s.mask.sum()) - r["n_fit"] for s, r in zip(seds[-len(rows):], rows) if "n_fit" in r]
        print(sample, len(rows), "stars;", sum(n != 0 for n in n_fit), "with another channel count than the calibration run",
              flush=True)
    np.savez_compressed(WORK / "seds.npz", flux=np.stack([s.flux for s in seds]),
                        error=np.stack([s.error for s in seds]), mask=np.stack([s.mask for s in seds]),
                        parallax=np.array([s.parallax_mas for s in seds]),
                        parallax_error=np.array([s.parallax_error_mas for s in seds]),
                        source_id=np.array([s.source_id for s in seds]),
                        metadata=np.array([json.dumps(s.metadata) for s in seds]))
    (WORK / "controls.json").write_text(json.dumps(controls, indent=1) + "\n")
    train = json.loads(TRAIN_DUST.read_text()) if "training" in SAMPLES else {}
    dust = {c["source_id"]: train[c["source_id"]] for c in controls if c["source_id"] in train}
    (WORK / "dust.json").write_text(json.dumps(dust) + "\n")
    print(len(dust), "of", len(controls), "with moments", flush=True)


def load_seds():
    from spbin import SED
    with np.load(WORK / "seds.npz") as a:
        return {str(sid): SED(a["flux"][i], a["error"][i], a["mask"][i], float(a["parallax"][i]),
                              float(a["parallax_error"][i]), str(sid), json.loads(str(a["metadata"][i])))
                for i, sid in enumerate(a["source_id"])}


def dust():
    """Edenhofer mean and width along each line of sight that lacks them, on
    distances 1000 / (parallax + z sigma), z in [-3.2, 3.2], inside the map;
    distances inside its inner edge take the moments at MAP_EDGE_PC, as the
    holdout moments do."""
    moments = json.loads((WORK / "dust.json").read_text())
    if HOLDOUT_DUST:
        moments.update({k: v for k, v in json.loads(Path(HOLDOUT_DUST).read_text()).items() if k not in moments})
    seds = load_seds()
    todo = [sid for sid in seds if sid not in moments]
    prior = EdenhoferPrior() if todo else None
    for i, sid in enumerate(todo):
        sed = seds[sid]
        parallax = sed.parallax_mas + np.linspace(-3.2, 3.2, 33) * sed.parallax_error_mas
        record = dict(d=[], mean=[], sigma=[])
        for d in np.unique(np.maximum(MAP_EDGE_PC, 1000 / parallax[parallax > 0])):
            try:
                mean, sigma = prior.moments(sed.metadata["ra"], sed.metadata["dec"], d)
            except ValueError:
                continue
            record["d"].append(float(d)); record["mean"].append(mean); record["sigma"].append(sigma)
        if record["d"]:
            moments[sid] = record
        if (i + 1) % 20 == 0:
            print(f"dust {i + 1}/{len(todo)}", flush=True)
    moments = {sid: moments[sid] for sid in seds if sid in moments}
    (WORK / "dust.json").write_text(json.dumps(moments) + "\n")
    print(len(moments), "of", len(seds), "with moments", flush=True)


def nuisance(config, record):
    if config == "gauss":
        return dict(extinction_prior=(0., .03), dust_prior=None)
    return dict(extinction_prior=None, dust_prior=CachedDust(record, FLOOR[config]))


def _float(value):
    return None if value is None or not np.isfinite(value) else float(value)


def summarize_fit(h, sed, record):
    c = h["companion"]
    out = dict(objective=h["objective"], chi2=h["chi2"], minus2lnL=h["minus2lnL"], penalty=h["penalty"],
               converged=h["converged"], extinction_e=h["extinction_e"], parallax_mas=h["parallax_mas"],
               z=(h["parallax_mas"] - sed.parallax_mas) / sed.parallax_error_mas,
               **{k: c[k] for k in ("teff", "mass", "logg", "age_gyr", "feh", "q", "m2")})
    if record is not None:
        out["map"] = CachedDust(record, 0.).moments(None, None, 1000 / h["parallax_mas"])
    return out


def one(task):
    from spbin.hrd import locate
    from spbin.whitedwarf import fit_whitedwarf_companion
    control, config, sed, record = task
    out = dict(source_id=control["source_id"], sample=control["sample"], config=config)
    try:
        result = fit_whitedwarf_companion(sed, hypotheses=HYPOTHESES, companion_age_gyr=None, companion_feh=None,
                                          extinction=None, fit_parallax=True, **nuisance(config, record))
        out.update({h: summarize_fit(result["hypotheses"][h], sed, record) for h in HYPOTHESES})
        if config == "map":
            out["region_at_map_mean"] = locate(sed, dust_prior=CachedDust(record, 0.))["region"]
    except ValueError as reason:
        out["failed"] = str(reason)
    return json.loads(json.dumps(out, default=_float))


def read_fits():
    rows = [json.loads(line) for line in (WORK / "fits.jsonl").read_text().splitlines() if line.strip()]
    return {(r["source_id"], r["config"]): r for r in rows}


def fit():
    controls = json.loads((WORK / "controls.json").read_text())
    moments = json.loads((WORK / "dust.json").read_text())
    seds = load_seds()
    path = WORK / "fits.jsonl"
    done = set(read_fits()) if path.exists() else set()
    if LIMIT:
        controls = controls[:LIMIT]
    todo = [(c, config, seds[c["source_id"]], moments.get(c["source_id"]))
            for c in controls for config in (CALIBRATION[c["sample"]],) + DUST_CONFIGS
            if (c["source_id"], config) not in done]
    missing = {t[0]["source_id"] for t in todo if t[3] is None}
    if missing:
        raise SystemExit(f"{len(missing)} controls have no Edenhofer moments; run the dust stage")
    print(len(todo), "fits", flush=True)
    with ProcessPoolExecutor(WORKERS) as pool, path.open("a") as handle:
        for i, out in enumerate(pool.map(one, todo, chunksize=1)):
            handle.write(json.dumps(out) + "\n")
            handle.flush()
            if (i + 1) % 50 == 0:
                print(f"fit {i + 1}/{len(todo)}", flush=True)
    print("fit finished", len(todo), flush=True)


def statistic(row):
    return row["dwarf"]["objective"] - row["ms+ms"]["objective"]


def reseed_one(task):
    """Minimise the dwarf again under the same prior by Nelder-Mead from its
    calibration solution, that solution with E at the map mean, and the pair's
    primary, each with the frame's E step of 0.02 and with an E step equal to
    the prior width, and from the eight best points of a grid in mass, age,
    [M/H] and parallax with E at the prior mean; then fit the pair again from
    the best dwarf."""
    from spbin.whitedwarf import AGE_SEEDS, SEED_MASSES, _Hypothesis, _prepare
    config, row, reference, sed, record = task
    n = nuisance(config, record)
    p = _prepare(sed, None, None, None, n["extinction_prior"], n["dust_prior"], None, True,
                 None, None, False, None, None)
    single = _Hypothesis(p, "dwarf", False, None, None)

    def start(fit):
        return dict(primary=np.log(fit["mass"]), age=np.log(fit["age_gyr"]), feh=fit["feh"],
                    e=fit["extinction_e"], z=fit["z"])

    def width(v):
        if config == "gauss":
            return .03
        return p.dust.moments(None, None, 1000 / (sed.parallax_mas + v["z"] * sed.parallax_error_mas))[1]

    def polish(v, e_step):
        steps = dict(primary=.05, age=.2, feh=.1, e=e_step, z=.2)
        bounds = [single.bounds[k] for k in single.names]
        x0 = np.array([np.clip(v[k], *b) for k, b in zip(single.names, bounds)])
        result = minimize(single.objective, x0, method="Nelder-Mead", bounds=bounds,
                          options=dict(initial_simplex=np.vstack([x0, x0 + np.diag([steps[k] for k in single.names])]),
                                       maxfev=4000, xatol=1e-5, fatol=1e-4))
        return float(result.fun), single.unpack(result.x)

    calibration = start(reference["dwarf"])
    e_map = CachedDust(record, 0.).moments(None, None, 1000 / reference["dwarf"]["parallax_mas"])[0]
    starts = {"calibration": calibration, "calibration, E at map mean": dict(calibration, e=e_map),
              "pair primary": start(row["ms+ms"])}
    fits = {}
    for name, v in starts.items():
        for label, e_step in (("0.02", .02), ("prior width", width(v))):
            fits[f"{name}, E step {label}"] = polish(v, e_step)

    def prior_mean(z):
        if config == "gauss":
            return 0.
        return p.dust.moments(None, None, 1000 / (sed.parallax_mas + z * sed.parallax_error_mas))[0]

    grid = [dict(primary=np.log(m), age=np.log(a), feh=feh, e=prior_mean(z), z=z)
            for m in SEED_MASSES["ms"] for a in AGE_SEEDS["ms"] for feh in (-.6, -.3, 0., .3) for z in (-1., 0., 1.)]
    ranked = sorted(grid, key=lambda v: single.objective([v[k] for k in single.names]))
    for i, v in enumerate(ranked[:8]):
        fits[f"grid {i}, E step prior width"] = polish(v, width(v))
    seed = min(fits, key=lambda name: fits[name][0])
    s_objective, s_values = fits[seed]
    pair = _Hypothesis(p, "ms+ms", False, None, None)
    p_summary, _ = pair.fit(initial=s_values)
    single_best = min(row["dwarf"]["objective"], s_objective)
    pair_best = min(row["ms+ms"]["objective"], p_summary["objective"])
    return dict(source_id=row["source_id"], sample=row["sample"], config=config, teff=row["dwarf"]["teff"],
                stat=statistic(row), single=row["dwarf"]["objective"], single_reseeded=s_objective, seed=seed,
                single_by_seed={name: f[0] for name, f in fits.items()},
                single_reseeded_values={k: float(v) for k, v in s_values.items()},
                single_at_calibration=float(single.objective([calibration[k] for k in single.names])),
                pair=row["ms+ms"]["objective"], pair_reseeded=p_summary["objective"],
                stat_reseeded=single_best - pair_best)


def reseed():
    fits = read_fits()
    moments = json.loads((WORK / "dust.json").read_text())
    seds = load_seds()
    tasks = [(r["config"], r, fits[(r["source_id"], CALIBRATION[r["sample"]])], seds[r["source_id"]],
              moments[r["source_id"]])
             for r in fits.values() if "failed" not in r and statistic(r) > THRESHOLD]
    print(len(tasks), "refits", flush=True)
    with ProcessPoolExecutor(WORKERS) as pool:
        out = list(pool.map(reseed_one, tasks, chunksize=1))
    (WORK / "reseed.json").write_text(json.dumps(out, indent=1) + "\n")
    for o in sorted(out, key=lambda o: (o["sample"], o["config"], -o["stat"])):
        print(f"{o['sample']:11s} {o['config']:7s} {o['source_id']} Teff {o['teff']:5.0f} stat {o['stat']:7.1f} -> "
              f"{o['stat_reseeded']:7.1f}  dwarf {o['single']:8.1f} -> {o['single_reseeded']:8.1f} ({o['seed']}; "
              f"at calibration {o['single_at_calibration']:8.1f})  pair {o['pair']:8.1f} -> {o['pair_reseeded']:8.1f}")


def upper95(k, n):
    return float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.


def rule(values):
    """First integer above the (k+1)-th largest value, k = floor(2% of N)."""
    values = np.sort(values)[::-1]
    return float(np.floor(values[int(np.floor(FALSE_POSITIVE * len(values)))]) + 1)


def percentiles(x, q=(2, 16, 50, 84, 98)):
    return [float(v) for v in np.percentile(x, q)] if len(x) else []


def summary():
    controls = json.loads((WORK / "controls.json").read_text())
    fits = read_fits()
    reseeded = {}
    if (WORK / "reseed.json").exists():
        reseeded = {(o["source_id"], o["config"]): o for o in json.loads((WORK / "reseed.json").read_text())}
    out = dict(threshold=THRESHOLD, samples={})
    for sample in SAMPLES:
        ids = [c["source_id"] for c in controls if c["sample"] == sample]
        stored = {c["source_id"]: c["calibration"]["dwarf"]["objective"] - c["calibration"]["ms+ms"]["objective"]
                  for c in controls if c["sample"] == sample}
        block = dict(n=len(ids), configs={})
        for config in (CALIBRATION[sample],) + DUST_CONFIGS:
            rows = [fits[(sid, config)] for sid in ids if (sid, config) in fits]
            ok = [r for r in rows if "failed" not in r]
            stat = np.array([statistic(r) for r in ok])
            after = np.array([reseeded[(r["source_id"], config)]["stat_reseeded"]
                              if (r["source_id"], config) in reseeded else statistic(r) for r in ok])
            k, k_after = int(np.sum(stat > THRESHOLD)), int(np.sum(after > THRESHOLD))
            c = dict(n=len(ok), failed=len(rows) - len(ok),
                     above=dict(k=k, fraction=k / len(ok), upper95=upper95(k, len(ok))),
                     above_reseeded=dict(k=k_after, fraction=k_after / len(ok), upper95=upper95(k_after, len(ok))),
                     rule=rule(stat), rule_reseeded=rule(after),
                     top=sorted(stat.tolist(), reverse=True)[:6], top_reseeded=sorted(after.tolist(), reverse=True)[:6],
                     percentiles=percentiles(stat),
                     converged={h: int(sum(r[h]["converged"] for r in ok)) for h in HYPOTHESES})
            if config == CALIBRATION[sample]:
                diff = np.array([statistic(r) - stored[r["source_id"]] for r in ok])
                c["minus_calibration_run"] = percentiles(diff, (0, 2, 50, 98, 100))
                c["e_minus_map_mean"] = percentiles([r["dwarf"]["extinction_e"] - r["dwarf"]["map"][0] for r in ok])
            else:
                for h in HYPOTHESES:
                    offset = np.array([r[h]["extinction_e"] - r[h]["map"][0] for r in ok])
                    sigma = np.array([max(FLOOR[config], r[h]["map"][1]) for r in ok])
                    c[f"e_offset_{h}"] = percentiles(offset)
                    c[f"e_pull_{h}"] = percentiles(offset / sigma)
                c["map_width"] = percentiles([r["dwarf"]["map"][1] for r in ok])
                calibration = {r["source_id"]: r for r in (fits.get((sid, CALIBRATION[sample])) for sid in ids)
                               if r is not None and "failed" not in r}
                pairs = [(r, calibration[r["source_id"]]) for r in ok if r["source_id"] in calibration]
                c["minus_calibration_config"] = percentiles([statistic(r) - statistic(q) for r, q in pairs])
                c["dwarf_objective_minus_calibration"] = percentiles(
                    [r["dwarf"]["objective"] - q["dwarf"]["objective"] for r, q in pairs])
                if config == "map":
                    regions = [r.get("region_at_map_mean") for r in ok]
                    c["region_at_map_mean"] = {g: regions.count(g) for g in sorted(set(regions))}
                c["exceeding"] = [dict(source_id=r["source_id"], teff=r["dwarf"]["teff"], stat=statistic(r),
                                       calibration_stat=statistic(calibration[r["source_id"]]),
                                       reseeded=reseeded.get((r["source_id"], config), {}).get("stat_reseeded"))
                                  for r in ok if statistic(r) > THRESHOLD]
            block["configs"][config] = c
        out["samples"][sample] = block
    (WORK / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def interval(k, n):
    """Two-sided 95% Clopper-Pearson interval."""
    return [float(beta.ppf(.025, k, n - k + 1)) if k > 0 else 0., float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.]


def completeness():
    """Fraction of the SB2s above each threshold, overall and in bins of the
    RV mass ratio, with the fitted q minus the RV ratio of those above."""
    controls = {c["source_id"]: c for c in json.loads((WORK / "controls.json").read_text())}
    fits, seds = read_fits(), load_seds()
    stored = {sid: c["calibration"]["dwarf"]["objective"] - c["calibration"]["ms+ms"]["objective"]
              for sid, c in controls.items()}
    out = dict(n=len(controls), configs={})
    for config in ("gauss",) + DUST_CONFIGS:
        rows = [fits[(sid, config)] for sid in controls if (sid, config) in fits]
        ok = [r for r in rows if "failed" not in r]
        stat = np.array([statistic(r) for r in ok])
        q_rv = np.array([controls[r["source_id"]]["q_rv"] for r in ok])
        q_fit = np.array([r["ms+ms"]["q"] for r in ok])
        c = dict(n=len(ok), failed=len(rows) - len(ok), converged={h: int(sum(r[h]["converged"] for r in ok))
                                                                   for h in HYPOTHESES},
                 percentiles=percentiles(stat), above={})
        for t in THRESHOLDS:
            above = stat > t
            block = {}
            for label, lo, hi in (("all", -np.inf, np.inf),) + Q_BINS:
                k, n = int(np.sum(above & (q_rv >= lo) & (q_rv < hi))), int(np.sum((q_rv >= lo) & (q_rv < hi)))
                block[label] = dict(k=k, n=n, fraction=k / n, interval=interval(k, n))
            block["q_fit_minus_rv"] = percentiles(q_fit[above] - q_rv[above], (5, 50, 95))
            c["above"][f"{t:g}"] = block
        c["e_minus_map_mean"] = {h: percentiles([r[h]["extinction_e"] - r[h]["map"][0] for r in ok]) for h in HYPOTHESES}
        if config == "gauss":
            c["minus_calibration_run"] = percentiles([statistic(r) - stored[r["source_id"]] for r in ok],
                                                     (0, 2, 50, 98, 100))
        else:
            gauss = {r["source_id"]: r for r in (fits.get((sid, "gauss")) for sid in controls)
                     if r is not None and "failed" not in r}
            pairs = [(r, gauss[r["source_id"]]) for r in ok if r["source_id"] in gauss]
            c["minus_gauss"] = percentiles([statistic(r) - statistic(g) for r, g in pairs])
            for h in HYPOTHESES:
                c[f"{h}_objective_minus_gauss"] = percentiles([r[h]["objective"] - g[h]["objective"] for r, g in pairs])
            c["versus_gauss"] = {f"{t:g}": dict(both=sum(statistic(r) > t and statistic(g) > t for r, g in pairs),
                                                only_here=sum(statistic(r) > t >= statistic(g) for r, g in pairs),
                                                only_gauss=sum(statistic(g) > t >= statistic(r) for r, g in pairs))
                                 for t in THRESHOLDS}
            c["map_width"] = percentiles([r["dwarf"]["map"][1] for r in ok])
            regions = [r.get("region_at_map_mean") for r in ok]
            c["region_at_map_mean"] = {g: regions.count(g) for g in sorted(set(regions))}
            in_ms = np.array([g == "ms" for g in regions])
            c["above_in_ms_at_map_mean"] = {f"{t:g}": dict(k=int(np.sum((stat > t) & in_ms)), n=int(in_ms.sum()))
                                            for t in THRESHOLDS}
        beyond = np.array([1000 / seds[r["source_id"]].parallax_mas >= MAP_EDGE_PC for r in ok])
        c["above_beyond_map_edge"] = {f"{t:g}": dict(k=int(np.sum((stat > t) & beyond)), n=int(beyond.sum()))
                                      for t in THRESHOLDS}
        out["configs"][config] = c
    (WORK / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    dict(inputs=inputs, dust=dust, fit=fit, reseed=reseed,
         summary=completeness if SAMPLE == "sb2" else summary)[sys.argv[1]]()
