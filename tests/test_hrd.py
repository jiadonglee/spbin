from pathlib import Path

import numpy as np
import pytest

from spbin import SED, StellarModel, WhiteDwarfModel
from spbin.extinction import EdenhoferPrior
from spbin.hrd import MS_THRESHOLD, MS_THRESHOLD_DUST, PMS_THRESHOLD, _region, fit_star, locate

FIXTURE = Path(__file__).parent / "fixtures" / "gaia_dr3_5148853253106611200.npz"


@pytest.mark.parametrize("point, region", [
    ((12.0, 0.2, np.nan), "wd"),
    ((4.5, -0.3, np.nan), "sdb"),
    ((5.0, 0.8, 5800.0), "ms"),
    ((8.0, 2.5, 3500.0), "pms"),
    ((5.5, 1.5, 4300.0), "pms"),
    ((5.0, 2.5, 3500.0), "unsupported"),
    ((3.0, 1.0, 4500.0), "giant"),
    ((0.0, 0.3, 7800.0), "hot_ms"),
    ((2.3, 0.4, 7200.0), "ms"),
    ((0.15, 0.2, 7200.0), "hot_ms"),
    ((0.5, 1.2, 4700.0), "giant"),
    ((9.0, 1.0, 5000.0), "unsupported"),
    ((14.0, 4.0, 2500.0), "unsupported"),
    ((-4.0, -0.3, 35000.0), "unsupported"),
    ((-1.0, 2.0, 3400.0), "unsupported"),
])
def test_region_rules(point, region):
    assert _region(*point)[0] == region


def test_locate_real_sb2_on_main_sequence():
    location = locate(SED.load(FIXTURE))
    assert location["region"] == "ms"
    assert location["hypotheses"] == ("dwarf", "ms+ms", "wd+dwarf")
    assert 4000 < location["teff_estimate"] < 5000
    with pytest.raises(ValueError):
        locate(SED(np.ones(168), np.ones(168), np.ones(168, bool), 10.0))


def _mock_binary(m1, q):
    stellar = StellarModel()
    pred = stellar.evaluate(m1, q, 5, 0)
    flux = pred["flux_10pc"] * .01
    g = -2.5 * np.log10(np.sum(10**(-0.4 * pred["M_G"]))) + 5
    metadata = dict(ra=0., dec=0., phot_g_mean_mag=g, phot_bp_mean_mag=g + 0.6, phot_rp_mean_mag=g - 0.6)
    return SED(flux, flux * .01, WhiteDwarfModel().support & stellar.support, 10, .02, "mock", metadata)


def test_main_sequence_binary_fits_three_hypotheses():
    result = fit_star(_mock_binary(.9, .7))
    assert result["region"] == "ms"
    assert set(result["objectives"]) == {"dwarf", "ms+ms", "wd+dwarf"}
    assert result["preferred"] == "ms+ms"
    assert abs(result["result"]["hypotheses"]["ms+ms"]["companion"]["q"] - .7) < .02
    assert result["delta"]["dwarf"] > 100


def test_single_star_is_not_promoted_to_a_composite():
    result = fit_star(_mock_binary(.8, 0))
    assert result["preferred"] == "dwarf"
    assert result["delta"]["ms+ms"] < 2
    assert result["detection_statistic"]["ms+ms"] < 2
    assert result["threshold"] is None


def test_ms_threshold_in_its_calibrated_configuration():
    result = fit_star(_mock_binary(.9, .7), feh=None, fit_parallax=True, extinction=None,
                      extinction_prior=(0., .03))
    statistic, threshold = result["detection_statistic"], result["threshold"]
    assert threshold == MS_THRESHOLD == {"ms+ms": 59.0, "wd+dwarf": None}
    assert statistic["ms+ms"] > threshold["ms+ms"]
    assert statistic["ms+ms"] == pytest.approx(result["objectives"]["dwarf"] - result["objectives"]["ms+ms"])
    single = fit_star(_mock_binary(.8, 0), feh=None, fit_parallax=True, extinction=None,
                      extinction_prior=(0., .03))
    assert single["detection_statistic"]["ms+ms"] < threshold["ms+ms"]
    assert fit_star(SED.load(FIXTURE), region="sdb")["detection_statistic"] is None


class ClearQuery:
    """Integrated map stand-in with no extinction along any line of sight."""

    integrated = True
    n_samples = None

    def __call__(self, coords, mode="mean"):
        return 0.


def test_ms_threshold_under_the_dust_prior():
    prior = EdenhoferPrior(ClearQuery(), sigma=.002)
    result = fit_star(_mock_binary(.9, .7), feh=None, fit_parallax=True, extinction=None, dust_prior=prior)
    assert result["threshold"] == MS_THRESHOLD_DUST == {"ms+ms": 102.0, "wd+dwarf": None}
    assert result["detection_statistic"]["ms+ms"] > MS_THRESHOLD_DUST["ms+ms"]
    single = fit_star(_mock_binary(.8, 0), feh=None, fit_parallax=True, extinction=None, dust_prior=prior)
    assert single["detection_statistic"]["ms+ms"] < MS_THRESHOLD["ms+ms"]


def test_unsupported_and_label_free_giant_return_no_fit():
    sed = SED.load(FIXTURE)
    assert fit_star(sed, region="unsupported")["result"] is None
    giant = fit_star(sed, region="giant")
    assert giant["result"] is None and "labels" in giant["location"]["reason"]
    with pytest.raises(ValueError):
        fit_star(sed, region="nowhere")


def _stub_route(monkeypatch):
    """Stand-in for the WD-route frame with fixed objectives for the asked hypotheses."""
    def route(sed, *, hypotheses, **kwargs):
        return dict(hypotheses={h: dict(objective=float(i)) for i, h in enumerate(hypotheses)})
    monkeypatch.setattr("spbin.hrd.fit_whitedwarf_companion", route)


@pytest.mark.parametrize("kwargs, options, expected", [
    (dict(feh=None, fit_parallax=True, extinction=None), {}, MS_THRESHOLD),
    (dict(feh=0.0), dict(companion_feh=None, fit_parallax=True, extinction=None), MS_THRESHOLD),
    (dict(feh=None, fit_parallax=True, extinction=None), dict(dust_prior=object()), MS_THRESHOLD_DUST),
    (dict(feh=None, fit_parallax=True, extinction=None), dict(companion_feh=0.0), None),
    (dict(feh=None, fit_parallax=True, extinction=None), dict(fit_parallax=False), None),
    (dict(feh=None, fit_parallax=True, extinction=None), dict(extinction=0.1), None),
    (dict(feh=None, fit_parallax=True, extinction=None), dict(use_spherex=True), None),
    (dict(feh=None, fit_parallax=True, extinction=None, age_gyr=5.0), {}, None),
])
def test_ms_threshold_follows_the_configuration_that_runs(monkeypatch, kwargs, options, expected):
    _stub_route(monkeypatch)
    result = fit_star(_mock_binary(.9, .7), region="ms", options=options, **kwargs)
    assert result["threshold"] == (None if expected is None else dict(expected))


@pytest.mark.parametrize("kwargs, options, expected", [
    (dict(fit_parallax=True, extinction=None), {}, PMS_THRESHOLD),
    (dict(fit_parallax=True, extinction=None), dict(companion_feh=-0.3), None),
    (dict(fit_parallax=True, extinction=None), dict(companion_age_gyr=0.05), None),
    (dict(fit_parallax=True, extinction=None), dict(blue=(np.ones(5), np.ones(5))), None),
])
def test_pms_threshold_follows_the_configuration_that_runs(monkeypatch, kwargs, options, expected):
    _stub_route(monkeypatch)
    result = fit_star(_mock_binary(.9, .7), region="pms", options=options, **kwargs)
    assert result["threshold"] == (None if expected is None else dict(expected))
