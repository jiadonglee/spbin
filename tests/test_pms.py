import numpy as np
import pytest

from spbin import SED, StellarModel, WhiteDwarfModel
from spbin.hrd import PMS_THRESHOLD, fit_star, locate
from spbin.model import HOT_CHANNELS
from spbin.whitedwarf import fit_whitedwarf_companion


@pytest.fixture(scope="module")
def pms():
    return StellarModel(pms=True)


def _young_network(pms):
    """The plain network on the pms route's ages, without the correction."""
    model = StellarModel()
    model.age_range_gyr = pms.age_range_gyr
    return model


def _mock(m1, q, age=.02):
    stellar = StellarModel(pms=True)
    pred = stellar.evaluate(m1, q, age, 0)
    flux = pred["flux_10pc"] * .01
    g = -2.5 * np.log10(np.sum(10**(-0.4 * pred["M_G"]))) + 5
    metadata = dict(phot_g_mean_mag=g, phot_bp_mean_mag=g + 1.0, phot_rp_mean_mag=g - 1.0)
    return SED(flux, flux * .01, WhiteDwarfModel().support & stellar.support, 10, .02, "mock", metadata)


def test_pms_route_ages_channels_and_correction(pms):
    assert pms.age_range_gyr == pytest.approx((10**-2.4, .5))
    assert pms.support[:HOT_CHANNELS].all() and not pms.support[HOT_CHANNELS:].any()
    with pytest.raises(ValueError):
        StellarModel(hot=True, pms=True)
    with pytest.raises(ValueError):
        pms.evaluate(.5, 0., 1., 0.)
    correction = pms.pms_correction([2900., 3000., 4200., 6500., 7000.], .03)
    assert correction.shape == (5, HOT_CHANNELS)
    np.testing.assert_allclose(correction[0], correction[1])
    assert not correction[3:].any()
    assert correction[2, :9].mean() > .05  # blue XP excess of a 30 Myr K dwarf
    assert not pms.pms_correction([4200.], .5).any()
    young = pms.evaluate(.6, .7, .03, 0.)
    plain = _young_network(pms).evaluate(.6, .7, .03, 0.)
    np.testing.assert_allclose(young["components"][:, :HOT_CHANNELS],
                               plain["components"][:, :HOT_CHANNELS]
                               * np.exp(pms.pms_correction(young["teff"], .03)))
    assert young["route"].endswith(pms.pms_summary["version"])


def test_pms_error_term_follows_the_correction(pms):
    young = pms.evaluate(.6, 0., .03, 0.)
    columns, variance = pms.error_factors(young)
    base_columns, base_variance = _young_network(pms).error_factors(young)
    np.testing.assert_allclose(columns[:, :-1], base_columns)
    knots = pms.pms_table["teff_knots"]
    scatter = np.interp(young["teff"][0], knots, pms.pms_table["amplitude_scatter"])
    expected = scatter * young["flux_10pc"][:HOT_CHANNELS] * pms.pms_correction(young["teff"], .03)[0]
    np.testing.assert_allclose(columns[:HOT_CHANNELS, -1], expected)
    assert np.all(variance >= base_variance)
    old = pms.evaluate(.6, 0., .5, 0.)
    columns, variance = pms.error_factors(old)
    assert not columns[:, -1].any()
    np.testing.assert_allclose(variance, _young_network(pms).error_factors(old)[1])


def test_pms_hypotheses_in_the_whitedwarf_frame():
    sed = _mock(.5, 0.)
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, hypotheses=("pms", "dwarf"))
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, hypotheses=("pms",), stellar=StellarModel())
    with pytest.raises(ValueError):
        fit_whitedwarf_companion(sed, hypotheses=("dwarf",), stellar=StellarModel(pms=True))
    result = fit_whitedwarf_companion(sed, hypotheses=("pms",), companion_age_gyr=None,
                                      companion_prior=dict(log_age=(7.5, .3)))
    labels = result["hypotheses"]["pms"]["companion"]
    assert labels["log_age"] == pytest.approx(np.log10(labels["age_gyr"]) + 9)
    assert abs(labels["log_age"] - np.log10(2e7)) < .1
    assert abs(labels["mass"] - .5) < .02
    assert result["hypotheses"]["pms"]["penalty"] == pytest.approx(((labels["log_age"] - 7.5) / .3)**2)


def test_pms_region_routes_and_calibrated_threshold():
    single = _mock(.5, 0.)
    location = locate(single)
    assert location["region"] == "pms" and location["hypotheses"] == ("pms", "pms+pms")
    result = fit_star(single, feh=None, fit_parallax=True, extinction=None, extinction_prior=(0., .03))
    assert set(result["objectives"]) == {"pms", "pms+pms"}
    assert result["preferred"] == "pms"
    assert result["result"]["hypotheses"]["pms"]["companion"]["feh"] == 0.
    statistic = result["detection_statistic"]["pms+pms"]
    assert statistic == pytest.approx(result["objectives"]["pms"] - result["objectives"]["pms+pms"])
    assert result["threshold"] == PMS_THRESHOLD and statistic < PMS_THRESHOLD["pms+pms"]
    assert fit_star(single, age_gyr=.02)["threshold"] is None
    pair = fit_star(_mock(.5, .8), age_prior=(np.log10(2e7), .3))
    assert pair["preferred"] == "pms+pms"
    assert abs(pair["result"]["hypotheses"]["pms+pms"]["companion"]["q"] - .8) < .05


@pytest.mark.parametrize("age", [None, .05])
def test_base_fit_refuses_the_pms_model(pms, age):
    from spbin.fit import fit
    sed = _mock(.6, 0)
    with pytest.raises(ValueError, match="fit_star"):
        fit(sed, "single", model=pms, age_gyr=age)
    pytest.importorskip("jax")
    from spbin._jax.fit import fit as jax_fit
    with pytest.raises(ValueError, match="fit_star"):
        jax_fit(sed, "single", model=pms, age_gyr=age)


def test_jax_whitedwarf_frame_refuses_the_pms_model(pms):
    pytest.importorskip("jax")
    from spbin._jax.whitedwarf import fit_whitedwarf_companion as jax_fit
    with pytest.raises(ValueError, match="pms"):
        jax_fit(_mock(.6, 0), hypotheses=("dwarf",), stellar=pms, companion_age_gyr=.05)
