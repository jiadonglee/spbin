from io import BytesIO
from types import SimpleNamespace

from astropy.table import Table
import numpy as np
import pytest
import requests

from spbin import StellarModel, fetch
from spbin.fetch import _photometry, _tap


def test_tap_normalizes_gaia_field_names_and_keeps_exact_id(monkeypatch):
    table = Table({"SOURCE_ID": [1521154374020165376], "ra": [190.1704458396]})
    output = BytesIO()
    table.write(output, format="votable")
    response = SimpleNamespace(content=output.getvalue(), raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    result = _tap("SELECT source_id, ra FROM gaiadr3.gaia_source")
    assert result.colnames == ["source_id", "ra"]
    assert int(result[0]["source_id"]) == 1521154374020165376


def _two_mass(**flags):
    return Table(rows=[dict(n_mates=0, n_neighbours=1, ph_qual="AAA", ext_key=0, designation="13274764+7858589",
                            j_m=10., h_m=10., ks_m=10., j_msigcom=.02, h_msigcom=.02, ks_msigcom=.02,
                            **flags)])


@pytest.mark.parametrize("flags, valid", [
    (dict(cc_flg="000", gal_contam=0), [True, True, True]),
    (dict(cc_flg="0c0", gal_contam=0), [True, False, True]),
    (dict(cc_flg="ccc", gal_contam=2), [False, False, False]),
    (dict(cc_flg="000", gal_contam=1), [False, False, False]),
    ({}, [False, False, False]),
])
def test_2mass_mask_needs_clean_contamination_flags(flags, valid):
    rows = [("2MASS", _two_mass(**flags), range(3), ("j_m", "h_m", "ks_m"),
             ("j_msigcom", "h_msigcom", "ks_msigcom"))]
    flux, _, mask, metadata = _photometry(rows, StellarModel().wavelength_um[61:66])
    assert mask[:3].tolist() == valid
    assert np.isfinite(flux[:3]).all() and metadata["2MASS"][0]["j_m"] == 10.


def test_download_takes_2mass_flags_from_irsa(monkeypatch, tmp_path):
    gaia = Table(rows=[dict(source_id=1, ra=1., dec=1., parallax=10., parallax_error=.1,
                            has_xp_continuous=False)])
    tables = {"gaia.ecsv": gaia, "2mass.ecsv": _two_mass(), "allwise.ecsv": Table(names=["n_mates"]),
              "2mass_flags.ecsv": Table(rows=[dict(cc_flg="00c", gal_contam=0)])}
    urls = {}

    def query(query, path, url=fetch.GAIA_TAP):
        urls[path.name] = url
        return tables[path.name].copy()

    monkeypatch.setattr(fetch, "_query_cached", query)
    with pytest.warns(UserWarning, match="no continuous XP"):
        sed = fetch.download(1, cache_dir=tmp_path)
    assert urls["2mass_flags.ecsv"] == fetch.IRSA_TAP and urls["2mass.ecsv"] == fetch.GAIA_TAP
    assert sed.mask[61:64].tolist() == [True, True, False]
    assert sed.metadata["2MASS"][0]["cc_flg"] == "00c"
