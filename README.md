<p align="center">
  <img src="docs/assets/spbin-logo.png" width="280" alt="spbin logo">
</p>
<p align="center">Spectrophotometric binaries: fit Gaia XP and infrared SEDs of single stars and unresolved binaries.</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-hrd-dark.png">
    <img src="docs/assets/hero-hrd.png" width="100%" alt="Binary classes across the Gaia HR diagram with their spbin model SEDs">
  </picture>
</p>

spbin fits single stars and unresolved binaries to public Gaia DR3 XP
spectra combined with Gaia-linked 2MASS and AllWISE photometry. Models
predict absolute fluxes at the Gaia parallax, with no free normalisation.
The package runs on NumPy and SciPy, with no GPU or separate model download.

[Documentation](https://jiadonglee.github.io/spbin/) · [中文文档](https://jiadonglee.github.io/spbin/zh/) · [Models, results and scope](docs/overview.md) · [API](docs/api.md)

## Install

```bash
pip install "spbin[download] @ git+https://github.com/jiadonglee/spbin.git"
```

Plain `pip install .` supports offline fitting and plotting. The extras are
`download`, `dust`, `spherex`, `jax` and `notebook`.

## Fit one source

```python
from spbin import download, fit, plot

sed = download("858860697467058688", cache_dir="data")
result = fit(sed, age_gyr=5.0, feh=0.0)
print(result["binary"]["m1"], result["binary"]["q"])
fig = plot(sed, result, path="sed.png")
```

`fit` compares a single star with a coeval binary. `result["delta"]` is the
single minus binary objective, a model-preference diagnostic rather than a
binary probability. Use `age_gyr=None, feh=None` to fit age and metallicity,
and `extinction=None, fit_parallax=True` to fit
[extinction and parallax](docs/extinction.md).

![Single and binary SED fits](docs/assets/sed-example.png)

## Fit any star

```python
from spbin import download, fit_star

sed = download("1521154374020165376", cache_dir="data")
result = fit_star(sed)
print(result["region"], result["preferred"], result["delta"])
```

`fit_star` places the star on the HR diagram and runs only the hypotheses of
its region:

| Region | Route | Hypotheses |
| --- | --- | --- |
| `ms` | `fit_whitedwarf_companion` | dwarf, coeval MS+MS with free q, DA+dwarf |
| `hot_ms` | `fit` with `StellarModel(hot=True)` | single, coeval binary |
| `giant` | `fit_giant_companion`, with `labels=dict(teff=, logg=, feh=)` | giant alone, giant+companion |
| `sdb` | `fit_subdwarf_companion` | FGK, subdwarf, subdwarf+dwarf, subdwarf+subgiant |
| `wd` | `fit_whitedwarf_companion` | dwarf, DA, DA+dwarf |

Each route can also be called directly. `spbin.orbit` solves Gaia photocentre
orbits for a faint companion or a luminous twin and compares their SEDs.
Fits stay inside the validated ranges; the models do not extrapolate. See
[models, results and scope](docs/overview.md) for the supported ranges, the
validation numbers and what is not covered.

## Documentation

- [Models, results and scope](docs/overview.md), [model reference](docs/model.md) and [API](docs/api.md)
- [Data and provenance](docs/data.md), [Gaia batch downloads](docs/gaia.md), [SPHEREx](docs/spherex.md)
- [Extinction](docs/extinction.md), [photocentre orbits](docs/orbit.md), [orblet interface](docs/orblet.md)
- Validation: [overview](docs/validation.md), [warm](docs/validation-warm.md), [hot](docs/validation-hot.md), [giants](docs/validation-giant.md), [subdwarfs](docs/validation-subdwarf.md), [white dwarfs](docs/validation-whitedwarf.md)
- [Example notebooks](examples/)

## Development

```bash
pip install -e ".[test]"
ruff check .
pytest -q
```

## Citation

If you use spbin in your research, please cite Li et al. (2026),
[*Unresolved binaries from Gaia XP to SPHEREx*](https://www.alphaxiv.org/abs/2611.29702v1),
arXiv:2611.29702.

```bibtex
@misc{Li2026spbin,
  author        = {Li, Jiadong and Rix, Hans-Walter and M{\"u}ller-Horn, Johanna
                   and El-Badry, Kareem and Pennell, Ben and Pang, Xiaoying and Liu, Chao},
  title         = {Unresolved binaries from {Gaia XP} to {SPHEREx}},
  year          = {2026},
  eprint        = {2611.29702},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.SR},
  doi           = {10.48550/arXiv.2611.29702},
  url           = {https://arxiv.org/abs/2611.29702}
}
```

## Credits

The bundled stellar network and hot-star table are extracted from
[J-CAPS](https://github.com/jiadonglee/J-Caps), using PARSEC stellar tracks,
CK04 (Castelli & Kurucz 2003) and TLUSTY BSTAR2006 (Lanz & Hubeny 2007)
model atmospheres. The giant template is built from APOGEE DR17. Hot-subdwarf spectra are
Tuebingen TMAP NLTE models (Werner et al. 2003; Rauch & Deetjen 2003) from
the TheoSSA service of the German Astrophysical Virtual Observatory.
Public XP spectra are calibrated with
[GaiaXPy](https://gaia-dpci.github.io/GaiaXPy-website/).
SPHEREx aperture extraction uses [XphereX](https://github.com/jiadonglee/XphereX),
[TallTable](https://github.com/cmhainje/talltable) and
[SPExPI](https://github.com/fkiwy/spexpi).
Catalogue data are retrieved through
[astroquery](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html).
Please acknowledge Gaia, 2MASS, WISE, APOGEE, PARSEC, CK04, TLUSTY, TMAP
and TheoSSA, GALEX and J-CAPS when using these data and models in research; see
[provenance](docs/data.md).
