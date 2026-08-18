## Merged Interplanetary Data from L1

Python client for the [MIDL solar wind dataset](https://csem.engin.umich.edu/MIDL/).

## Install

```
pip install csem-midl
```

## Quickstart

```python
import midl

data = midl.load("2005-01-01 00:00", "2005-01-01 01:00", 32)
print(data)
```

```
<xarray.Dataset> Size: 4kB
Dimensions:  (time: 61)
Coordinates:
  * time     (time) datetime64[ns] 488B 2005-01-01 ... 2005-01-01T01:00:00
Data variables:
    Bx       (time) float64 488B -6.05 -3.94 -3.7 -3.75 ... -2.01 -1.78 -2.08
    By       (time) float64 488B 3.84 3.5 3.12 2.94 3.2 ... 4.97 5.3 5.29 5.2
    Bz       (time) float64 488B -0.93 -4.06 -4.83 -4.92 ... 1.25 2.04 2.05 2.11
    Ux       (time) float64 488B -422.3 -429.9 -433.6 ... -437.4 -441.8 -441.8
    Uy       (time) float64 488B -13.38 -12.91 -9.05 ... -28.15 -27.06 -26.68
    Uz       (time) float64 488B 26.93 47.59 54.86 54.99 ... 0.46 1.42 0.92
    rho      (time) float64 488B 5.169 5.439 5.481 5.359 ... 6.131 6.363 6.323
    T        (time) float64 488B 1.33e+05 1.321e+05 ... 1.159e+05 1.159e+05
Attributes:
    source:             MIDL
    url:                https://csem.engin.umich.edu/MIDL/
    target:             32Re
    midl_propagation:   {'method': 'ballistic', 'target_re': 32.0}
```
```python
# 14Re and 32Re Ballistically-propagated data are available for direct download
midl.load("2024-05-10 00:00", "2024-05-11 01:00", 14, method="ballistic")
midl.load("2024-05-10 00:00", "2024-05-11 01:00", 32, method="ballistic") 

# [-70,70]Re MHD-propagated data are available for direct download
midl.load("2024-05-10 00:00", "2024-05-11 01:00", 30, method="mhd")

# Any value can be ballistically propagated client-side
midl.load("2024-05-10 00:00", "2024-05-11 01:00", 20.25, method="ballistic") 

# Data at L1 can be downloaded directly as well 
midl.load("2024-05-10 00:00", "2024-05-11 01:00", "L1")
```
Client-side MHD propagation is not supported.

### Coordinate systems

MIDL data is native GSM. Pass `coords="GSE"` or `coords="SM"` (case-insensitive,
default `"GSM"`) to receive the vector variables (Bx, By, Bz, Ux, Uy, Uz)
rotated into that frame, minute by minute:

```python
midl.load("2024-05-10 00:00", "2024-05-11 01:00", 32, coords="GSE")
midl.load("2024-05-10 00:00", "2024-05-11 01:00", "L1", coords="SM")
```

Scalars (rho, T) and provenance columns are untouched, and the L1 `X`
variable always remains the reference satellite X_GSM position regardless
of `coords`. The rotation conventions and the published per-minute angle
tables they use are documented at
https://csem.engin.umich.edu/MIDL/format.html.

Saving to file:
```python
midl.to_csv(data, "storm.csv")
midl.to_dat(data, "storm.dat")
```

Data is cached locally after the first download.

## Using with spacepy

MIDL Datasets convert to a [spacepy](https://spacepy.github.io/) `SpaceData` in a few lines:

```python
import midl
from spacepy.datamodel import SpaceData, dmarray

ds = midl.load("2005-01-01 00:00", "2005-01-01 01:00", 32)

sd = SpaceData(attrs=dict(ds.attrs))
sd["Epoch"] = dmarray(ds["time"].values, attrs={"units": "UTC"})
for name, da in ds.data_vars.items():
    sd[name] = dmarray(da.values, attrs=dict(da.attrs))
```

## Tutorials

- [Merging MIDL with IMP-8](https://csem.engin.umich.edu/MIDL/imp-8-tutorial.html) — merge MIDL L1 with external satellite data to fill gaps in the 1998–2004 ACE-only era

## See also

- [MIDL-Pipeline](https://github.com/connordimarco/MIDL-Pipeline) — the code that produces the MIDL dataset
- [MSWIM2D](https://csem.engin.umich.edu/MSWIM2D/) — 2-D MHD solar wind model (1–75 AU) driven by MIDL L1 data; Python client [`mswim2d`](https://github.com/connordimarco/CSEM-MSWIM2D)
