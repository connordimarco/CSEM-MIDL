# midl

Python client for the [MIDL solar wind dataset](https://csem.engin.umich.edu/MIDL/).

## Install

```
pip install csem-midl
```

## Quickstart

```python
import midl

ds = midl.load("2015-03-17", "2015-03-18", "32re")
print(ds)

# Save to file
midl.to_csv(ds, "storm.csv")
midl.to_dat(ds, "storm.dat")
```

`midl.load()` returns an `xarray.Dataset` with time coordinate and variables:
Bx, By, Bz (nT), Ux, Uy, Uz (km/s), rho (cm^-3), T (K).

Data is cached locally after the first download.
