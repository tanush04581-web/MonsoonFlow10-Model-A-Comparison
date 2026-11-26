"""Build the daily 6-channel India dataset.

  python -m monsoon_r1.build_dataset train  --start 2005-01-01 --end 2021-12-31 --out data/r1/train.npz   (WeatherBench2 1 deg)
  python -m monsoon_r1.build_dataset arco   --dates 2023-03-01,2023-03-02 --out data/r1/x.npz              (ARCO-ERA5 0.25 deg)

Both paths end in the same regridding: conservative 1 deg cell means centred on INTEGER degrees (what WeatherBench2
ships), then the mean of the 4 surrounding integer points gives the half-degree cell centres of the contract grid.
For ARCO the 1 deg step is a box mean over the 0.25 deg points with weights [.5,1,1,1,.5]/4, so train and
hidden-test fields come out consistent. WeatherBench2's ERA5 stops on 2023-01-10; later dates need ARCO.
"""
from __future__ import annotations
import argparse, datetime as dt, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .contract import CHANNELS, LAT, LON, NC

WB2 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2_VARS = ["2m_temperature", "total_precipitation_6hr", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "total_column_water_vapour"]
ARCO_VARS = ["2m_temperature", "total_precipitation", "10m_u_component_of_wind", "10m_v_component_of_wind",
             "mean_sea_level_pressure", "total_column_water_vapour"]
# integer-degree window needed for the half-degree centres 6.5..37.5 / 65.5..100.5
INT_LAT = np.arange(6, 39); INT_LON = np.arange(65, 102)          # 33 x 37 points


def half_degree(x: np.ndarray) -> np.ndarray:
    """[..., 33(lat 6..38), 37(lon 65..101)] -> [..., 32, 36] mean of the 4 surrounding integer points."""
    return 0.25 * (x[..., :-1, :-1] + x[..., 1:, :-1] + x[..., :-1, 1:] + x[..., 1:, 1:])


def _open(url):
    import zarr
    from zarr.storage import FsspecStore
    return zarr.open_group(store=FsspecStore.from_url(url, storage_options={"token": "anon"}, read_only=True), mode="r")


def wb2_daily(start: dt.date, end: dt.date, workers: int = 12, block_days: int = 20) -> tuple[np.ndarray, list[dt.date]]:
    """[D,6,32,36] daily fields for start..end inclusive from WeatherBench2 (6-hourly, dims time x lon x lat)."""
    g = _open(WB2)
    t0 = dt.datetime(1959, 1, 1)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    lat_i = INT_LAT + 90; lon_i = INT_LON % 360                         # lat ascending -90..90, lon 0..359
    blocks = [days[i:i + block_days] for i in range(0, len(days), block_days)]

    def read(args):
        var, blk = args
        i0 = int((dt.datetime.combine(blk[0], dt.time()) - t0).total_seconds() // 21600)
        n = len(blk) * 4 + 1                                              # +1: the 24:00 step closes the last day's rain
        a = np.asarray(g[var][i0:i0 + n, lon_i[0]:lon_i[-1] + 1, lat_i[0]:lat_i[-1] + 1], dtype="f4")  # [n, lon, lat]
        return a.transpose(0, 2, 1)                                       # [n, lat, lon]

    out = np.zeros((len(days), NC, 32, 36), "f4")
    jobs = [(v, b) for b in blocks for v in WB2_VARS]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(read, jobs))
    k = 0; d0 = 0
    for blk in blocks:
        for c, var in enumerate(WB2_VARS):
            a = res[k]; k += 1
            for j in range(len(blk)):
                s = a[j * 4:(j + 1) * 4]                                  # 00,06,12,18 UTC
                if c == 1:                                                # tp: accumulations ending 06,12,18 and next 00
                    day = a[j * 4 + 1:j * 4 + 5].sum(axis=0) * 1000.0     # m -> mm
                else:
                    day = s.mean(axis=0)
                out[d0 + j, c] = half_degree(day)
        d0 += len(blk)
    return out, days


def _box1deg(x: np.ndarray) -> np.ndarray:
    """0.25 deg rows/cols covering 5.5..38.5 / 64.5..101.5 (integer +-0.5) -> 33 x 37 integer-centred 1 deg means."""
    w = np.array([0.5, 1, 1, 1, 0.5]) / 4.0
    nlat, nlon = 33, 37
    out = np.zeros(x.shape[:-2] + (nlat, nlon), "f4")
    for i in range(nlat):
        for j in range(nlon):
            blk = x[..., 4 * i:4 * i + 5, 4 * j:4 * j + 5]
            out[..., i, j] = np.einsum("...ab,a,b->...", blk, w, w)
    return out


def arco_daily(days: list[dt.date], workers: int = 32) -> np.ndarray:
    """[D,6,32,36] from ARCO-ERA5 0.25 deg hourly: means from 00/06/12/18 UTC, rain from all 24 hours."""
    g = _open(ARCO); E0 = dt.datetime(1900, 1, 1)
    lat0 = 90.0 - 38.5; lon0 = 64.5                                       # first needed 0.25 deg row/col (lat desc)
    r0 = int((90.0 - 38.5) / 0.25); c0 = int(64.5 / 0.25)
    r1 = r0 + 4 * 32 + 5 - 0; c1 = c0 + 4 * 36 + 5                        # rows for lat 38.5..5.5, cols 64.5..101.5
    r1 = r0 + 133; c1 = c0 + 149

    def hidx(t): return int((t - E0).total_seconds() // 3600)

    def read(a):
        var, h = a
        x = np.asarray(g[var][h, r0:r1, c0:c1], dtype="f4")               # lat descending
        return x[::-1]                                                     # -> ascending (5.5 .. 38.5)

    jobs = []
    for d in days:
        base = hidx(dt.datetime.combine(d, dt.time()))
        for c, var in enumerate(ARCO_VARS):
            hours = range(1, 25) if c == 1 else (0, 6, 12, 18)
            for h in hours:
                jobs.append((var, base + h))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(read, jobs))
    out = np.zeros((len(days), NC, 32, 36), "f4"); k = 0
    for di, d in enumerate(days):
        for c, var in enumerate(ARCO_VARS):
            n = 24 if c == 1 else 4
            stack = np.stack(res[k:k + n]); k += n
            day = stack.sum(axis=0) * 1000.0 if c == 1 else stack.mean(axis=0)
            out[di, c] = half_degree(_box1deg(day))
    return out


def static_fields() -> np.ndarray:
    """[2,32,36]: land_sea_mask (0..1) and orography (surface geopotential / g, metres) on the contract grid."""
    g = _open(WB2)
    lat_i = INT_LAT + 90; lon_i = INT_LON % 360
    def get(name):
        a = np.asarray(g[name][lon_i[0]:lon_i[-1] + 1, lat_i[0]:lat_i[-1] + 1], dtype="f4") if g[name].ndim == 2 else None
        return half_degree(a.T)
    lsm = get("land_sea_mask"); oro = get("geopotential_at_surface") / 9.80665
    return np.stack([np.clip(lsm, 0, 1), oro]).astype("f4")


def export_nc(npz: Path, out: Path, static: np.ndarray | None = None) -> None:
    """The file teams receive (india_climate_train.nc). Written single-threaded: the HDF5 library is not thread safe."""
    import xarray as xr
    z = np.load(npz); dates = [dt.datetime.strptime(str(int(x)), "%Y%m%d") for x in z["dates"]]
    ds = xr.Dataset({"data": (("time", "channel", "lat", "lon"), z["data"])},
                    coords={"time": dates, "channel": list(CHANNELS), "lat": LAT, "lon": LON})
    ds["data"].attrs["units"] = "t2m K | tp mm/day | u10 m/s | v10 m/s | msl Pa | tcwv kg/m2"
    if static is not None:
        ds["static"] = (("static_channel", "lat", "lon"), static); ds = ds.assign_coords(static_channel=["land_sea_mask", "orography_m"])
    ds.to_netcdf(out, encoding={"data": {"zlib": True, "complevel": 4}})


def _save(path: Path, data: np.ndarray, days: list[dt.date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=data, dates=np.array([int(f"{d:%Y%m%d}") for d in days]), lat=LAT, lon=LON)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--start", required=True); t.add_argument("--end", required=True); t.add_argument("--out", required=True)
    t.add_argument("--chunk", type=int, default=120); t.add_argument("--workers", type=int, default=6)
    a_ = sp.add_parser("arco"); a_.add_argument("--dates", required=True); a_.add_argument("--out", required=True)
    e = sp.add_parser("export-nc"); e.add_argument("--npz", required=True); e.add_argument("--out", required=True)
    m = sp.add_parser("merge"); m.add_argument("--parts", nargs="+", required=True); m.add_argument("--out", required=True)
    m.add_argument("--artifacts", help="also write climatology.npy and stats.json into this directory")
    a = ap.parse_args(argv); t0 = time.time()
    if a.cmd == "export-nc":
        export_nc(Path(a.npz), Path(a.out), static_fields()); print("wrote", a.out); return 0
    if a.cmd == "merge":
        zs = [np.load(p) for p in a.parts]
        data = np.concatenate([z["data"] for z in zs]); dates = np.concatenate([z["dates"] for z in zs])
        order = np.argsort(dates); data, dates = data[order], dates[order]
        if len(set(dates.tolist())) != len(dates):
            raise SystemExit("duplicate dates across parts")
        d = [dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100) for x in dates]
        gaps = [(d[i], d[i + 1]) for i in range(len(d) - 1) if (d[i + 1] - d[i]).days != 1]
        if gaps:
            raise SystemExit(f"gaps between parts: {gaps[:3]}")
        if not np.isfinite(data).all():
            raise SystemExit("non-finite values in merged data")
        data[:, 1] = np.maximum(data[:, 1], 0.0)          # ERA5 rain has tiny negatives (-1e-7): physical rain is never negative
        _save(Path(a.out), data, d); print("merged", data.shape, d[0], d[-1])
        if a.artifacts:
            from . import stats
            st = stats.build_artifacts(Path(a.out), Path(a.artifacts)); print("stats std:", [round(x, 3) for x in st["std"]])
        return 0
    if a.cmd == "train":
        s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
        cdir = Path(a.out).with_suffix(".chunks"); cdir.mkdir(parents=True, exist_ok=True)
        parts, alld = [], []
        cur = s
        while cur <= e:                                                   # one file per chunk: a crash or stall never loses finished work
            nxt = min(e, cur + dt.timedelta(days=a.chunk - 1)); f = cdir / f"{cur:%Y%m%d}_{nxt:%Y%m%d}.npz"
            if f.exists():
                try:
                    z = np.load(f); x, d = z["data"], [dt.date(int(v) // 10000, int(v) // 100 % 100, int(v) % 100) for v in z["dates"]]; tag = "cached"
                except Exception:
                    f.unlink(); continue
            else:
                for attempt in range(4):
                    try:
                        x, d = wb2_daily(cur, nxt, workers=a.workers); break
                    except Exception as ex:
                        print(f"{cur}..{nxt} attempt {attempt + 1} failed: {type(ex).__name__}", flush=True); time.sleep(10 * (attempt + 1))
                else:
                    raise SystemExit(f"chunk {cur}..{nxt} failed 4 times")
                _save(f, x, d); tag = "ok"
            parts.append(x); alld += d
            print(f"{cur}..{nxt} {tag} ({time.time() - t0:.0f}s)", flush=True); cur = nxt + dt.timedelta(days=1)
        _save(Path(a.out), np.concatenate(parts), alld)
    else:
        days = [dt.date.fromisoformat(x) for x in a.dates.split(",")]
        _save(Path(a.out), arco_daily(days), days)
    print("done", a.out, f"{time.time() - t0:.0f}s"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
