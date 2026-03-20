import os
from glob import glob
import numpy as np
import rasterio
from rasterio.transform import Affine
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view

# -------------------------
# SETTINGS
# -------------------------
MOSAIC_GLOB = "./AST08/epochs/mosaic_daily/AST08_*_mosaic.tif"
OUTDIR = "./AST08/results/"
os.makedirs(OUTDIR, exist_ok=True)

# Scaling: your mosaics store temperature as 0.1 K → multiply by 0.1
SCALE = 0.1

# -------------------------
# LOAD MOSAICS
# -------------------------
print("Loading Mosaics...")
files = sorted(glob(MOSAIC_GLOB))
if not files:
    raise RuntimeError("No mosaics found. Check MOSAIC_GLOB path.")

# Check CRS / transform / dims consistency
def get_meta(path):
    with rasterio.open(path) as ds:
        return ds.crs, ds.transform, ds.width, ds.height, ds.nodata, ds.profile

crs0, transform0, width0, height0, nodata0, profile0 = get_meta(files[0])
for f in files[1:]:
    crs, transform, width, height, nodata, profile = get_meta(f)
    if crs!=crs0 or width!=width0 or height!=height0:
        raise RuntimeError(f"Inconsistent grid in {f}")

# Load stack → shape (T, H, W)
stack_native = []
for f in files:
    with rasterio.open(f) as ds:
        img = ds.read(1).astype(np.float32)
        stack_native.append(img)
stack_native = np.stack(stack_native, axis=0)  # (T, H, W)

# Apply 0.1 K scaling
stack = stack_native * SCALE   # now in Kelvin

# Time coordinates (optional)
dates = []
for f in files:
    base = os.path.basename(f)
    dt = base.split("_")[1]    # e.g. 2001-05-04
    dates.append(datetime.strptime(dt, "%Y-%m-%d"))
dates = np.array(dates)

T, H, W = stack.shape

# -------------------------
# 1) GLOBAL ANOMALY
# -------------------------
# For each time step: subtract median of the whole image
print("Calculating Global Anomaly...")
global_anoms = np.zeros((T, H, W), dtype=np.float32)

for i in range(T):
    img = stack[i]
    med = np.nanmedian(img)
    global_anoms[i] = img - med

global_anom_mean = np.nanmean(global_anoms, axis=0)  # (H, W)

# -------------------------
# 2) LOCAL ANOMALY (radius=1 and radius=2)
# -------------------------
print("Calculating Local Anomaly...")

def local_nanmedian(img, radius):
    """
    Compute nanmedian in a square neighbourhood using sliding windows.
    radius=1 → 3x3   radius=2 → 5x5
    Returns an array of same shape as img.
    Windows containing all-NaN return NaN.
    """
    size = 2*radius + 1
    pad = radius

    # Pad with NaN around edges so output matches input size
    padded = np.pad(img, pad_width=pad, mode='constant', constant_values=np.nan)

    # Extract sliding windows (H, W, size, size)
    windows = sliding_window_view(padded, (size, size))

    # Compute nanmedian along last two dims
    med = np.nanmedian(windows, axis=(-2, -1))

    return med


local_anom_r1_list = []
local_anom_r2_list = []

for i in range(T):
    img = stack[i]

    neigh1 = local_nanmedian(img, radius=1)  # 3x3
    neigh2 = local_nanmedian(img, radius=2)  # 5x5

    local_anom_r1_list.append(img - neigh1)
    local_anom_r2_list.append(img - neigh2)

local_anom_r1 = np.nanmean(np.stack(local_anom_r1_list, axis=0), axis=0)
local_anom_r2 = np.nanmean(np.stack(local_anom_r2_list, axis=0), axis=0)

# -------------------------
# 3) MEAN TEMPERATURE
# -------------------------
print("Calculating Mean Temperatures...")
mean_temp = np.nanmean(stack, axis=0)

# -------------------------
# 4) RMS deviation from pixel median
# -------------------------
print("Calculating RMS of pixel timeseires...")
pixel_medians = np.nanmedian(stack, axis=0)  # shape (H, W)
diff_sq = (stack - pixel_medians) ** 2
rms = np.sqrt(np.nanmean(diff_sq, axis=0))

# -------------------------
# 5) TREND (slope)
# -------------------------
print("Calculating the trend of each pixel timeseries...")
# Convert dates to numeric (days since first date)
t_numeric = np.array([(d - dates[0]).days for d in dates], dtype=np.float32)

# For each pixel: slope = cov(t, T) / var(t)
trend = np.full((H, W), np.nan, dtype=np.float32)

t_mean = np.nanmean(t_numeric)
t_var = np.nanvar(t_numeric)

if t_var == 0:
    raise RuntimeError("All dates identical – cannot compute trend.")

for y in range(H):
    Ts = stack[:, y, :]         # (T, W)
    Ts_mean = np.nanmean(Ts, axis=0)  # (W,)
    cov = np.nanmean((t_numeric[:,None] - t_mean) * (Ts - Ts_mean), axis=0)
    trend[y, :] = cov / t_var   # K per day

# -------------------------
# SAVE OUTPUTS
# -------------------------
def save_tif(arr, path):
    prof = profile0.copy()
    prof.update({
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "DEFLATE",
        "predictor": 3
    })
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)

save_tif(global_anom_mean,       os.path.join(OUTDIR, "global_anomaly.tif"))
save_tif(local_anom_r1,          os.path.join(OUTDIR, "local_anomaly_radius1.tif"))
save_tif(local_anom_r2,          os.path.join(OUTDIR, "local_anomaly_radius2.tif"))
save_tif(mean_temp,              os.path.join(OUTDIR, "mean_temperature.tif"))
save_tif(rms,                    os.path.join(OUTDIR, "rms_timeseries.tif"))
save_tif(trend,                  os.path.join(OUTDIR, "trend.tif"))

print("All summaries written to", OUTDIR)
