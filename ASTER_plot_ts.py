#!/Usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTER Interactive Viewer — Advanced Version
SECTION 1 of 6 — Imports, User Config, File Discovery, Metadata
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as xy_transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Rectangle
from glob import glob
import re
from datetime import datetime
import matplotlib.dates as mdates

# ----------------------------------------------------------
# USER CONFIGURATION
# ----------------------------------------------------------

# Path to mosaicked temperature rasters
MOSAIC_GLOB = "./AST08/epochs/mosaic_daily/AST08_*_mosaic.tif"

# Path to derived summary TIFFs
RESULT_DIR = "./AST08/results/"

SUMMARY_FILES = {
    "global_anomaly":       os.path.join(RESULT_DIR, "global_anomaly.tif"),
    "local_anomaly_r1":     os.path.join(RESULT_DIR, "local_anomaly_radius1.tif"),
    "local_anomaly_r2":     os.path.join(RESULT_DIR, "local_anomaly_radius2.tif"),
    "mean_temperature":     os.path.join(RESULT_DIR, "mean_temperature.tif"),
    "rms_timeseries":       os.path.join(RESULT_DIR, "rms_timeseries.tif"),
    "trend":                os.path.join(RESULT_DIR, "trend.tif"),
}

# Temperature stored as 0.1K → multiply by 0.1 to obtain Kelvin
SCALE_TO_K = 0.1

# Whether to preload all rasters into memory (faster interaction)
PRELOAD = True

# Percentile stretch for visualization
PERCENT_STRETCH = (2, 98)

# ----------------------------------------------------------
# COLOURMAPS
# ----------------------------------------------------------
from cmcrameri import cm

CM_TEMP     = cm.lajolla                 # Absolute temperature layers
CM_MEAN     = cm.lajolla                 # Mean temperature summary
CM_DIFF     = cm.managua_r               # Difference maps + anomalies
CM_ANOM     = cm.managua_r               # Global/local anomaly summaries
CM_RMS      = cm.bilbao                  # RMS
CM_TREND    = cm.roma_r                  # Trend (K/year)

# ----------------------------------------------------------
# DISCOVER MOSAIC FILES
# ----------------------------------------------------------
FILES = sorted(glob(MOSAIC_GLOB))
if not FILES:
    raise RuntimeError(f"No mosaics found for pattern: {MOSAIC_GLOB}")

# Parse dates from filenames
DATE_RE = re.compile(r"AST08_(\d{4}-\d{2}-\d{2})_mosaic\.tif$")
DATES = []
for f in FILES:
    m = DATE_RE.search(os.path.basename(f))
    if m is None:
        raise ValueError(f"Could not parse date from filename: {f}")
    DATES.append(datetime.strptime(m.group(1), "%Y-%m-%d"))
DATES = np.array(DATES)

# ----------------------------------------------------------
# EXTRACT GRID METADATA (consistent across all mosaics)
# ----------------------------------------------------------
with rasterio.open(FILES[0]) as ds0:
    crs0       = ds0.crs
    transform0 = ds0.transform
    H          = ds0.height
    W          = ds0.width
    nodata0    = ds0.nodata
    profile0   = ds0.profile

# Sanity check across all files
for f in FILES[1:]:
    with rasterio.open(f) as ds:
        if (ds.height != H) or (ds.width != W) or (ds.crs != crs0):
            raise RuntimeError(f"Inconsistent grid found in file: {f}")

# ----------------------------------------------------------
# PRELOAD MOSAICS (optional but fast)
# ----------------------------------------------------------
if PRELOAD:
    print("Preloading mosaics into memory...")
    STACK = np.zeros((len(FILES), H, W), dtype=np.float32)
    for i, f in enumerate(FILES):
        with rasterio.open(f) as ds:
            STACK[i] = ds.read(1).astype(np.float32) * SCALE_TO_K
else:
    STACK = None

def get_epoch(i):
    """Return mosaic image for epoch i, in Kelvin."""
    if PRELOAD:
        return STACK[i]
    with rasterio.open(FILES[i]) as ds:
        return ds.read(1).astype(np.float32) * SCALE_TO_K

# ----------------------------------------------------------
# LOAD SUMMARY LAYERS
# ----------------------------------------------------------
SUMMARY = {}
for key, path in SUMMARY_FILES.items():
    if not os.path.exists(path):
        print(f"WARNING: Summary TIFF missing: {path}")
        continue
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
    if key == "trend":
        # Convert from K/day → K/year
        arr = arr * 365.25
    SUMMARY[key] = arr

"""
ASTER Interactive Viewer — Advanced Version
SECTION 2 of 6 — Helper Functions (NaN‑safe TS, lon/lat, stretch, reference pixel)
"""

# ----------------------------------------------------------
# ROBUST PERCENTILE STRETCH FOR VISUALIZATION
# ----------------------------------------------------------

def robust_limits(arr, pct=PERCENT_STRETCH):
    """Return (vmin, vmax) from percentile stretch, ignoring NaNs."""
    f = arr[np.isfinite(arr)]
    if f.size == 0:
        return 0, 1
    lo, hi = np.nanpercentile(f, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(f), np.nanmax(f)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
    return float(lo), float(hi)

# ----------------------------------------------------------
# PIXEL → LON/LAT
# ----------------------------------------------------------

def rc_to_lonlat(row, col):
    """
    Convert row/col indices to lon/lat (WGS84).
    Works even if original CRS is UTM.
    """
    x, y = rasterio.transform.xy(transform0, row, col, offset="center")
    if crs0 and crs0.to_string() != "EPSG:4326":
        lon, lat = xy_transform(crs0, "EPSG:4326", [x], [y])
        return float(lon[0]), float(lat[0])
    return float(x), float(y)

# ----------------------------------------------------------
# NAN‑SAFE TIME SERIES FOR A SINGLE PIXEL
# ----------------------------------------------------------

def get_pixel_ts(row, col):
    """
    Return a TS for (row, col) shape=(T,), float32.
    Preserves NaNs exactly as in the mosaics.
    """
    vals = []
    for i, f in enumerate(FILES):
        if PRELOAD:
            vals.append(STACK[i, row, col])
        else:
            with rasterio.open(f) as ds:
                if row < 0 or row >= ds.height or col < 0 or col >= ds.width:
                    vals.append(np.nan)
                    continue
                window = Window(col, row, 1, 1)
                v = ds.read(1, window=window)[0, 0]
                vals.append(v * SCALE_TO_K)
    return np.array(vals, dtype=np.float32)

# ----------------------------------------------------------
# NAN‑SAFE TIME SERIES FOR A RECTANGULAR REGION
# ----------------------------------------------------------

def get_region_ts(r0, c0, r1, c1):
    """
    Mean TS over region [r0:r1, c0:c1], inclusive.
    Fully NaN‑safe.
    """
    r0, r1 = sorted([r0, r1])
    c0, c1 = sorted([c0, c1])
    vals = []
    for i, f in enumerate(FILES):
        if PRELOAD:
            region = STACK[i, r0:r1+1, c0:c1+1]
        else:
            with rasterio.open(f) as ds:
                window = Window(c0, r0, c1-c0+1, r1-r0+1)
                region = ds.read(1, window=window).astype(np.float32) * SCALE_TO_K
        vals.append(np.nanmean(region))
    return np.array(vals, dtype=np.float32)

# ----------------------------------------------------------
# AUTOMATIC REFERENCE PIXEL (GLOBAL ANOMALY ~ 0)
# ----------------------------------------------------------

if "global_anomaly" in SUMMARY:
    glob = SUMMARY["global_anomaly"]
    idx = np.nanargmin(np.abs(glob))
    ref_auto_row, ref_auto_col = np.unravel_index(idx, glob.shape)
else:
    # fallback centre
    ref_auto_row, ref_auto_col = H // 2, W // 2

"""
ASTER Interactive Viewer — Advanced Version
SECTION 3 of 6 — Figure Setup, UI Elements, Rectangles, Initial Rendering
"""

# ----------------------------------------------------------
# CREATE MAIN MAP FIGURE
# ----------------------------------------------------------

fig_map, ax_map = plt.subplots(figsize=(8.0, 6.5))
plt.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.95)

# ----------------------------------------------------------
# LAYER SELECTOR (LEFT SIDE)
# ----------------------------------------------------------

layer_labels = ["temperature", "difference"] + list(SUMMARY.keys())

ax_mode = plt.axes([0.03, 0.25, 0.13, 0.50])  # Left-aligned panel
mode_selector = RadioButtons(ax_mode, layer_labels, active=0)

current_layer = "temperature"
current_idx   = 0

# ----------------------------------------------------------
# EPOCH SLIDER
# ----------------------------------------------------------

ax_slider = plt.axes([0.25, 0.05, 0.55, 0.03])
slider = Slider(ax_slider, "Epoch", 0, len(FILES)-1, valinit=0, valstep=1)

# ----------------------------------------------------------
# INITIAL IMAGE (TEMPERATURE)
# ----------------------------------------------------------

img0 = get_epoch(current_idx)
vmin0, vmax0 = robust_limits(img0)
im = ax_map.imshow(
    img0,
    cmap=CM_TEMP,
    vmin=vmin0,
    vmax=vmax0,
    origin="upper"
)

# ----------------------------------------------------------
# COLORBAR WITH UNITS
# ----------------------------------------------------------

cb = fig_map.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
cb.set_label("Temperature (K)")

ax_map.set_title(f"Epoch {DATES[current_idx].date()} — Temperature")

# ----------------------------------------------------------
# RECTANGLES: TS pixel (solid) + reference region (dashed)
# ----------------------------------------------------------

# Initial TS pixel: center
ts_row, ts_col = H // 2, W // 2

rect_ts = Rectangle(
    (ts_col - 0.5, ts_row - 0.5),
    1, 1,
    edgecolor="lime",
    facecolor="none",
    linewidth=1.8
)

# Reference region (initially single pixel from global anomaly)
ref_r0 = ref_auto_row
ref_c0 = ref_auto_col
ref_r1 = ref_auto_row
ref_c1 = ref_auto_col

rect_ref = Rectangle(
    (ref_c0 - 0.5, ref_r0 - 0.5),
    1, 1,
    edgecolor="cyan",
    facecolor="none",
    linestyle="--",
    linewidth=1.8
)

ax_map.add_patch(rect_ts)
ax_map.add_patch(rect_ref)

dragging_ref = False  # tracks right-drag for reference region

# ----------------------------------------------------------
# SECOND FIGURE — TIME SERIES
# ----------------------------------------------------------

fig_ts, ax_diff = plt.subplots(figsize=(9.0, 4.0))
ax_abs = ax_diff.twinx()


# Convert python datetime array → matplotlib float dates
DATES_MPL = mdates.date2num(DATES)

# Force x-axis to interpret values as dates
ax_diff.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_diff.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
fig_ts.autofmt_xdate()


line_abs,  = ax_abs.plot([], [], 'o-', color="tab:blue", markersize=4)
line_diff, = ax_diff.plot([], [], 'o-', color="tab:red", markersize=4)

ax_abs.set_ylabel("Temperature (K)", color="tab:blue")
ax_diff.set_ylabel("ΔK (pixel − reference)", color="tab:red")
ax_diff.set_xlabel("Date")
ax_diff.grid(True)

fig_map.canvas.draw_idle()
fig_ts.canvas.draw_idle()


"""
ASTER Interactive Viewer — Advanced Version
SECTION 4 of 6 — Update Engine (Map, Colourbar, TS, Rectangles)
"""

# ----------------------------------------------------------
# UPDATE RECTANGLES (TS pixel + reference region)
# ----------------------------------------------------------

def update_rectangles():
    # TS pixel → 1×1 green box
    rect_ts.set_xy((ts_col - 0.5, ts_row - 0.5))

    # Reference region → dashed cyan rectangle
    r0, r1 = sorted([ref_r0, ref_r1])
    c0, c1 = sorted([ref_c0, ref_c1])
    width  = (c1 - c0 + 1)
    height = (r1 - r0 + 1)

    rect_ref.set_xy((c0 - 0.5, r0 - 0.5))
    rect_ref.set_width(width)
    rect_ref.set_height(height)

    fig_map.canvas.draw_idle()

# ----------------------------------------------------------
# UPDATE TIME SERIES (ABSOLUTE + DIFFERENCE)
# ----------------------------------------------------------

def update_timeseries():
    """
    Compute and plot:
    • Abs TS (blue, right axis)
    • Difference TS (red, left axis)
    Fully NaN‑safe: if TS pixel or reference series is all NaN,
    the plot clears and displays 'No valid data...'.
    """
    ts_pixel = get_pixel_ts(ts_row, ts_col)
    ts_ref   = get_region_ts(ref_r0, ref_c0, ref_r1, ref_c1)

    # Check NaN validity
    if np.all(np.isnan(ts_pixel)):
        line_abs.set_data([], [])
        line_diff.set_data([], [])
        ax_diff.set_title("No valid data at selected pixel", color="red")
        fig_ts.canvas.draw_idle()
        return

    if np.all(np.isnan(ts_ref)):
        line_abs.set_data([], [])
        line_diff.set_data([], [])
        ax_diff.set_title("No valid data in reference region", color="red")
        fig_ts.canvas.draw_idle()
        return

    ts_diff = ts_pixel - ts_ref

    # Update lines
    
    line_abs.set_data(DATES_MPL, ts_pixel)
    line_diff.set_data(DATES_MPL, ts_diff)
    
    # Explicit x‑limits
    ax_diff.set_xlim(DATES_MPL.min(), DATES_MPL.max())
    ax_abs.set_xlim(DATES_MPL.min(), DATES_MPL.max())
    
    # Axis limits — ΔK
    finite_d = np.isfinite(ts_diff)
    if finite_d.any():
        #lo, hi = np.nanpercentile(ts_diff[finite_d], (2, 98))
        lo = np.nanmin(ts_diff[finite_d])
        hi = np.nanmax(ts_diff[finite_d])
        if lo == hi:
            lo -= 1
            hi += 1
        ax_diff.set_ylim(lo, hi)

    # Axis limits — absolute temperature
    finite_a = np.isfinite(ts_pixel)
    if finite_a.any():
        #lo2, hi2 = np.nanpercentile(ts_pixel[finite_a], (2, 98))
        lo2 = np.nanmin(ts_pixel[finite_a])
        hi2 = np.nanmax(ts_pixel[finite_a])
        if lo2 == hi2:
            lo2 -= 1
            hi2 += 1
        ax_abs.set_ylim(lo2, hi2)

    ax_diff.set_title("Time Series (Absolute & ΔK)", color="black")

    fig_ts.canvas.draw_idle()

# ----------------------------------------------------------
# UPDATE MAP RENDERING BASED ON CURRENT MODE + EPOCH
# ----------------------------------------------------------

def update_map():
    global im, cb, current_layer, current_idx

    layer = current_layer
    idx   = current_idx

    # ------------------------------------------------------
    # 1. ABSOLUTE TEMPERATURE
    # ------------------------------------------------------
    if layer == "temperature":
        arr = get_epoch(idx)
        vmin, vmax = robust_limits(arr)
        cmap = CM_TEMP
        cb_label = "Temperature (K)"
        title = f"Epoch {DATES[idx].date()} — Temperature"

    # ------------------------------------------------------
    # 2. DIFFERENCE MAP (pixel minus reference region)
    # ------------------------------------------------------
    elif layer == "difference":
        ts_ref = get_region_ts(ref_r0, ref_c0, ref_r1, ref_c1)
        if np.all(np.isnan(ts_ref)):
            arr = np.full((H, W), np.nan, dtype=np.float32)
        else:
            arr_pixel = get_epoch(idx)
            arr = arr_pixel - ts_ref[idx]

        vmax = np.nanmax(np.abs(arr))
        vmin = -vmax
        cmap = CM_DIFF
        cb_label = "ΔK"
        title = f"Epoch {DATES[idx].date()} — ΔK"

    # ------------------------------------------------------
    # 3. SUMMARY LAYERS
    # ------------------------------------------------------
    else:
        arr = SUMMARY[layer].copy()

        # Special colormaps
        if layer == "mean_temperature":
            cmap = CM_MEAN
            cb_label = "Temperature (K)"
            vmin, vmax = robust_limits(arr)
        elif layer == "rms_timeseries":
            cmap = CM_RMS
            cb_label = "RMS (K)"
            vmin, vmax = robust_limits(arr)
        elif layer == "trend":
            cmap = CM_TREND
            cb_label = "Trend (K/year)"
            vmax = np.nanmax(np.abs(arr))
            vmin = -vmax
        else:
            cmap = CM_ANOM
            cb_label = "ΔK"
            vmax = np.nanmax(np.abs(arr))
            vmin = -vmax

        # Cleaner title for summaries
        pretty = layer.replace("_", " ").title()
        title = pretty

    # ------------------------------------------------------
    # APPLY TO IMAGE ARTIST
    # ------------------------------------------------------
    im.set_data(arr)
    im.set_clim(vmin, vmax)
    im.set_cmap(cmap)
    ax_map.set_title(title)

    # ------------------------------------------------------
    # UPDATE COLOURBAR
    # ------------------------------------------------------
    cb.set_label(cb_label)
    cb.update_normal(im)

    fig_map.canvas.draw_idle()

# End SECTION 4

"""
ASTER Interactive Viewer — Advanced Version
SECTION 5 of 6 — Interaction (Mouse, Slider, Layer Selector)
"""

# ----------------------------------------------------------
# CALLBACK: SLIDER (Epoch)
# ----------------------------------------------------------

def on_slider(val):
    global current_idx
    current_idx = int(slider.val)
    update_map()
    update_timeseries()
    update_rectangles()

slider.on_changed(on_slider)

# ----------------------------------------------------------
# CALLBACK: LAYER SELECTOR
# ----------------------------------------------------------

def on_mode(label):
    global current_layer
    current_layer = label
    update_map()         # redraw map
    update_timeseries()  # keep TS visible
    update_rectangles()  # ensure rectangles still drawn

mode_selector.on_clicked(on_mode)

# ----------------------------------------------------------
# MOUSE EVENTS ON MAP
# ----------------------------------------------------------

def on_click(event):
    """Left-click selects TS pixel.
       Right-click starts a reference-region drag."""
    global ts_row, ts_col, dragging_ref
    global ref_r0, ref_c0, ref_r1, ref_c1

    if event.inaxes != ax_map:
        return

    # Convert event to pixel indices
    if event.xdata is None or event.ydata is None:
        return

    row = int(round(event.ydata))
    col = int(round(event.xdata))

    # Bounds check
    if not (0 <= row < H and 0 <= col < W):
        return

    # LEFT CLICK = TS pixel
    if event.button == 1:
        ts_row, ts_col = row, col
        update_rectangles()
        update_timeseries()
        update_map()

    # RIGHT CLICK = begin dragging reference region
    elif event.button == 3:
        dragging_ref = True
        ref_r0 = ref_r1 = row
        ref_c0 = ref_c1 = col
        update_rectangles()

fig_map.canvas.mpl_connect("button_press_event", on_click)

# ----------------------------------------------------------
# MOUSE DRAG (update reference region dynamically)
# ----------------------------------------------------------

def on_motion(event):
    global ref_r1, ref_c1

    if not dragging_ref:
        return
    if event.inaxes != ax_map:
        return

    if event.xdata is None or event.ydata is None:
        return

    row = int(round(event.ydata))
    col = int(round(event.xdata))

    # Bounds check
    if not (0 <= row < H and 0 <= col < W):
        return

    ref_r1 = row
    ref_c1 = col
    update_rectangles()

fig_map.canvas.mpl_connect("motion_notify_event", on_motion)

# ----------------------------------------------------------
# MOUSE RELEASE (finalise reference region)
# ----------------------------------------------------------

def on_release(event):
    global dragging_ref
    if dragging_ref:
        dragging_ref = False
        update_rectangles()
        update_timeseries()
        update_map()

fig_map.canvas.mpl_connect("button_release_event", on_release)

"""
ASTER Interactive Viewer — Advanced Version
SECTION 6 of 6 — Final Initialisation & Main Loop
"""

# ----------------------------------------------------------
# SAFETY: ensure reference region variables are properly initialised
# ----------------------------------------------------------

# Convert possible None values into integers
ref_r0 = int(ref_r0)
ref_c0 = int(ref_c0)
ref_r1 = int(ref_r1)
ref_c1 = int(ref_c1)

# Clamp coordinates in case global anomaly pixel was at border
ref_r0 = max(0, min(H-1, ref_r0))
ref_c0 = max(0, min(W-1, ref_c0))
ref_r1 = max(0, min(H-1, ref_r1))
ref_c1 = max(0, min(W-1, ref_c1))

# ----------------------------------------------------------
# DO ONE FULL UPDATE CYCLE ON STARTUP
# ----------------------------------------------------------

update_rectangles()
update_timeseries()
update_map()

print("SECTION 6 loaded successfully. Viewer initialised.")

# ----------------------------------------------------------
# START THE APPLICATION
# ----------------------------------------------------------
plt.show()
