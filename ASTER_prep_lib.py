import re
from datetime import datetime
import rioxarray as rxr
import xarray as xr
from glob import glob
from collections import defaultdict
import numpy as np
import os
from dataclasses import dataclass
from datetime import datetime
import subprocess
import rasterio
from rasterio.enums import Resampling

PATTERN = re.compile(
    r"^AST_08_"
    r"(?P<collection>\d{3})"
    r"(?P<acq_mmddyyyy>\d{8})"
    r"(?P<acq_hhmmss>\d{6})_"
    r"(?P<proc_yyyymmdd>\d{8})"
    r"(?P<proc_hhmmss>\d{6})_"
    r"(?P<layer>SKT(?:_QA_DataPlane2|_QA_DataPlane)?)"
    r"\.tif$"
)

@dataclass
class Ast08Filename:
    collection: str
    acq_datetime: datetime
    proc_datetime: datetime
    layer: str
    filename: str


@dataclass
class GridConfig:
    dst_crs: str = "EPSG:32719"   # UTM 19S (adjust if you later move AOI)
    res: int = 90                 # 90 m native ASTER SKT
    # Bounding box in WGS84 (lon/lat)
    bbox: tuple = (-68.22, -21.75, -67.65, -21.55)  # (minx, miny, maxx, maxy)

# User masking options
@dataclass
class MaskOptions:
    mask_cloud: bool = True
    mask_snow: bool = True

def parse_ast08_filename(path):
    fname = os.path.basename(path)
    m = PATTERN.match(fname)
    if not m:
        raise ValueError(f"Invalid AST_08 filename: {fname}")

    acq_date = datetime.strptime(
        m.group("acq_mmddyyyy") + m.group("acq_hhmmss"),
        "%m%d%Y%H%M%S"
    )
    proc_date = datetime.strptime(
        m.group("proc_yyyymmdd") + m.group("proc_hhmmss"),
        "%Y%m%d%H%M%S"
    )

    return Ast08Filename(
        collection=m.group("collection"),
        acq_datetime=acq_date,
        proc_datetime=proc_date,
        layer=m.group("layer"),
        filename=path
    )

def gdalwarp_near_clip(src, dst, grid: GridConfig):
    """
    Nearest-neighbour reprojection to fixed grid + clip to bbox (WGS84),
    producing perfectly aligned outputs across dates. (GDALwarp behavior) 
    """
    minx, miny, maxx, maxy = grid.bbox
    cmd = [
        "gdalwarp",
        "-r", "near",
        "-t_srs", grid.dst_crs,
        "-tr", str(grid.res), str(grid.res),
        "-te_srs", "EPSG:4326",
        "-te", str(minx), str(miny), str(maxx), str(maxy),
        "-overwrite",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        src, dst
    ]
    subprocess.check_call(cmd)

def mosaic_daily_gdalwarp(src_files, dst_path, grid: GridConfig):
    """
    Mosaic a list of already-warped files into one aligned mosaic for that day.
    """
    minx, miny, maxx, maxy = grid.bbox
    cmd = [
        "gdalwarp",
        "-r", "near",
        "-t_srs", grid.dst_crs,
        "-tr", str(grid.res), str(grid.res),
        "-te_srs", "EPSG:4326",
        "-te", str(minx), str(miny), str(maxx), str(maxy),
        "-overwrite",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        *src_files,
        dst_path
    ]
    subprocess.check_call(cmd)

def warp_and_mosaic(masked_dir, warped_dir, mosaic_dir, grid: GridConfig):
    os.makedirs(warped_dir, exist_ok=True)
    os.makedirs(mosaic_dir, exist_ok=True)

    files = glob(os.path.join(masked_dir, "*_SKT_masked.tif"))
    # Group by acquisition DATE (ignore track)
    groups = defaultdict(list)
    for f in files:
        meta = parse_ast08_filename(os.path.basename(f).replace("_masked",""))
        groups[meta.acq_datetime.date()].append(f)

    # Per-scene warp first (near + clip)
    warped_by_day = defaultdict(list)
    for acq_date, flist in groups.items():
        for src in flist:
            base = os.path.basename(src).replace("_masked.tif", "_warped.tif")
            dst = os.path.join(warped_dir, base)
            gdalwarp_near_clip(src, dst, grid)
            warped_by_day[acq_date].append(dst)

    # Then mosaic by day
    for acq_date, warped_list in warped_by_day.items():
        out = os.path.join(mosaic_dir, f"AST08_{acq_date}_mosaic.tif")
        mosaic_daily_gdalwarp(warped_list, out, grid)
        print("Wrote:", out)

def mask_scene_native(skt_path, qa_path, out_masked_path, *, mask_cloud=True, mask_snow=True, mask_sat=True, mask_TES=True, mask_cold=True, mask_hot=True, mask_emis=True, mask_atms=True):
    """
    Apply mask in native ASTER geometry so QA aligns perfectly with SKT.
    bit 0 = cloud, bit 1 = snow/ice (AST_08 primary QA layer). [1](https://lpdaac.usgs.gov/documents/2265/ASTER_User_Guide_V4_pcP80n5.pdf)[2](https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/default.htm)
    """
    with rasterio.open(skt_path) as ds_skt, rasterio.open(qa_path) as ds_qa:
        skt = ds_skt.read(1)  # float (Kelvin)
        qa  = ds_qa.read(1)   # uint16

        # Build mask
        mask = np.zeros_like(qa, dtype=bool)
        if mask_cloud:
            mask |= (qa & (1 << 0)) > 0
        if mask_snow:
            mask |= (qa & (1 << 1)) > 0
        if mask_sat:
            mask |= (qa & (1 << 2)) > 0
        if mask_TES:
            mask |= (qa & (1 << 3)) > 0
        if mask_cold:
            mask |= (qa & (1 << 4)) > 0
        if mask_hot:
            mask |= (qa & (1 << 5)) > 0
        if mask_emis:
            mask |= (qa & (1 << 6)) > 0
        if mask_atms:
            mask |= (qa & (1 << 7)) > 0


        # Apply mask: set to NaN (keeps nodata semantics in float)
        skt_masked = skt.astype("float32")
        skt_masked[mask] = np.nan

        # Write masked SKT in native geometry
        profile = ds_skt.profile.copy()
        profile.update(dtype="float32", nodata=np.nan, compress="DEFLATE", predictor=3, tiled=True, blockxsize=256, blockysize=256)

        os.makedirs(os.path.dirname(out_masked_path), exist_ok=True)
        with rasterio.open(out_masked_path, "w", **profile) as dst:
            dst.write(skt_masked, 1)
