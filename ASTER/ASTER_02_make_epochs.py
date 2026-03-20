import ASTER_prep_lib as prep
import os
from glob import glob

def run_pipeline(input_dir, work_dir, grid: GridConfig, mask_opts: MaskOptions):
    masked_dir = os.path.join(work_dir, "masked_native")
    warped_dir = os.path.join(work_dir, "warped_clipped")
    mosaic_dir = os.path.join(work_dir, "mosaic_daily")
    os.makedirs(masked_dir, exist_ok=True)

    # 1) Mask in native geometry for each scene
    skt_files = sorted(glob(os.path.join(input_dir, "*_SKT.tif")))
    for skt in skt_files:
        qa = skt.replace("_SKT.tif", "_SKT_QA_DataPlane.tif")
        if not os.path.exists(qa):
            print("WARNING: missing QA for", skt)
            continue
        base = os.path.basename(skt).replace("_SKT.tif", "_SKT_masked.tif")
        out_masked = os.path.join(masked_dir, base)
        prep.mask_scene_native(
            skt, qa, out_masked,
            mask_cloud=mask_opts.mask_cloud,
            mask_snow=mask_opts.mask_snow
        )

    # 2) Warp (nearest), clip, and mosaic per-day
    prep.warp_and_mosaic(masked_dir, warped_dir, mosaic_dir, grid)

if __name__ == "__main__":
    grid = prep.GridConfig(
        dst_crs="EPSG:32719",
        res=90,
        bbox=(-68.22, -21.75, -67.65, -21.55)  # lon/lat
    )
    opts = prep.MaskOptions(mask_cloud=True, mask_snow=True)

    run_pipeline(
        input_dir="./aster_ast08",   # your download folder
        work_dir="./AST08/epochs", # pipeline workspace
        grid=grid,
        mask_opts=opts
    )
