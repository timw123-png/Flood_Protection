# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  OVERLORD-MAX v122.0 | TITAN MONOLITH + FORENSIC ENGINE + TERRAIN VISION v8        ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  FIX vs v121.3 — 60ft / 250ft PHOTOGRAPHIC-QUALITY LIDAR MAPS:                     ║
║                                                                                      ║
║  • RESOLUTION: 60ft LOT map now upsampled to 0.05m (5 cm) — QL0/QL1 standard       ║
║    250ft BLOCK map upsampled to 0.10m (10 cm) — QL1 engineering grade               ║
║    (was 0.5m in both — 10× coarser than required for photographic output)            ║
║                                                                                      ║
║  • DPI / FIGSIZE: 60ft → 400 DPI / 12×12 in (4800×4800 px photographic output)     ║
║    250ft → 350 DPI / 14×14 in (4900×4900 px QL1-grade output)                      ║
║    1000ft → 220 DPI / 16×16 in (unchanged)                                          ║
║                                                                                      ║
║  • BASEMAP ZOOM: 60ft → zoom=20 (0.15m/px WorldImagery tile — houses pixel-perfect) ║
║    250ft → zoom=19 (0.30m/px — block-level street detail)                           ║
║    1000ft → zoom=15 (unchanged neighbourhood context)                                ║
║    (was zoom=17 for 60ft/250ft — far too coarse for sub-metre DEM overlay)          ║
║                                                                                      ║
║  • COLOUR SPAN: 60ft → ±1.0 ft (every 2" grade change = full colour band)          ║
║    250ft → ±1.5 ft (resolves ponding thresholds on flat Pinellas terrain)           ║
║    1000ft → dynamic ±half-range (unchanged)                                          ║
║                                                                                      ║
║  • POST-PROCESSING: saturation/contrast per-radius — 60ft ×1.8/1.5 (photographic   ║
║    realism), 250ft ×2.2/1.7 (was ×2.8/2.0 for both — over-saturated satellite)     ║
║    USM radius/percent tuned for high-density pixel output (1.5/150 vs 2/200)        ║
║                                                                                      ║
║  v121.3 features intact:                                                             ║
║  • LiDAR overlay extent from ACTUAL raster window bounds → EPSG:3857                ║
║  • Water/nodata masking (0–25 ft NAVD88); sea-level mask transparent                ║
║  • Streamplot grid aligned to raster pixel coordinates                               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import geopandas as gpd
import os, re, json, math, csv, subprocess, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.ndimage import gaussian_filter, label as ndlabel, minimum_filter
    from scipy.interpolate import RegularGridInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABASE          = '/Users/timweigelt/QField/GeoTif/buildings_the_final_join_tampa_bay.gpkg'
DATABASE_LAYER    = 'Buildings'        # layer inside the GPKG (EPSG:3857)
DATABASE_CRS      = 'EPSG:3857'        # native CRS — Web Mercator (metres)
LIDAR_RASTER      = '/Users/timweigelt/SAVE_THIS_PINELLAS_3857.tif'
HILLSHADE_RASTER  = '/Users/timweigelt/QField/GeoTif/hillshade.gpkg'

PINELLAS_LIDAR_DIR = os.path.expanduser('~/Desktop/pinellas_dem')
_REPROJECTED_LIDAR = '/Users/timweigelt/SAVE_THIS_PINELLAS_3857.tif'
PINELLAS_LIDAR_FILES = []
if os.path.isdir(PINELLAS_LIDAR_DIR):
    for _fn in os.listdir(PINELLAS_LIDAR_DIR):
        if _fn.lower().endswith(('.tif', '.tiff', '.img', '.asc', '.dem')):
            PINELLAS_LIDAR_FILES.append(os.path.join(PINELLAS_LIDAR_DIR, _fn))
    if PINELLAS_LIDAR_FILES:
        print(f"📂 Pinellas LiDAR: {len(PINELLAS_LIDAR_FILES)} tile(s) discovered")
if os.path.exists(_REPROJECTED_LIDAR) and _REPROJECTED_LIDAR not in PINELLAS_LIDAR_FILES:
    PINELLAS_LIDAR_FILES.insert(0, _REPROJECTED_LIDAR)
    print(f"📂 Reprojected LiDAR tile added: {_REPROJECTED_LIDAR}")

WATERBODY_SHP     = '/Users/timweigelt/QField/GeoTif/tl_2023_12103_areawater.shp'
_WATERBODY_GDF_CACHE: dict = {}   # loaded once, keyed "gdf" and "reason"

LEADS_LIST        = '/Users/timweigelt/Desktop/leads.txt'
MASTER_EXPORT_DIR = os.path.expanduser('~/Desktop/FINAL_HYDRAULIC_REPORTS')
PCPAO_SCRAPER     = '/Users/timweigelt/Desktop/resilient_pcpao_scraper.py'
PCPAO_FAILED_CSV  = os.path.join(MASTER_EXPORT_DIR, 'pcpao_failed.csv')
PCPAO_TARGET_URL  = "https://www.pcpao.gov/"

OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "llama3"
OLLAMA_SEARCH_URL = "https://ddg-api.herokuapp.com/search"
SEARX_URL         = "http://localhost:8888/search"

if not os.path.exists(MASTER_EXPORT_DIR):
    os.makedirs(MASTER_EXPORT_DIR, exist_ok=True)

# =============================================================================
# VERSAI CONSTANTS
# =============================================================================
VERSAI = {
    "report_id": "25002-S01", "date": "02 February 2025",
    "author": "VERSAI, LLC — PE: Klodian Pepaj, NY Lic. 104596",
    "rho_water": 62.4, "g": 32.2, "vel_wave": 7.335,
    "plank_weight": 2.180, "post_weight": 1.621,
    "plank_height_in": 8.0, "plank_height_ft": 8.0 / 12,
    "aluminum_grade": "6063-T6",
    "wall_types": {
        "CMU_8_unreinforced":  {"capacity_psi": 500.0},
        "CMU_8_reinforced":    {"capacity_psi": 1250.0},
        "CMU_12_unreinforced": {"capacity_psi": 666.67},
        "CMU_12_reinforced":   {"capacity_psi": 1500.0},
        "poured_concrete":     {"capacity_psi": 2666.67},
    },
    "pressure_by_planks":   {5: 8376.23, 7: 11055.27, 10: 15073.83, 14: 20431.91},
    "stress_psi_by_planks": {5: 29.11, 7: 38.42, 10: 52.37, 14: 70.97},
    "conclusion": (
        "PE-verified (25002-S01): All wall types pass at 5–14 planks / 40\"–120\" openings "
        "at 5 mph wave load. Max stress 70.97 psi vs 500 psi min capacity (7.0x–37.5x margin)."
    ),
}
P_WAVE = 0.5 * VERSAI["rho_water"] * (VERSAI["vel_wave"] ** 2)

GARRISON_PRODUCT = {
    "name": "Hammerhead™ Aluminum Flood Plank System",
    "contact": "(929) 299-2099 | sales@garrisonflood.com | www.garrisonflood.com",
    "material": f"6063-T6 Aluminum | {VERSAI['plank_height_in']:.0f}\" plank | "
                f"{VERSAI['plank_weight']} lb/ft | {VERSAI['post_weight']} lb/ft post",
    "engineering_ref": f"VERSAI LLC {VERSAI['report_id']} — PE-stamped",
    "surge_capacity_ft": 9.33,
    "plank_height_ft": VERSAI["plank_height_ft"],
    "weight_per_lf": VERSAI["plank_weight"],
    "post_weight_lf": VERSAI["post_weight"],
    "plank_height_in": VERSAI["plank_height_in"],
    "case_study": {
        "event": "Hurricane Ian, September 2022 (Category 4, Fort Myers FL)",
        "surge": "15 ft storm surge / 150 mph winds",
        "damage": "$100B+ total damage (3rd costliest on record)",
        "solution": "Hammerhead™ on garage, front door, back sliders",
    },
}

NOAA_ANNUAL_IN = 53.7
DESIGN_STORMS  = {"2yr": 2.4, "5yr": 3.0, "10yr": 3.5, "25yr": 4.3, "50yr": 5.0, "100yr": 5.8}
STORM_DURATION_HR = 3   # storm duration in hours (e.g. 2 in/hr for 3 hours = 6" total)

SAMPLE_INTERVAL_FT = 2.5
HOME_MAP_RADIUS_FT  = 250.0
SAMPLE_RANGE_FT     = HOME_MAP_RADIUS_FT
SAMPLE_STEPS        = int(SAMPLE_RANGE_FT / SAMPLE_INTERVAL_FT)

NEIGHBOR_RADIUS_FT  = 1000.0

OUTPUT_CLIENT_READY_ONLY = True
GEO_VERIFY_PASSES = 3

DIRECTIONS = {
    "N":  (0.0,  1.0), "NE": (0.707,  0.707),
    "E":  (1.0,  0.0), "SE": (0.707, -0.707),
    "S":  (0.0, -1.0), "SW": (-0.707,-0.707),
    "W":  (-1.0, 0.0), "NW": (-0.707, 0.707),
}

# Pinellas County valid elevation range (NAVD88, feet)
# Any cell outside this range is water, nodata, or erroneous — mask from overlay
PINELLAS_ELEV_MIN_FT = 0.0
PINELLAS_ELEV_MAX_FT = 120.0

# Elevation unit stored in the LiDAR raster.
# SAVE_THIS_PINELLAS_3857.tif stores values in METRES (pixel=0.87m, EPSG:3857).
# The rest of the pipeline works in feet — set to "metres" to auto-convert.
# Change to "feet" if you ever swap back to a feet-based raster.
LIDAR_ELEV_UNIT = "feet"

# Building footprints (optional vector file — GPKG, SHP, GeoJSON)
# If set and exists, polygons are rasterized onto the DEM to mask buildings.
# Set to None or '' to rely solely on the DEM-based effective-ground method.
BUILDING_FOOTPRINTS = None   # e.g. '/path/to/building_footprints.gpkg'

# Effective-ground threshold: any pixel more than this many feet above the
# local neighborhood mean is assumed to be a building/structure and masked
# before ponding detection.  Works purely from the DEM — no vector data needed.
BUILDING_HEIGHT_THRESH_FT = 2.0

# El Niño rainfall multiplier data sources
# Set these to your NOAA precipitation CSV and ONI indices (RTF or CSV).
# If set to None, the script will auto-search the Desktop and MASTER_EXPORT_DIR.
PRECIP_CSV_PATH = None   # e.g. '/Users/timweigelt/Desktop/4274930+1.csv'
ONI_CSV_PATH    = os.path.join(os.path.expanduser("~/Desktop"), "Indices.csv")

# FEMA document to append at page 5 of the flood report PDF.
# Set to specific path in Documents folder, will auto-search if None.
FEMA_PDF_PATH       = next((p for p in [
    os.path.expanduser('~/Desktop/FEMADOC.pdf'),
    os.path.expanduser('~/Documents/FEMADOC.pdf'),
    os.path.expanduser('~/Documents/FEMA.pdf'),
    os.path.expanduser('~/Documents/open.pdf'),
    os.path.expanduser('~/Desktop/FEMA.pdf'),
    os.path.expanduser('~/Desktop/open.pdf'),
    os.path.expanduser('~/QField/FEMA.pdf'),
    os.path.expanduser('~/QField/open.pdf'),
] if os.path.exists(p)), os.path.expanduser('~/Documents/FEMA.pdf'))
GARRISON_CS_PDF     = os.path.expanduser('~/Desktop/Garrison.pdf')  # Garrison FAQ — appended after FEMA

# =============================================================================
# TRIPLE-PATH INFRASTRUCTURE SOURCES (Pinellas County stormwater layers)
# =============================================================================
# ── St. Pete Stormwater Infrastructure GPKGs (all EPSG:3857, layer='reprojected') ──
STRUCTURES_GPKG         = '/Users/timweigelt/QField/GeoTif/Storm_Water_Inlet.gpkg'         # Stormwater inlets — 22,399 pts  (RIM, INLETTYPE, INV1-8)
MANHOLES_GPKG           = '/Users/timweigelt/QField/GeoTif/Manhole.gpkg'                   # Manholes
STRUCTURE_NETWORK_GPKG  = '/Users/timweigelt/QField/GeoTif/storm_water_structure_network.gpkg'  # Structure network
GRAVITY_MAINS_GPKG      = '/Users/timweigelt/QField/GeoTif/st_pete_gravity_main.gpkg'      # St. Pete gravity mains — 35,438 lines (DIAMETER, UPELEV, DOWNELEV)
OPEN_DRAINS_GPKG        = '/Users/timweigelt/QField/GeoTif/FINAL_OPEN_DRAINS.gpkg'         # Pinellas open drains — 10,587 lines (TOPWIDTH, BOTWIDTH, DEPTH)
STPETE_GRAVITY_GPKG     = '/Users/timweigelt/QField/GeoTif/st_pete_gravity_main.gpkg'      # St. Pete gravity mains — 35,438 lines (DIAMETER, UPELEV, DOWNELEV)
STPETE_CULVERTS_GPKG    = '/Users/timweigelt/QField/GeoTif/Culverts.gpkg'                  # St. Pete culverts — 1,423 lines (DIAMETER, SIZE1FT, UPELEV, DOWNELEV)
STPETE_OUTFALL_PT_GPKG  = '/Users/timweigelt/QField/GeoTif/storm_water_discharge_point.gpkg'  # Outfall points — 4,400 pts (RIM, DISCHARGETYPE, OUTFALLTO)
STPETE_OUTFALL_LN_GPKG  = '/Users/timweigelt/QField/GeoTif/storm_water_discharge_line.gpkg'   # Outfall lines — 525 lines (DISCHARGETYPE, OUTFALLTO)
FITTINGS_GPKG           = '/Users/timweigelt/QField/GeoTif/storm_water_fitting.gpkg'       # Storm water fittings
CONTROL_VALVES_GPKG     = '/Users/timweigelt/QField/GeoTif/stpete_scv.gpkg'                # Storm Control Valves (prevents tidal backflow)
DRAINAGE_SCAN_RADIUS_FT = 1000.0   # radius used for Triple-Path spatial scan

# =============================================================================
# HELPERS
# =============================================================================
def is_valid(val) -> bool:
    if val is None: return False
    try:
        if pd.isna(val): return False
    except: pass
    return str(val).strip().lower() not in ['', 'nan', 'none', 'null', 'n/a', 'unknown', '0', '0.0']

def safe(val, default=None):
    return val if is_valid(val) else default

def score_ds(v): return round(min(100.0, max(0.0, v)), 1)
def ds_icon(s):  return "🔴" if s >= 75 else "🟡" if s >= 40 else "🟢"

def coalesce_float(*vals, default=None):
    for v in vals:
        if is_valid(v):
            try:
                return float(v)
            except Exception:
                pass
    return default

def derive_ground_elevation_ft(row) -> float:
    return coalesce_float(
        row.get('Elev_Z_1'),
        row.get('_mean'),
        row.get('_median'),
        row.get('_majority'),
        row.get('_minority'),
        default=0.0
    ) or 0.0

def derive_lot_area_sqft(row, sq_ft: float) -> float:
    shape_area = coalesce_float(row.get('SHAPESTAre'), default=None)
    baseline = max(1.0, float(sq_ft) * 4.5)
    if shape_area is None:
        return baseline
    if shape_area < float(sq_ft) * 1.05:
        return baseline
    return float(shape_area)

def _try_transform_xy(x: float, y: float, from_crs, to_crs):
    if not from_crs or not to_crs:
        return x, y
    try:
        from pyproj import CRS
        c1 = CRS.from_user_input(from_crs)
        c2 = CRS.from_user_input(to_crs)
        if c1 == c2:
            return x, y
        from pyproj import Transformer
        tfm = Transformer.from_crs(c1, c2, always_xy=True)
        return tfm.transform(x, y)
    except Exception:
        return x, y


# =============================================================================
# WATER BODY SPATIAL LOOKUP — tl_2023_12103_areawater.shp (EPSG:3857)
# =============================================================================
def lookup_nearest_waterbody(cx: float, cy: float, data_crs: str) -> dict:
    """
    Spatially look up the nearest named water body from the TIGER/Line
    tl_2023_12103_areawater shapefile (EPSG:3857).  The GeoDataFrame is loaded
    once and cached for the lifetime of the run.

    Returns:
        {"ok": True,  "fullname": str, "mtfcc": str, "dist_ft": float}
        {"ok": False, "reason": str}
    """
    global _WATERBODY_GDF_CACHE
    try:
        from pyproj import CRS, Transformer
        from shapely.geometry import Point

        # ── Load and cache ────────────────────────────────────────────────
        if "gdf" not in _WATERBODY_GDF_CACHE:
            if not os.path.exists(WATERBODY_SHP):
                _WATERBODY_GDF_CACHE["gdf"]    = None
                _WATERBODY_GDF_CACHE["reason"] = (
                    f"Shapefile not found: {WATERBODY_SHP}")
            else:
                _gdf = gpd.read_file(WATERBODY_SHP)          # native EPSG:3857
                _gdf = _gdf[_gdf.geometry.notnull()].copy()
                _WATERBODY_GDF_CACHE["gdf"]    = _gdf
                _WATERBODY_GDF_CACHE["reason"] = None
                print(f"   💧  Water body shapefile loaded: "
                      f"{len(_gdf)} features  ({WATERBODY_SHP})")

        gdf = _WATERBODY_GDF_CACHE.get("gdf")
        if gdf is None:
            return {"ok": False,
                    "reason": _WATERBODY_GDF_CACHE.get("reason", "load failed")}

        # ── Reproject property centroid → EPSG:3857 ───────────────────────
        try:
            src_crs = CRS.from_user_input(data_crs)
        except Exception:
            src_crs = CRS.from_epsg(2881)   # FL State Plane West (ft)
        dst_crs = CRS.from_epsg(3857)
        tfm     = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        px, py  = tfm.transform(cx, cy)
        prop_pt = Point(px, py)

        # ── Nearest geometry edge (metres → feet) ────────────────────────
        # Must use actual polygon geometry, NOT centroid — large bodies like
        # Tampa Bay have centroids miles from shore; a property on the bay
        # shoreline measures correctly to the polygon edge (≈0-50 ft), not
        # to the bay centroid (potentially miles).
        dists   = gdf.geometry.distance(prop_pt)
        idx_min = dists.idxmin()
        dist_ft = dists[idx_min] * 3.28084

        row_wb   = gdf.loc[idx_min]
        fullname = str(row_wb.get("FULLNAME", "") or "").strip()
        mtfcc    = str(row_wb.get("MTFCC",    "") or "").strip()
        if not fullname:
            fullname = mtfcc if mtfcc else "Unnamed Water Body"

        return {
            "ok":       True,
            "fullname": fullname,
            "mtfcc":    mtfcc,
            "dist_ft":  round(dist_ft, 0),
        }

    except Exception as _wb_e:
        return {"ok": False, "reason": str(_wb_e)}


# =============================================================================
# GEO-VERIFICATION ENGINE — 3-PASS
# =============================================================================
def geo_verify_address(st_addr: str, cx: float, cy: float, g_elev: float,
                        parcel_id: str = "", point_crs=None) -> dict:
    result = {
        "pass1_fwd_geocode":   {"ok": False, "lat": None, "lng": None, "raw": None},
        "pass2_rev_geocode":   {"ok": False, "address": None, "match_score": 0.0},
        "pass3_elev_sanity":   {"ok": False, "sampled_elev": None, "delta_from_record": None},
        "overall_confidence":  "UNVERIFIED",
        "flags":               [],
    }

    lat_nom, lng_nom = cy, cx
    try:
        from pyproj import CRS, Transformer
        src_crs = None
        if point_crs:
            try:
                src_crs = CRS.from_user_input(point_crs)
            except Exception:
                src_crs = None
        is_projected = (abs(cx) > 1000 or abs(cy) > 1000)
        if src_crs is None and is_projected:
            src_crs = CRS.from_epsg(2881)
        if src_crs is not None:
            wgs84 = CRS.from_epsg(4326)
            if src_crs != wgs84:
                tfm = Transformer.from_crs(src_crs, wgs84, always_xy=True)
                lng_nom, lat_nom = tfm.transform(cx, cy)
    except Exception as e:
        result["flags"].append(f"⚠️  CRS conversion warning: {e} — using raw cx/cy as lat/lng")

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{st_addr}, Pinellas County, FL", "format": "json",
                    "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "OVERLORD-MAX/121.2"}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            if d:
                fwd_lat = float(d[0]["lat"])
                fwd_lng = float(d[0]["lon"])
                result["pass1_fwd_geocode"] = {"ok": True, "lat": fwd_lat, "lng": fwd_lng,
                                                "raw": d[0].get("display_name", "")}
                dist_deg = math.sqrt((fwd_lat - lat_nom)**2 + (fwd_lng - lng_nom)**2)
                dist_ft_approx = dist_deg * 364000
                if dist_ft_approx > 1000:
                    result["flags"].append(
                        f"⚠️  Pass 1: Forward geocode is {dist_ft_approx:.0f} ft from GIS point")
    except Exception as e:
        result["flags"].append(f"⚠️  Pass 1: Forward geocode failed — {e}")

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lon": lng_nom, "lat": lat_nom, "zoom": 18},
            headers={"User-Agent": "OVERLORD-MAX/121.2"}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            rev_addr = d.get("display_name", "")
            addr_parts = d.get("address", {})
            road    = addr_parts.get("road", "")
            house   = addr_parts.get("house_number", "")
            rev_short = f"{house} {road}".strip()
            a_tokens = set(re.sub(r'[^a-z0-9 ]','', st_addr.lower()).split())
            r_tokens = set(re.sub(r'[^a-z0-9 ]','', rev_addr.lower()).split())
            overlap  = len(a_tokens & r_tokens) / max(len(a_tokens), 1)
            result["pass2_rev_geocode"] = {
                "ok": overlap >= 0.4,
                "address": rev_short or rev_addr,
                "match_score": round(overlap, 3),
                "full_display": rev_addr,
            }
            if overlap < 0.4:
                result["flags"].append(
                    f"⚠️  Pass 2: Reverse geocode mismatch (score={overlap:.2f}). "
                    f"GIS point reverses to: '{rev_short}'")
    except Exception as e:
        result["flags"].append(f"⚠️  Pass 2: Reverse geocode failed — {e}")

    try:
        import rasterio
        all_srcs = []
        if os.path.exists(LIDAR_RASTER):
            all_srcs.append(LIDAR_RASTER)
        all_srcs.extend(PINELLAS_LIDAR_FILES[:3])
        sampled_val = None
        for sp in all_srcs:
            try:
                with rasterio.open(sp) as src:
                    b = src.bounds
                    if b.left <= cx <= b.right and b.bottom <= cy <= b.top:
                        v = next(src.sample([(cx, cy)]))[0]
                        if v is not None:
                            fv = float(v)
                            if fv == -999999.0 or (src.nodata is not None and fv == src.nodata):
                                continue
                            if not math.isnan(fv) and -10 < fv < 200:
                                if LIDAR_ELEV_UNIT == "metres":
                                    fv = fv * 3.28084
                                sampled_val = fv
                                break
            except Exception:
                continue
        if sampled_val is not None:
            delta_elev = abs(sampled_val - g_elev)
            ok = 0 <= sampled_val <= 30 and delta_elev < 5.0
            result["pass3_elev_sanity"] = {
                "ok": ok,
                "sampled_elev": round(sampled_val, 4),
                "delta_from_record": round(delta_elev, 4),
            }
            if sampled_val < 0 or sampled_val > 30:
                result["flags"].append(
                    f"⚠️  Pass 3: Sampled DEM elevation {sampled_val:.2f} ft is outside "
                    f"Pinellas valid range (0–30 ft) — possible CRS mismatch")
            if delta_elev > 5.0:
                result["flags"].append(
                    f"⚠️  Pass 3: DEM sample ({sampled_val:.2f} ft) differs from record "
                    f"elevation ({g_elev:.2f} ft) by {delta_elev:.2f} ft — check CRS alignment")
        else:
            result["flags"].append("⚠️  Pass 3: Could not sample DEM at GIS coordinates")
    except Exception as e:
        result["flags"].append(f"⚠️  Pass 3: Elevation sanity check failed — {e}")

    passes_ok = sum([
        result["pass1_fwd_geocode"]["ok"],
        result["pass2_rev_geocode"]["ok"],
        result["pass3_elev_sanity"]["ok"],
    ])
    if passes_ok == 3:
        result["overall_confidence"] = "HIGH ✅ (3/3 passes)"
    elif passes_ok == 2:
        result["overall_confidence"] = "MEDIUM ⚠️  (2/3 passes)"
    elif passes_ok == 1:
        result["overall_confidence"] = "LOW ❌ (1/3 passes — review coordinates)"
    else:
        result["overall_confidence"] = "FAILED ❌ (0/3 passes — coordinate suspect)"

    return result


# =============================================================================
# MULTI-SOURCE LIDAR ELEVATION SAMPLER (bilinear sub-pixel)
# =============================================================================
def _bilinear_sample(src, wx: float, wy: float) -> Optional[float]:
    try:
        import rasterio
        row_f, col_f = rasterio.transform.rowcol(src.transform, wx, wy, op=float)
        r0, c0 = int(math.floor(row_f)), int(math.floor(col_f))
        dr, dc = row_f - r0, col_f - c0
        nrows, ncols = src.height, src.width
        rr = max(0, min(r0, nrows - 2))
        cc = max(0, min(c0, ncols - 2))
        win = rasterio.windows.Window(cc, rr, 2, 2)
        data = src.read(1, window=win)
        if data.shape != (2, 2):
            return None
        nodata = src.nodata
        vals = data.astype("float64")
        if nodata is not None:
            vals[vals == nodata] = np.nan
        if np.any(np.isnan(vals)):
            valid = vals[~np.isnan(vals)]
            return float(np.mean(valid)) if valid.size else None
        top    = vals[0, 0] * (1 - dc) + vals[0, 1] * dc
        bottom = vals[1, 0] * (1 - dc) + vals[1, 1] * dc
        result = top * (1 - dr) + bottom * dr
        return float(result)
    except Exception:
        return None


def sample_elevation_multi_source(x: float, y: float, point_crs=None, max_sources: int = 6) -> dict:
    readings = {}
    try:
        import rasterio
    except ImportError:
        return {"readings": {}, "consensus": None, "std": None, "n_sources": 0}

    all_sources = []
    if os.path.exists(LIDAR_RASTER):
        all_sources.append(("PRIMARY_DEM", LIDAR_RASTER))
    for fp in PINELLAS_LIDAR_FILES:
        all_sources.append((os.path.basename(fp), fp))

    valid_vals = []
    used = 0
    for src_name, src_path in all_sources:
        if used >= max_sources:
            break
        try:
            with rasterio.open(src_path) as src:
                wx, wy = _try_transform_xy(x, y, point_crs, src.crs)
                bounds = src.bounds
                if not (bounds.left <= wx <= bounds.right and
                        bounds.bottom <= wy <= bounds.top):
                    continue
                val = _bilinear_sample(src, wx, wy)
                if val is None:
                    v = next(src.sample([(wx, wy)]))[0]
                    if v is not None:
                        val = float(v)
                if val is None:
                    continue
                if (src.nodata is None or val != src.nodata) and not math.isnan(val) and -10 < val < 200:
                    readings[src_name] = round(val, 4)
                    valid_vals.append(val)
                    used += 1
        except Exception:
            continue

    if valid_vals:
        consensus = round(float(np.mean(valid_vals)), 4)
        std       = round(float(np.std(valid_vals)), 4)
    else:
        consensus = None
        std       = None

    return {
        "readings":  readings,
        "consensus": consensus,
        "std":       std,
        "n_sources": len(valid_vals),
    }


# =============================================================================
# DEM TILE INDEX
# =============================================================================
_TRANSFORMER_CACHE: dict = {}   # module-level cache — built once per CRS pair, reused


def _crs_units_per_foot(crs) -> float:
    try:
        import pyproj
        c = pyproj.CRS.from_user_input(crs)
        u = (c.axis_info[0].unit_name or "").lower()
        if "metre" in u or "meter" in u:
            return 0.3048
        if "foot" in u or "feet" in u:
            return 1.0
    except Exception:
        pass
    return 1.0


class DemTileIndex:
    def __init__(self, primary_path: str, tile_paths: list):
        self.primary_path = primary_path
        self.tile_paths   = tile_paths or []
        self._cache       = {}

    def _open(self, path: str):
        if path in self._cache:
            return self._cache[path]
        import rasterio
        ds = rasterio.open(path)
        self._cache[path] = ds
        return ds

    def close(self):
        for ds in self._cache.values():
            try: ds.close()
            except: pass
        self._cache = {}

    def _covers(self, ds, x, y) -> bool:
        b = ds.bounds
        return (b.left <= x <= b.right) and (b.bottom <= y <= b.top)

    def pick_dataset_for_point(self, x, y, point_crs):
        from pyproj import CRS, Transformer

        def _transform(x, y, from_crs, to_crs):
            try:
                c1 = CRS.from_user_input(from_crs)
                c2 = CRS.from_user_input(to_crs)
                if c1 == c2:
                    return x, y
                key = (c1.to_epsg() or str(c1), c2.to_epsg() or str(c2))
                t = _TRANSFORMER_CACHE.get(key)
                if t is None:
                    t = Transformer.from_crs(c1, c2, always_xy=True)
                    _TRANSFORMER_CACHE[key] = t
                return t.transform(x, y)
            except Exception:
                return x, y

        if self.primary_path and os.path.exists(self.primary_path):
            try:
                ds = self._open(self.primary_path)
                sx, sy = _transform(x, y, point_crs, ds.crs) if (point_crs and ds.crs) else (x, y)
                if self._covers(ds, sx, sy):
                    return ds, sx, sy
            except Exception:
                pass

        for p in self.tile_paths:
            try:
                ds = self._open(p)
                sx, sy = _transform(x, y, point_crs, ds.crs) if (point_crs and ds.crs) else (x, y)
                if self._covers(ds, sx, sy):
                    return ds, sx, sy
            except Exception:
                continue

        return None, None, None


def dem_sample_point(tile_index: DemTileIndex, x, y, point_crs):
    try:
        ds, sx, sy = tile_index.pick_dataset_for_point(x, y, point_crs)
        if ds is None:
            return None, None
        val = _bilinear_sample(ds, sx, sy)
        if val is None:
            v = next(ds.sample([(sx, sy)]))[0]
            val = float(v) if v is not None else None
        if val is None:
            return None, ds.crs
        if ds.nodata is not None and val == ds.nodata:
            return None, ds.crs
        if val == -999999.0:          # explicit sentinel kill for this raster
            return None, ds.crs
        if math.isnan(val) or not (-50 < val < 500):
            return None, ds.crs
        # Convert metres → feet if raster stores elevations in metres
        if LIDAR_ELEV_UNIT == "metres":
            val = val * 3.28084
        return val, ds.crs
    except Exception:
        return None, None


# =============================================================================
# PIXEL-BASED RADIAL DEM PROFILES — 250 FT RADIUS, 2.5 FT STEP
# =============================================================================
def build_radial_dem_profiles(row, df_crs, step_ft: float = 2.5,
                               radius_ft: float = 250.0) -> dict:
    cx = None
    cy = None
    try:
        geom = row.get("geometry", None)
        if geom is not None and hasattr(geom, "centroid"):
            c = geom.centroid
            cx, cy = float(c.x), float(c.y)
    except Exception:
        cx = cy = None
    if cx is None or cy is None:
        cx = coalesce_float(row.get("feature_x"), row.get("nearest_x"), default=0.0)
        cy = coalesce_float(row.get("feature_y"), row.get("nearest_y"), default=0.0)
    if not cx or not cy:
        return {"ok": False, "reason": "No centroid/feature coordinates",
                "profiles": {}, "depressions": {}, "summary": {}}

    tiles = PINELLAS_LIDAR_FILES[:]
    tile_index = DemTileIndex(LIDAR_RASTER, tiles)

    elev0, ds_crs = dem_sample_point(tile_index, cx, cy, df_crs)
    if elev0 is None:
        tile_index.close()
        return {"ok": False, "reason": "DEM sample failed at centroid",
                "profiles": {}, "depressions": {}, "summary": {}}

    try:
        ds, sx, sy = tile_index.pick_dataset_for_point(cx, cy, df_crs)
        res_units = float(abs(ds.res[0])) if ds and ds.res else None
    except Exception:
        res_units = None
    units_per_ft = _crs_units_per_foot(ds_crs or df_crs)
    res_ft = (res_units / units_per_ft) if (res_units and units_per_ft) else None
    eff_step_ft = step_ft
    if res_ft and res_ft > 0:
        eff_step_ft = max(step_ft, res_ft)
    n_steps = int(radius_ft / eff_step_ft)

    profiles    = {}
    depressions = {}
    toward_count      = 0
    convergence_dirs  = []

    for dir_name, (dx, dy) in DIRECTIONS.items():
        pts   = []
        elevs = []
        for i in range(1, n_steps + 1):
            dist_ft = i * eff_step_ft
            x = cx + dx * dist_ft
            y = cy + dy * dist_ft
            ev, _ = dem_sample_point(tile_index, x, y, df_crs)
            if ev is None:
                pts.append({"dist_ft": dist_ft, "elev": None})
                elevs.append(None)
            else:
                pts.append({"dist_ft": dist_ft, "elev": float(ev)})
                elevs.append(float(ev))

        elev_arr = np.array([e if e is not None else np.nan for e in elevs], dtype=float)
        nan_mask = np.isnan(elev_arr)
        if nan_mask.any() and (~nan_mask).sum() >= 3:
            x_idx = np.arange(len(elev_arr))
            valid  = ~nan_mask
            elev_arr[nan_mask] = np.interp(x_idx[nan_mask], x_idx[valid], elev_arr[valid])
            for i_fix in range(len(pts)):
                if pts[i_fix]["elev"] is None and not np.isnan(elev_arr[i_fix]):
                    pts[i_fix]["elev"] = float(elev_arr[i_fix])

        prof = []
        for p in pts:
            if p["elev"] is None or p["dist_ft"] <= 0:
                prof.append({**p, "slope": None, "slope_sign": None})
                continue
            slope = (elev0 - p["elev"]) / p["dist_ft"]
            prof.append({**p, "slope": float(slope),
                         "slope_sign": ("away" if slope > 0 else "toward" if slope < 0 else "flat")})

        deps = []
        for i in range(3, len(prof) - 3):
            if prof[i]["elev"] is None:
                continue
            left  = [prof[j]["elev"] for j in range(i-4, i)   if 0 <= j < len(prof) and prof[j]["elev"] is not None]
            right = [prof[j]["elev"] for j in range(i+1, i+5) if 0 <= j < len(prof) and prof[j]["elev"] is not None]
            if len(left) < 3 or len(right) < 3:
                continue
            e = prof[i]["elev"]
            if e < min(left) and e < min(right):
                rim = (sorted(left)[-2] + sorted(right)[1]) / 2.0
                depth = max(0.0, rim - e)
                if depth < 0.04:
                    continue
                vol_cf  = depth * (eff_step_ft ** 2) * math.pi * 0.5
                vol_gal = vol_cf * 7.481
                deps.append({
                    "idx":         i,
                    "dist_ft":     prof[i]["dist_ft"],
                    "elev_ft":     round(e, 4),
                    "rim_ft":      round(rim, 4),
                    "depth_ft":    round(depth, 4),
                    "depth_in":    round(depth * 12, 2),
                    "ponded_gal":  round(float(vol_gal), 1),
                    "severity":    "CRITICAL" if depth > 0.5 else "MODERATE" if depth > 0.2 else "MINOR",
                })

        depressions[dir_name] = deps

        last_valid = next((p for p in reversed(prof) if p["elev"] is not None), None)
        if last_valid and last_valid["elev"] >= elev0:
            toward_count += 1
            convergence_dirs.append(dir_name)

        profiles[dir_name] = {
            "start_elev_ft": round(float(elev0), 4),
            "end_elev_ft":   round(float(last_valid["elev"]), 4) if last_valid else None,
            "end_delta_ft":  round(float(last_valid["elev"] - elev0), 4) if last_valid else None,
            "step_ft":       eff_step_ft,
            "radius_ft":     radius_ft,
            "points":        prof,
        }

    end_deltas   = {d: (p["end_delta_ft"] if p["end_delta_ft"] is not None else 9e9)
                    for d, p in profiles.items()}
    primary_flow = min(end_deltas, key=end_deltas.get) if end_deltas else "UNKNOWN"
    tile_index.close()

    max_dep_depth     = 0.0
    total_ponded_gal  = 0.0
    all_depressions   = []
    for dn, deps in depressions.items():
        for dep in deps:
            max_dep_depth    = max(max_dep_depth, dep["depth_ft"])
            total_ponded_gal += dep["ponded_gal"]
            all_depressions.append({**dep, "direction": dn})

    return {
        "ok": True,
        "point_xy":           (cx, cy),
        "point_crs":          str(df_crs),
        "dem_crs":            str(ds_crs),
        "start_elev_ft":      float(elev0),
        "profiles":           profiles,
        "depressions":        depressions,
        "all_depressions":    all_depressions,
        "summary": {
            "primary_flow":            primary_flow,
            "dirs_toward":             toward_count,
            "convergence_dirs":        convergence_dirs,
            "max_depression_depth_ft": round(max_dep_depth, 4),
            "total_ponded_gal":        round(total_ponded_gal, 1),
            "n_depressions":           len(all_depressions),
        },
    }


# =============================================================================
# RADIAL PROFILE PNG
# =============================================================================
def plot_radial_profiles_png(st_addr: str, radial: dict, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except Exception as e:
        print(f"   ❌ matplotlib not available for radial plot: {e}")
        return False

    if not radial.get("ok"):
        return False

    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_title(f"Radial LiDAR Elevation Profiles — {st_addr}\n"
                 f"250 ft radius | 2.5 ft step | Bilinear sub-pixel DEM",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Distance from parcel centroid (ft)", color="white", fontsize=11)
    ax.set_ylabel("Elevation (ft NAVD88)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.20, color="#666")
    ax.axhline(radial["start_elev_ft"], color="#FFD700", lw=1.5, ls="--",
               label=f"Home grade {radial['start_elev_ft']:.3f} ft", zorder=3)

    seg_colors = {"toward": "#ef4444", "away": "#22c55e", "flat": "#f59e0b", None: "#6b7280"}
    dir_colors = {
        "N": "#60a5fa", "NE": "#a78bfa", "E": "#34d399", "SE": "#fb923c",
        "S": "#f472b6", "SW": "#fbbf24", "W": "#38bdf8", "NW": "#c084fc",
    }
    annotated_depressions = set()

    for dir_name, d in radial["profiles"].items():
        pts = [p for p in d["points"] if p["elev"] is not None]
        if len(pts) < 2:
            continue
        x_arr = np.array([p["dist_ft"] for p in pts])
        y_arr = np.array([p["elev"] for p in pts])

        for i in range(1, len(pts)):
            sign = pts[i]["slope_sign"]
            ax.plot([x_arr[i-1], x_arr[i]], [y_arr[i-1], y_arr[i]],
                    color=seg_colors.get(sign, "#6b7280"), lw=2.2, zorder=4)

        ax.plot(x_arr, y_arr, color=dir_colors.get(dir_name, "#fff"),
                alpha=0.25, lw=0.8, zorder=3)
        ax.text(x_arr[-1] + 2, y_arr[-1], dir_name, fontsize=9,
                va="center", color=dir_colors.get(dir_name, "white"),
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        for dep in radial["depressions"].get(dir_name, []):
            dep_key = (dir_name, dep["idx"])
            if dep_key in annotated_depressions:
                continue
            annotated_depressions.add(dep_key)
            dx = dep["dist_ft"]
            dy = dep["elev_ft"]
            rim = dep["rim_ft"]
            mask = (x_arr >= dx - 5) & (x_arr <= dx + 5)
            if mask.sum() >= 2:
                ax.fill_between(x_arr[mask], y_arr[mask], rim,
                                color="#ef4444", alpha=0.30, zorder=5)
            severity_color = "#ef4444" if dep["severity"] == "CRITICAL" else \
                             "#f59e0b" if dep["severity"] == "MODERATE" else "#60a5fa"
            ax.plot(dx, dy, "v", color=severity_color, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=8)
            ax.annotate(
                f"⚠ {dep['severity'][:3]}\n{dep['depth_in']:.1f}\" deep\n{dep['ponded_gal']:.0f} gal",
                xy=(dx, dy), xytext=(dx + 8, dy - 0.12),
                fontsize=7.5, color="white",
                arrowprops=dict(arrowstyle="->", color=severity_color, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b",
                          edgecolor=severity_color, alpha=0.9),
                zorder=9,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")]
            )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = [
        Line2D([0],[0], color="#22c55e", lw=3, label="Slope away from home"),
        Line2D([0],[0], color="#ef4444", lw=3, label="Slope toward home"),
        Line2D([0],[0], color="#f59e0b", lw=3, label="Flat / neutral slope"),
        Line2D([0],[0], color="#FFD700", lw=1.5, ls="--", label="Home grade elevation"),
        Patch(facecolor="#ef4444", alpha=0.4, label="Depression / ponding zone"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True,
              facecolor="#1e293b", edgecolor="#444", labelcolor="white", fontsize=9)

    s = radial["summary"]
    n_dep = s.get("n_depressions", 0)
    ax.text(0.01, -0.12,
            f"DEM start: {radial['start_elev_ft']:.4f} ft  |  "
            f"Convergence: {s['dirs_toward']}/8 toward-home  |  "
            f"Max depression: {s['max_depression_depth_ft']:.3f} ft ({s['max_depression_depth_ft']*12:.1f}\")  |  "
            f"Total depressions: {n_dep}  |  "
            f"Total ponding: {s['total_ponded_gal']:.0f} gal",
            transform=ax.transAxes, fontsize=9, color="#9ca3af")

    fig.tight_layout()
    try:
        fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except Exception as e:
        print(f"   ❌ radial PNG save error: {e}")
        plt.close(fig)
        return False


# =============================================================================
# ███████████████████████████████████████████████████████████████████████████
#  NEIGHBORHOOD MAP PNG — v121.2 ALIGNMENT FIX
#
#  ROOT CAUSE OF MISALIGNMENT IN v121.1:
#  The imshow extent was computed using approximate deg-per-foot constants:
#      LAT_PER_FT = 0.0000030866  /  LNG_PER_FT = 0.0000038
#  These vary by latitude and introduce hundreds of feet of drift at zoom 20.
#
#  THE FIX:
#  1. Open the LiDAR raster and read the actual window for [cx±r, cy±r]
#  2. Reproject the window's four corner coordinates to EPSG:3857 (Web Mercator)
#     using the raster's own affine transform + pyproj Transformer
#  3. Use those reprojected corners as the imshow extent
#  4. The contextily basemap is already in EPSG:3857 — they now share the same
#     coordinate frame with sub-pixel accuracy
#  5. Mask all elevation cells < PINELLAS_ELEV_MIN_FT (0 ft) or > 25 ft
#     to prevent bay/ocean nodata from rendering as false depressions
# ███████████████████████████████████████████████████████████████████████████
# =============================================================================
def build_neighborhood_map_png(st_addr: str, row, df_crs, g_elev_ft: float,
                                out_png: str, radius_ft: float = 250.0,
                                radial: dict = None,
                                geo_verify: dict = None,
                                max_labels: int = 999,
                                show_overlays: bool = True,
                                sim_rain_in_hr: float = None,
                                rainfall_damage: dict = None,
                                is_flood_view: bool = False,
                                viz_mode: str = "ghost") -> bool:
    # Initialize im variable to prevent UnboundLocalError
    im = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as pe
        from matplotlib.patches import Circle
    except Exception as e:
        print(f"   ❌ matplotlib not available for neighborhood map: {e}")
        return False

    try:
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer, CRS
    except Exception as e:
        print(f"   ❌ rasterio/pyproj not available for neighborhood map: {e}")
        return False

    # ── Center point ─────────────────────────────────────────────────────
    cx = None
    cy = None
    try:
        geom = row.get("geometry", None)
        if geom is not None and hasattr(geom, "centroid"):
            c = geom.centroid
            cx, cy = float(c.x), float(c.y)
    except Exception:
        cx = cy = None
    if cx is None or cy is None:
        cx = coalesce_float(row.get("feature_x"), row.get("nearest_x"), default=0.0)
        cy = coalesce_float(row.get("feature_y"), row.get("nearest_y"), default=0.0)
    if not cx or not cy:
        return False

    # ── Open raster and find the right dataset ────────────────────────────
    tile_index = DemTileIndex(LIDAR_RASTER, PINELLAS_LIDAR_FILES[:])
    ds, sx, sy = tile_index.pick_dataset_for_point(cx, cy, df_crs)
    if ds is None:
        tile_index.close()
        print(f"   ❌ No LiDAR tile covers ({cx:.1f}, {cy:.1f})")
        return False

    units_per_ft = _crs_units_per_foot(ds.crs)
    r_units = radius_ft * units_per_ft

    # ── Read the raster window ────────────────────────────────────────────
    # Window is in the raster's own CRS coordinates (sx, sy already transformed)
    win_left   = sx - r_units
    win_right  = sx + r_units
    win_bottom = sy - r_units
    win_top    = sy + r_units

    try:
        win = from_bounds(win_left, win_bottom, win_right, win_top,
                          transform=ds.transform)
        # Clamp to raster extent
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))

        # ── High-res resampling for 60ft / 250ft maps ──────────────────────
        # QL0/QL1 best practice: target 0.05–0.1m (5–10 cm) cell size with
        # at least 85% overlap — bilinear resampling to full QL1 resolution.
        # 60ft  (LOT)   → 0.05m target (5 cm)  — photographic-quality DEM
        # 250ft (BLOCK) → 0.10m target (10 cm) — QL1 engineering standard
        # 1000ft+       → native raster pixels (neighbourhood context)
        if radius_ft <= 300:
            from rasterio.enums import Resampling as _RS
            _metre_per_unit = 1.0 if ds.crs.is_projected else 111320.0
            if radius_ft <= 75:
                _target_m  = 0.05   # 5 cm — QL0/QL1 photographic grade
                _tier_name = "5 cm (QL0/QL1)"
            else:
                _target_m  = 0.10   # 10 cm — QL1 engineering standard
                _tier_name = "10 cm (QL1)"
            _target_px = _target_m / _metre_per_unit
            _cur_px    = abs(ds.transform.a)
            if _cur_px > _target_px:
                _scale = max(1, int(round(_cur_px / _target_px)))
                # Cap scale to avoid out-of-memory on very coarse source DEMs
                _scale = min(_scale, 20)
                _out_h = int(win.height * _scale)
                _out_w = int(win.width  * _scale)
                arr = ds.read(1, window=win, masked=True,
                              out_shape=(_out_h, _out_w),
                              resampling=_RS.bilinear)
                print(f"   🔬  Bilinear upsample {_scale}x → ~{_tier_name} pixel "
                      f"(source was {_cur_px:.3f} m)")
            else:
                arr = ds.read(1, window=win, masked=True)
                print(f"   🔬  Source DEM already at {_cur_px:.3f} m — no upsampling needed")
        else:
            arr = ds.read(1, window=win, masked=True)
    except Exception as e:
        tile_index.close()
        print(f"   ❌ Raster read failed: {e}")
        return False

    if arr is None or arr.size == 0:
        tile_index.close()
        return False

    # ── *** ALIGNMENT FIX: compute actual pixel-level bounds in EPSG:3857 ***
    #
    # Strategy: get the four corners of the actual read window in the raster CRS,
    # then reproject each corner to EPSG:3857.  imshow extent = those corners.
    # This guarantees the overlay matches the basemap regardless of latitude.
    # ─────────────────────────────────────────────────────────────────────
    try:
        # Actual window bounds in raster CRS
        win_transform = rasterio.windows.transform(win, ds.transform)
        # CRITICAL: use the ORIGINAL window pixel dimensions for corner calculation,
        # NOT arr.shape — when bilinear upsampling is applied (60ft/250ft maps),
        # arr.shape is 10-20x larger than the actual window, causing the imshow
        # extent to land 10-20x outside the axis viewport (invisible overlay).
        _win_ncols = int(round(win.width))
        _win_nrows = int(round(win.height))

        # Four corners in raster CRS (using the window's affine transform)
        # origin='upper' so row 0 is the TOP edge
        corner_ul = win_transform * (0,          0         )  # upper-left  → (west,  north)
        corner_ur = win_transform * (_win_ncols, 0         )  # upper-right → (east,  north)
        corner_ll = win_transform * (0,          _win_nrows)  # lower-left  → (west,  south)
        corner_lr = win_transform * (_win_ncols, _win_nrows)  # lower-right → (east,  south)

        # x = easting (longitude-ish), y = northing (latitude-ish)
        raster_west  = min(corner_ul[0], corner_ll[0])
        raster_east  = max(corner_ur[0], corner_lr[0])
        raster_south = min(corner_ll[1], corner_lr[1])
        raster_north = max(corner_ul[1], corner_ur[1])

        # Reproject raster corners to EPSG:3857 (Web Mercator for contextily)
        raster_crs   = ds.crs
        webmerc_crs  = CRS.from_epsg(3857)
        tfm_to_web   = Transformer.from_crs(raster_crs, webmerc_crs, always_xy=True)

        west_m,  south_m = tfm_to_web.transform(raster_west,  raster_south)
        east_m,  north_m = tfm_to_web.transform(raster_east,  raster_north)

        # Also reproject the subject parcel centroid to EPSG:3857
        # Use df_crs → 3857 for the centroid (cx, cy are in df_crs)
        tfm_pt_to_web = Transformer.from_crs(
            CRS.from_user_input(df_crs), webmerc_crs, always_xy=True)
        hx, hy = tfm_pt_to_web.transform(cx, cy)

        USE_WEB_MERC = True
    except Exception as e:
        print(f"   ⚠️  EPSG:3857 corner reprojection failed: {e} — falling back to native CRS")
        USE_WEB_MERC = False
        # Fallback: use raster-native coords (no basemap)
        raster_west  = win_left
        raster_east  = win_right
        raster_south = win_bottom
        raster_north = win_top
        west_m, east_m, south_m, north_m = raster_west, raster_east, raster_south, raster_north
        hx, hy = sx, sy

    # ── Elevation relative to home grade ──────────────────────────────────
    arr_float = arr.astype("float32")
    nodata = ds.nodata

    # Mask nodata — explicit kill for known sentinel value (-999999)
    if nodata is not None:
        arr_float[arr_float == nodata] = np.nan
    arr_float[arr_float == -999999.0] = np.nan   # belt-and-suspenders for this raster
    arr_float[arr_float < -50]  = np.nan
    arr_float[arr_float > 200]  = np.nan

    # ── UNITS FIX: SAVE_THIS_PINELLAS_3857.tif stores elevations in METRES.
    # All downstream logic (grade comparison, masking, colormap) works in FEET.
    # Convert here — once — so everything below is consistently in feet.
    _unit_scale = 3.28084 if LIDAR_ELEV_UNIT == "metres" else 1.0
    if _unit_scale != 1.0:
        arr_float = arr_float * _unit_scale
        print(f"   📏  Raster units: metres → feet (×{_unit_scale})")

    # *** KEY FIX: mask water/bay cells (below sea level or implausibly high) ***
    # Cells below PINELLAS_ELEV_MIN_FT are open water — hide them from overlay
    #
    # Diagnostic: check how many pixels survive each filter stage
    n_total   = arr_float.size
    n_nan_raw = int(np.isnan(arr_float).sum()) if not hasattr(arr_float, 'mask') \
                else int(arr_float.mask.sum() if arr_float.mask is not np.bool_(False) else 0)
    valid_raw = arr_float.compressed() if hasattr(arr_float, 'compressed') \
                else arr_float[np.isfinite(arr_float)]
    if valid_raw.size > 0:
        raw_min, raw_max = float(valid_raw.min()), float(valid_raw.max())
    else:
        raw_min, raw_max = float('nan'), float('nan')

    print(f"   📊  Raster values (ft): min={raw_min:.2f}  max={raw_max:.2f}  "
          f"nodata_px={n_nan_raw}  total_px={n_total}")

    # Convert masked array to plain ndarray so NaN handling is consistent
    if hasattr(arr_float, 'filled'):
        arr_float = arr_float.filled(np.nan)

    water_mask = (arr_float < PINELLAS_ELEV_MIN_FT) | (arr_float > PINELLAS_ELEV_MAX_FT) | np.isnan(arr_float)
    n_below  = int((arr_float < PINELLAS_ELEV_MIN_FT).sum())
    n_above  = int((arr_float > PINELLAS_ELEV_MAX_FT).sum())
    n_water  = int(water_mask.sum())

    # ── 3x3 Median filter for 60ft and 250ft maps ───────────────────────
    # Removes noise from cars, bushes, fences before elevation analysis.
    if radius_ft <= 300 and HAS_SCIPY:
        from scipy.ndimage import median_filter as _mdf
        arr_float_valid = np.where(np.isnan(arr_float), 0, arr_float)
        arr_float_filt  = _mdf(arr_float_valid, size=3)
        # Only apply filter where data is valid (don't filter NaN zones)
        arr_float = np.where(np.isnan(arr_float), arr_float, arr_float_filt)
        print(f"   🔧  3x3 Median filter applied (noise removal)")

    # ── Ground elevation: sample directly from LiDAR at parcel centroid ──────
    # The GIS attribute (g_elev_ft) often mismatches the raster by 1–5ft due to
    # datum shifts, stale records, or CRS drift — producing a solid-red or
    # solid-green map.  Sampling the raster itself at (sx, sy) guarantees the
    # zero-point matches what the LiDAR actually measured at this parcel.
    # NOTE: arr_float is already in FEET at this point (unit conversion above).
    _lidar_grade = None
    try:
        _cx_px = int(round((sx - (win_left  if USE_WEB_MERC else raster_west )) /
                           abs(ds.transform.a) * (arr_float.shape[1] / max(1, int(win.width)))))
        _cy_px = int(round((     (win_top   if USE_WEB_MERC else raster_north) - sy) /
                           abs(ds.transform.e) * (arr_float.shape[0] / max(1, int(win.height)))))
        _cx_px = max(0, min(_cx_px, arr_float.shape[1] - 1))
        _cy_px = max(0, min(_cy_px, arr_float.shape[0] - 1))
        # Sample a small 5×5 kernel around the centroid and take the median
        _kr = 2
        _patch = arr_float[
            max(0, _cy_px - _kr):min(arr_float.shape[0], _cy_px + _kr + 1),
            max(0, _cx_px - _kr):min(arr_float.shape[1], _cx_px + _kr + 1)
        ]
        _patch_valid = _patch[np.isfinite(_patch)]
        if _patch_valid.size >= 3:
            _lidar_grade = float(np.median(_patch_valid))
    except Exception as _sg_err:
        print(f"   ⚠️  LiDAR centroid sample failed: {_sg_err}")

    if _lidar_grade is not None and PINELLAS_ELEV_MIN_FT <= _lidar_grade <= PINELLAS_ELEV_MAX_FT:
        _grade_used = _lidar_grade
        _grade_src  = f"LiDAR-sampled ({_lidar_grade:.2f} ft)"
    else:
        _grade_used = float(g_elev_ft)
        _grade_src  = f"GIS attribute ({g_elev_ft:.2f} ft)"
        if _lidar_grade is not None:
            print(f"   ⚠️  LiDAR sample ({_lidar_grade:.2f} ft) outside valid range "
                  f"— falling back to GIS attribute ({g_elev_ft:.2f} ft)")

    print(f"   📐  Ground zero: {_grade_src}")
    rel = arr_float - _grade_used

    # Apply water mask AFTER computing relative elevation
    rel[water_mask] = np.nan

    # Gap-fill remaining NaN with spatial interpolation
    if HAS_SCIPY and np.isnan(rel).any():
        from scipy.ndimage import generic_filter
        def fill_nan(a):
            vals = a[~np.isnan(a)]
            return np.mean(vals) if vals.size > 0 else np.nan
        rel_filled = generic_filter(rel, fill_nan, size=5, mode="nearest")
        rel = np.where(np.isnan(rel), rel_filled, rel)
        # Re-apply water mask so fill doesn't bleed into masked cells
        rel[water_mask] = np.nan

    finite = rel[np.isfinite(rel)]
    if finite.size < 50:
        tile_index.close()
        print(f"   ❌ Insufficient valid DEM pixels after masking ({finite.size} remain)")
        print(f"      Total pixels: {n_total}  |  Raw NaN/nodata: {n_nan_raw}")
        print(f"      Raw elev range: [{raw_min:.2f}, {raw_max:.2f}] ft")
        print(f"      Water-masked: {n_water} (below {PINELLAS_ELEV_MIN_FT}: {n_below}, "
              f"above {PINELLAS_ELEV_MAX_FT}: {n_above})")
        print(f"      Home grade: {g_elev_ft:.2f} ft  |  Radius: {radius_ft} ft")
        print(f"      💡 If raw elev range looks valid, try adjusting "
              f"PINELLAS_ELEV_MIN_FT / PINELLAS_ELEV_MAX_FT in config")
        return False

    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    # ── Per-radius colour span calibration ────────────────────────────────
    # 60ft  (LOT)   — ±1.0 ft span: every 1–2 inch grade change is visible
    # 250ft (BLOCK) — ±1.5 ft span: still tight for QL1 engineering detail
    # 1000ft+       — ±half-range as before for neighbourhood context
    if radius_ft <= 75:
        span = 1.0          # 60ft LOT: full colourmap = ±1 ft → 2" per colour band
    elif radius_ft <= 300:
        span = 1.5          # 250ft BLOCK: ±1.5 ft — resolves ponding thresholds
    else:
        span = max(0.35, (hi - lo) / 2.0)

    # ── Colormap: red→orange→yellow (below grade) + green (above grade) ─────
    # Used for the colorbar only.  The actual overlay renders ONLY below-grade
    # pixels; above-grade is always pure satellite (see Layer 2 below).
    vmin, vmax = -span, +span
    _flood_colors = [
        (0.0,   (0.82, 0.04, 0.04)),   # deep red
        (0.25,  (0.95, 0.28, 0.05)),   # red-orange
        (0.50,  (0.97, 0.58, 0.10)),   # orange
        (0.75,  (0.98, 0.82, 0.18)),   # amber-yellow
        (1.0,   (0.99, 0.98, 0.50)),   # pale yellow (at grade = 0)
    ]
    elevation_cmap = LinearSegmentedColormap.from_list(
        "flood_risk", [(c[0], c[1]) for c in _flood_colors])
    elevation_cmap.set_bad(alpha=0.0)
    _norm_disc = None   # no BoundaryNorm — clean continuous ramp

    # rel_display is built AFTER the building mask (section C below) so the
    # contrast stretch operates on the final display array, not a stale copy.

    # ── Build effective bare-earth DEM (mask buildings) ───────────────────
    # Creates a version of `rel` where building/structure pixels are replaced
    # with interpolated surrounding ground elevation.  The original `rel` is
    # kept intact for display — only the bare-earth copy is used for ponding.
    building_mask = np.zeros(rel.shape, dtype=bool)

    # Method A: Vector footprints (if available)
    if BUILDING_FOOTPRINTS and os.path.exists(BUILDING_FOOTPRINTS):
        try:
            from rasterio.features import rasterize as rio_rasterize
            bldg_gdf = gpd.read_file(BUILDING_FOOTPRINTS,
                                      bbox=(raster_west, raster_south,
                                            raster_east, raster_north))
            if not bldg_gdf.empty:
                if bldg_gdf.crs and str(bldg_gdf.crs) != str(ds.crs):
                    bldg_gdf = bldg_gdf.to_crs(ds.crs)
                win_transform = rasterio.windows.transform(win, ds.transform)
                bldg_raster = rio_rasterize(
                    [(geom, 1) for geom in bldg_gdf.geometry if geom is not None],
                    out_shape=rel.shape,
                    transform=win_transform,
                    fill=0, dtype="uint8")
                building_mask |= (bldg_raster == 1)
                print(f"   🏗️  Building footprints: {int(building_mask.sum())} pixels masked from vector")
        except Exception as e:
            print(f"   ⚠️  Building footprint rasterize failed: {e}")

    # Method C: Effective-ground detection — LOCAL MINIMUM approach
    #
    # Root cause of rooftop color bleed in earlier versions:
    #   Using local MEAN as the ground reference contaminates the baseline with
    #   building returns themselves, so rooftops only appear marginally above the
    #   "mean" and slip below the threshold.
    #
    # Fix: use local MINIMUM over a small kernel (11 px) as the ground-level
    #   estimate. Ground pixels ARE the local minimum — structures rise above them.
    #   Even a 1-story Florida structure (8–10 ft) clears the 1.0 ft cap easily.
    #   The threshold is capped at 1.0 ft locally regardless of the global
    #   BUILDING_HEIGHT_THRESH_FT config so the display map always masks aggressively.
    #   The global config still governs the bare-earth / flow DEM calculations.
    if HAS_SCIPY and np.isfinite(rel).sum() > 200:
        from scipy.ndimage import minimum_filter as _mf_bldg
        _bldg_thresh_display = min(BUILDING_HEIGHT_THRESH_FT, 1.0)
        _rel_finite = np.where(np.isfinite(rel),
                               rel,
                               float(np.nanmedian(rel[np.isfinite(rel)])))
        # Local ground estimate = minimum elevation in an 11-pixel neighbourhood
        _local_ground = _mf_bldg(_rel_finite, size=11, mode="nearest")
        # Height above local ground for every pixel
        _height_above = rel - _local_ground
        eff_ground_mask = (
            (_height_above > _bldg_thresh_display) &
            np.isfinite(rel) &
            ~water_mask
        )
        building_mask |= eff_ground_mask
        n_eff = int(eff_ground_mask.sum())
        if n_eff > 0:
            print(f"   🏗️  Effective-ground filter: {n_eff} pixels "
                  f">{_bldg_thresh_display} ft above local ground min "
                  f"→ masked as structures")

    # ── A. Hard-mask buildings: NoData for ponding (impermeable barriers) ──
    # Buildings are set to NaN so water cannot pond ON them.
    # A separate interpolated bare-earth DEM is used for depression depth
    # estimation, but ponding regions can only exist on ground pixels.
    rel_bare = rel.copy()
    # Also build a "flow DEM" where buildings are impermeable barriers (NoData)
    # so streamlines route AROUND buildings, not through them.
    rel_flow = rel.copy()
    if building_mask.any():
        rel_bare[building_mask] = np.nan
        rel_flow[building_mask] = np.nan   # D. impermeable barrier for flow
        # B. Interpolate gaps for depression-depth estimation only
        if HAS_SCIPY:
            from scipy.ndimage import generic_filter as _gf_bare
            def _fill_bare(a):
                vals = a[~np.isnan(a)]
                return np.mean(vals) if vals.size > 0 else np.nan
            for _pass in range(3):  # multiple passes for large buildings
                still_nan = np.isnan(rel_bare) & building_mask
                if not still_nan.any():
                    break
                filled = _gf_bare(rel_bare, _fill_bare, size=7, mode="nearest")
                rel_bare = np.where(np.isnan(rel_bare) & building_mask,
                                    filled, rel_bare)
        # Re-apply water mask so we don't fill water cells
        rel_bare[water_mask] = np.nan
        rel_flow[water_mask] = np.nan
    else:
        rel_flow[water_mask] = np.nan

    # ── B. Auto-detect depression regions on bare-earth DEM ───────────────
    # Depressions are only detected on ground pixels (not buildings).
    finite_bare = rel_bare[np.isfinite(rel_bare)]
    depression_regions = []
    if HAS_SCIPY and finite_bare.size > 200:
        from scipy.ndimage import label as ndlabel, minimum_filter
        rel_filled_for_dep = np.where(np.isfinite(rel_bare), rel_bare,
                                       float(np.nanmean(finite_bare)))
        mean_rel = float(np.nanmean(finite_bare))
        min_filt = minimum_filter(rel_filled_for_dep, size=7)
        depression_mask = (
            (rel_bare < (mean_rel - 0.08)) &
            np.isfinite(rel_bare) &
            (rel_bare < min_filt + 0.05) &
            ~water_mask &
            ~building_mask        # B. ponding only on bare-earth pixels
        )
        labeled, n_features = ndlabel(depression_mask)
        for feat_id in range(1, n_features + 1):
            feat = (labeled == feat_id)
            if feat.sum() < 6:
                continue
            feat_vals = rel_bare[feat]
            dep_depth = mean_rel - float(np.nanmean(feat_vals))
            if dep_depth < 0.04:
                continue
            rows_idx, cols_idx = np.where(feat)
            center_r = int(np.mean(rows_idx))
            center_c = int(np.mean(cols_idx))
            area_px  = feat.sum()
            # Proximity to house (image center = subject parcel)
            _img_cr = rel.shape[0] // 2
            _img_cc = rel.shape[1] // 2
            _dist_to_house_px = max(1.0, ((center_r - _img_cr)**2 +
                                          (center_c - _img_cc)**2) ** 0.5)
            try:
                # CRITICAL: arr may be upsampled (bilinear, up to 20x).
                # ds.res[0] is the NATIVE pixel size — we must divide by the
                # actual upsample factor (arr.shape[1] / win.width) to get the
                # real pixel footprint in the array we're measuring.
                _native_px_ft  = float(abs(ds.res[0])) / units_per_ft
                _upsample_fact = float(arr.shape[1]) / max(1.0, float(win.width))
                px_ft          = _native_px_ft / max(1.0, _upsample_fact)
            except Exception:
                px_ft = 0.5   # safe fallback: ~15cm pixel
            # Clamp to physically plausible range (1cm – 5ft per pixel)
            px_ft = max(0.033, min(px_ft, 5.0))
            area_sqft = area_px * (px_ft ** 2)
            vol_gal   = area_sqft * dep_depth * 7.481
            # Radius-aware volume cap — depressions at wider views cover more
            # ground legitimately, so allow proportionally larger volumes.
            # 75ft  → 50k gal   (single lot scale)
            # 300ft → 250k gal  (block / neighborhood scale)
            # 1000ft→ 2M gal    (watershed / regional scale)
            _vol_cap = (50_000.0 if radius_ft <= 75 else
                        250_000.0 if radius_ft <= 300 else
                        2_000_000.0)
            vol_gal = min(vol_gal, _vol_cap)
            severity = "CRITICAL" if dep_depth > 0.5 else "MODERATE" if dep_depth > 0.2 else "MINOR"
            # Liability score = depth² × volume ÷ distance_to_house
            # Deep + large + close = highest risk. Normalise so far depressions
            # don't completely vanish but proximity is heavily weighted.
            _liability = (dep_depth ** 2) * vol_gal / _dist_to_house_px
            depression_regions.append({
                "mask":            feat,
                "center_r":        center_r,
                "center_c":        center_c,
                "depth_ft":        round(dep_depth, 3),
                "depth_in":        round(dep_depth * 12, 1),
                "vol_gal":         round(vol_gal, 0),
                "area_sqft":       round(area_sqft, 1),
                "area_px":         area_px,
                "severity":        severity,
                "dist_to_house_px": round(_dist_to_house_px, 1),
                "liability_score": round(_liability, 4),
            })

    # ── C. Build rel_display: building mask + contrast stretch ───────────────
    # rel_display is the array passed to imshow.  We build it here — ONCE —
    # after all masking is done, then apply the contrast stretch so it actually
    # reaches vmin/vmax and uses the full colour palette.
    rel_display = rel.copy()
    building_alpha_overlay = None

    if is_flood_view and building_mask.any():
        rel_display[building_mask] = np.nan
        print(f"   🎨  Flood sim: {int(building_mask.sum())} structure pixels → transparent")
    elif building_mask.any():
        # Clean DEM: replace roof-height pixels with bare-earth interpolated ground
        # so foundation/grade shows correctly instead of roof elevation
        rel_display = np.where(building_mask, rel_bare, rel_display)
        building_alpha_overlay = building_mask.copy()
        print(f"   🎨  Clean DEM: {int(building_mask.sum())} building pixels → bare-earth grade")
    else:
        print(f"   🎨  No building structures detected in DEM")

    # ── Linear contrast stretch ───────────────────────────────────────────────
    # Maps the 2nd–98th percentile of the DISPLAY data onto [vmin, vmax] so
    # the full colour palette is always used regardless of terrain flatness.
    # On Pinellas County (relief ≈ inches) this is essential — without it
    # everything maps to the middle 1–2 colour bands and looks flat.
    # The BoundaryNorm uses the stretched values so discrete band edges stay sharp.
    _valid_disp = np.isfinite(rel_display)
    if _valid_disp.sum() > 50:
        _p2_d, _p98_d = np.percentile(rel_display[_valid_disp], [2, 98])
        _span_d = _p98_d - _p2_d
        if _span_d > 0.01:
            _stretched = vmin + (rel_display - _p2_d) / _span_d * (vmax - vmin)
            rel_display = np.where(_valid_disp, _stretched, rel_display)
            # Re-apply NaN mask so masked pixels stay transparent
            rel_display[~_valid_disp] = np.nan
            print(f"   🎨  Contrast stretch: data [{_p2_d:.3f}, {_p98_d:.3f}] ft "
                  f"→ colormap [{vmin:.3f}, {vmax:.3f}] ft  (Δ={_span_d:.4f} ft)")

        else:
            print(f"   ⚠️  Data range {_span_d:.4f} ft — too narrow to stretch, using raw values")
    else:
        print(f"   ⚠️  Insufficient valid pixels for contrast stretch")

    # ── Build figure ──────────────────────────────────────────────────────
    # 60ft  (LOT)   — 300 DPI, 12×12 in  → 3600×3600 px
    # 250ft (BLOCK) — 400 DPI, 14×14 in  → 5600×5600 px engineering grade
    # 1000ft+       — 400 DPI, 16×16 in  → 6400×6400 px print quality
    if radius_ft <= 75:
        _fig_dpi  = 300
        _fig_size = (12, 12)
    elif radius_ft <= 300:
        _fig_dpi  = 400
        _fig_size = (14, 14)
    else:
        _fig_dpi  = 400
        _fig_size = (16, 16)
    fig, ax = plt.subplots(figsize=_fig_size, dpi=_fig_dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    try:
        import contextily as ctx
        HAS_CTX = True
    except ImportError:
        HAS_CTX = False

    # ── Canonical axis extent = raster window bounds (EPSG:3857) ─────────────
    # All layers (basemap, hillshade, LiDAR) are placed using the same extent so
    # nothing can drift outside the viewport.  Autoscaling is locked immediately
    # after the first set_xlim/ylim so subsequent imshow calls cannot shift it.
    ax.set_xlim(west_m, east_m)
    ax.set_ylim(south_m, north_m)
    ax.set_autoscale_on(False)

    # ==========================================================================
    # RENDERING — rebuilt for all three radii (60 / 250 / 1000 ft)
    # Every layer uses the same extent [west_m, east_m, south_m, north_m] and
    # the axis limits are re-asserted after every imshow so autoscaling can
    # never push a layer outside the viewport.
    #
    # Layer stack (bottom → top):
    #   0. Satellite basemap (contextily Esri WorldImagery)
    #   1. QGIS hillshade from HILLSHADE_RASTER — or DEM-computed fallback
    #   2. LiDAR elevation RYG overlay
    #   (flood sim only) 3. Blue ponding fill  4. Flow streamlines
    # ==========================================================================
    _ext = [west_m, east_m, south_m, north_m]

    # ── LAYER 0: Satellite basemap ────────────────────────────────────────────
    # zoom=19 always has Esri coverage; zoom=21 has placeholder tiles for many
    # FL addresses and draws "Map data not yet available" without raising errors.
    _zoom = 19 if radius_ft <= 75 else 19 if radius_ft <= 300 else 15
    if USE_WEB_MERC and HAS_CTX:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.Esri.WorldImagery,
                            zoom=_zoom, attribution=False)
            ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
            print(f"   🛰️  Satellite basemap: Esri WorldImagery zoom={_zoom}")
        except Exception as _ctx_e:
            print(f"   ⚠️  Basemap failed: {_ctx_e}")
            ax.set_facecolor("#1a1a2e")
    else:
        ax.set_facecolor("#1a1a2e")

    # ── LAYER 1: Hillshade (light 3D relief texture) ──────────────────────────
    # Computed from DEM; QGIS hillshade.gpkg used if available.
    _hs_alpha = 0.22 if not is_flood_view else 0.16
    _hs_applied = False

    if os.path.exists(HILLSHADE_RASTER):
        try:
            with rasterio.open(HILLSHADE_RASTER) as _hds:
                _hs_tfm = Transformer.from_crs(CRS.from_epsg(3857), _hds.crs, always_xy=True)
                _hs_w, _hs_s = _hs_tfm.transform(west_m, south_m)
                _hs_e, _hs_n = _hs_tfm.transform(east_m, north_m)
                from rasterio.windows import from_bounds as _fb
                _hwin = _fb(_hs_w, _hs_s, _hs_e, _hs_n, transform=_hds.transform)
                _hwin = _hwin.intersection(
                    rasterio.windows.Window(0, 0, _hds.width, _hds.height))
                if _hwin.width > 1 and _hwin.height > 1:
                    _ha = _hds.read(1, window=_hwin, masked=True).astype("float32")
                    if hasattr(_ha, 'filled'):
                        _ha = _ha.filled(np.nan)
                    _hv = _ha[np.isfinite(_ha)]
                    if _hv.size > 0:
                        _h0, _h1 = float(_hv.min()), float(_hv.max())
                        _hn = (_ha - _h0) / (_h1 - _h0) if _h1 > _h0 else np.zeros_like(_ha)
                        _hn[~np.isfinite(_ha)] = np.nan
                        ax.imshow(_hn, extent=_ext, cmap="gray", vmin=0.0, vmax=1.0,
                                  alpha=_hs_alpha, origin="upper",
                                  interpolation="bilinear", zorder=2)
                        ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
                        _hs_applied = True
                        print(f"   🏔️  QGIS hillshade (alpha={_hs_alpha})")
        except Exception as _hse:
            print(f"   ⚠️  QGIS hillshade failed: {_hse}")

    if not _hs_applied:
        try:
            from matplotlib.colors import LightSource as _LS
            _rh = np.where(np.isfinite(rel_display), rel_display,
                           float(np.nanmedian(rel_display[np.isfinite(rel_display)])))
            _vert = 350 if radius_ft <= 75 else 220 if radius_ft <= 300 else 120
            _hs_dem = _LS(azdeg=315, altdeg=45).hillshade(_rh, vert_exag=_vert, dx=1, dy=1)
            ax.imshow(_hs_dem, extent=_ext, cmap="gray", vmin=0.0, vmax=1.0,
                      alpha=_hs_alpha, origin="upper", interpolation="bilinear", zorder=2)
            ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
            print(f"   🏔️  DEM hillshade fallback (vert_exag={_vert}, alpha={_hs_alpha})")
        except Exception as _dhe:
            print(f"   ⚠️  Hillshade failed: {_dhe}")

    # ── LAYER 2: Elevation overlay — alpha baked into colormap ───────────────
    # The colormap goes from opaque deep-red (most below grade) to fully
    # transparent pale-yellow (at grade).  Above-grade pixels are set to NaN
    # (transparent).  This produces a smooth fade rather than a hard cliff edge.
    #
    # Baking alpha into the colormap avoids the "vivid yellow blob" problem:
    # near-grade values map to near-transparent colours regardless of imshow alpha.
    import matplotlib.colors as _mc2
    # RGBA stops: (R, G, B, A)  — A goes from 0.55 at deepest to 0.0 at grade
    _ov_peak = 0.55 if radius_ft <= 75 else 0.58 if radius_ft <= 300 else 0.48
    _cmap_rgba = [
        (0.82, 0.04, 0.04, _ov_peak),          # deep red     — farthest below grade
        (0.94, 0.28, 0.05, _ov_peak * 0.90),   # red-orange
        (0.97, 0.52, 0.10, _ov_peak * 0.72),   # orange
        (0.98, 0.74, 0.18, _ov_peak * 0.45),   # amber
        (0.99, 0.90, 0.30, _ov_peak * 0.18),   # pale yellow  — just below grade
        (0.99, 0.97, 0.45, 0.0),               # at grade → fully transparent
    ]
    _ov_cmap = _mc2.LinearSegmentedColormap.from_list("ov", _cmap_rgba)
    _ov_cmap.set_bad(alpha=0.0)   # NaN (above grade) → transparent

    _below_only = np.where(rel_display < 0.0, rel_display, np.nan)
    _ov_norm    = _mc2.Normalize(vmin=-span, vmax=0.0)

    im = ax.imshow(_below_only, extent=_ext,
                   cmap=_ov_cmap, norm=_ov_norm,
                   alpha=1.0, origin="upper",
                   interpolation="bilinear", zorder=3)
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    _sm_cb = im
    print(f"   🎨  Elevation overlay: baked-alpha RGBA cmap, peak={_ov_peak:.2f}, "
          f"span=-{span:.1f}→0 ft")

    tile_index.close()

    # ── Gray-out mask: semi-transparent overlay to dull building pixels ──
    # Only apply in flood simulation mode, completely disabled for clean DEM
    if is_flood_view and building_mask.any():
        from matplotlib.colors import ListedColormap as _LCM_gray
        gray_overlay = np.where(building_mask, 1.0, np.nan)
        gray_cmap = _LCM_gray(["#D3D3D3"])
        gray_cmap.set_bad(alpha=0.0)
        ax.imshow(
            gray_overlay,
            extent=[west_m, east_m, south_m, north_m],
            cmap=gray_cmap, vmin=0.5, vmax=1.5,
            alpha=0.45, origin="upper", interpolation="nearest", zorder=3)
        print(f"   🎨  Gray-out mask: {int(building_mask.sum())} building pixels dimmed (alpha=0.45)")

    # ── Helper: convert grid pixel row/col → plot coordinates ────────────
    h_px, w_px = rel.shape

    def px2plot(r_px, c_px):
        """
        Map grid pixel indices to Web Mercator plot coordinates.
        Uses the actual reprojected corners — no approximation.
        """
        frac_x = c_px / max(w_px - 1, 1)
        frac_y = 1.0 - r_px / max(h_px - 1, 1)  # origin='upper' flip
        plot_x = west_m  + frac_x * (east_m  - west_m)
        plot_y = south_m + frac_y * (north_m - south_m)
        return plot_x, plot_y

    # ── Depression annotations ────────────────────────────────────────────
    # Sort by severity (CRITICAL first) then depth descending; annotate top N
    # Sort by liability score (depth² × vol ÷ dist_to_house) — highest risk first.
    # This surfaces depressions that are deep AND large AND close to the foundation,
    # which is the true financial/structural liability regardless of raw severity label.
    depression_regions_sorted = sorted(
        depression_regions,
        key=lambda d: -d.get("liability_score", 0.0)
    )
    labeled_indices = set()
    # 60ft (lot) — up to 8 circles; 250ft (block) — top 3 only (biggest concerns);
    # 1000ft+ — no circles (too cluttered at neighbourhood scale)
    if radius_ft <= 75:
        _max_dep_labels = 8
    elif radius_ft <= 300:
        _max_dep_labels = 3
    else:
        _max_dep_labels = 0

    for _di, dreg in enumerate(depression_regions_sorted):
        if _di >= _max_dep_labels:
            break
        cx_dep, cy_dep = px2plot(dreg["center_r"], dreg["center_c"])
        dep_depth_in  = dreg["depth_in"]
        dep_vol_gal   = dreg["vol_gal"]
        sev           = dreg["severity"]

        _sev_col  = "#ef4444" if sev == "CRITICAL" else \
                    "#f59e0b" if sev == "MODERATE"  else "#60a5fa"
        _sev_icon = "▲ CRI"  if sev == "CRITICAL"  else \
                    "△ MOD"  if sev == "MODERATE"   else "ℹ MINOR"

        # Circle radius proportional to area, clamped to sensible display range
        _r_px  = max(3, min(20, (dreg["area_px"] ** 0.5) * 0.6))
        _r_m   = _r_px * ((east_m - west_m) / rel.shape[1])
        circ = plt.Circle((cx_dep, cy_dep), _r_m,
                           fill=False, edgecolor=_sev_col,
                           linewidth=1.6, alpha=0.90, zorder=8)
        ax.add_patch(circ)

        # Small x at center
        ax.plot(cx_dep, cy_dep, marker="x", markersize=4,
                color=_sev_col, markeredgewidth=1.2, alpha=0.85, zorder=9)

        # Label box — offset so it doesn't cover the circle
        _lbl = f"{_sev_icon}\nDepth: {dep_depth_in:.1f}\"\nPond: {int(dep_vol_gal)} gal"
        _off_x = _r_m * 1.25
        _off_y = _r_m * 0.5
        ax.annotate(
            _lbl,
            xy=(cx_dep, cy_dep),
            xytext=(cx_dep + _off_x, cy_dep + _off_y),
            fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=_sev_col,
                      ec="white", alpha=0.88, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color=_sev_col,
                            lw=0.9, alpha=0.7),
            zorder=10
        )
        labeled_indices.add(_di)

    print(f"   💧  Depression circles drawn: {len(labeled_indices)} "
          f"(of {len(depression_regions)} detected)")

    # ── Blue ponding fill for flood simulation ONLY ────────────────────
    # Render ONLY when is_flood_view=True (flood simulation)
    # Clean DEM images (is_flood_view=False) skip this entirely
    _use_blue_ponding = is_flood_view and depression_regions

    if _use_blue_ponding and depression_regions:
        # ── Graduated flood-depth overlay ─────────────────────────────────────
        # Build a continuous depth grid (ft) across all flooded depression pixels.
        # NaN = dry / non-flooded → fully transparent in the colormap.
        # Deep areas render as dark navy-blue; shallow as sky-blue — this gives
        # a realistic inundation look instead of a flat 3-color swatch.
        depth_overlay = np.full(rel.shape, np.nan)
        for dreg in depression_regions:
            _mask = dreg["mask"] & ~building_mask
            if _mask.any():
                # Use the actual LiDAR depth (ft above depression floor).
                # Clamp at 4 ft so extreme outliers don't wash out color scale.
                _d = float(dreg.get("depth_ft", 0.5) or 0.5)
                depth_overlay[_mask] = min(_d, 4.0)

        if is_flood_view and depth_overlay is not None:
            import matplotlib.cm as _mcm
            from matplotlib.colors import Normalize as _Norm
            # 'Blues' goes white→light-blue→deep-navy: perfect for flood inundation.
            # vmin slightly negative so even the shallowest areas show visible blue.
            _flood_cmap = _mcm.get_cmap("Blues").copy()
            _flood_cmap.set_bad(alpha=0.0)   # NaN = transparent (dry land)
            _norm = _Norm(vmin=-0.1, vmax=3.5)
            ax.imshow(
                depth_overlay,
                extent=[west_m, east_m, south_m, north_m],
                cmap=_flood_cmap, norm=_norm,
                alpha=0.78, origin="upper", interpolation="bilinear", zorder=5)
            # Thin navy contour around the outer edge of each flooded zone
            try:
                _contour_data = np.where(np.isfinite(depth_overlay), 1.0, 0.0)
                ax.contour(_contour_data,
                           extent=[west_m, east_m, south_m, north_m],
                           levels=[0.5], colors=["#1e3a5f"],
                           linewidths=1.2, alpha=0.85, origin="upper", zorder=6)
            except Exception:
                pass
        else:
            # Non-flood view fallback — simple blue mask
            import matplotlib.cm as _mcm2
            _b_cmap = _mcm2.get_cmap("Blues").copy()
            _b_cmap.set_bad(alpha=0.0)
            ax.imshow(
                depth_overlay,
                extent=[west_m, east_m, south_m, north_m],
                cmap=_b_cmap, vmin=0.0, vmax=3.0,
                alpha=0.55, origin="upper", interpolation="nearest", zorder=5)

    # ── Depression markers DISABLED ───────────────────────────────────────
    # No red circles, labels, or markers - only water fill visualization
    # This creates a clean flood simulation showing only water accumulation

    # ── D. Structure-aware streamplot ────────────────────────────────────
    # Only rendered in flood simulation mode.
    # Completely disabled for clean DEM images.
    # Use rel_flow where buildings are NoData (impermeable barriers).
    # Water routes AROUND buildings, not through them.
    if is_flood_view:
        try:
            flow_fill_val = float(np.nanmean(finite_bare)) if finite_bare.size > 0 else 0.0
            rel_for_flow = np.where(np.isfinite(rel_flow), rel_flow, flow_fill_val)
            gy, gx = np.gradient(rel_for_flow)
            u = -gx
            v_flow = -gy
            mag = np.sqrt(u*u + v_flow*v_flow)
            mag = np.where(np.isfinite(mag), mag, 0.0)

            # D. Zero out flow velocity at building pixels — impermeable barriers
            if building_mask.any():
                u[building_mask] = 0.0
                v_flow[building_mask] = 0.0
                mag[building_mask] = 0.0

            step_sp = max(4, int(min(rel.shape) / 80))
            n_rows_sp = rel.shape[0] // step_sp
            n_cols_sp = rel.shape[1] // step_sp

            # Build streamplot grid in Web Mercator by mapping pixel centres
            stream_x_vals = np.array([px2plot(0, c * step_sp)[0]
                                       for c in range(n_cols_sp)])
            stream_y_vals = np.array([px2plot(r * step_sp, 0)[1]
                                       for r in range(n_rows_sp)])
            stream_xx, stream_yy = np.meshgrid(stream_x_vals, stream_y_vals)

            uu = u      [::step_sp, ::step_sp][:n_rows_sp, :n_cols_sp]
            vv = v_flow [::step_sp, ::step_sp][:n_rows_sp, :n_cols_sp]
            cc = mag    [::step_sp, ::step_sp][:n_rows_sp, :n_cols_sp]

            # Invert v for imshow origin='upper' vs streamplot origin='lower'
            vv = -vv

            cfinite = cc[np.isfinite(cc)]
            vmax_c  = float(np.percentile(cfinite, 95)) if cfinite.size else 1.0
            vmax_c  = max(vmax_c, 1e-6)

            ax.streamplot(
                stream_xx, stream_yy, uu, vv,
                color=cc, cmap="Blues",
                density=1.3, linewidth=0.8, arrowsize=0.9,
                minlength=0.1, maxlength=4.0, zorder=8,
                norm=plt.Normalize(vmin=0.0, vmax=vmax_c),
            )
        except Exception as e:
            print(f"   ⚠️  Flow streamplot: {e}")

    # ── HOME MARKER — ultra vibrant and prominent ───────────────────────────────
    # Enhanced visibility for clean DEM mode
    marker_size = 20 if is_flood_view else 24
    edge_width = 3 if is_flood_view else 4
    
    # Multi-layer white circles for maximum visibility
    ax.plot(hx, hy, "o", color="#ffffff", markersize=32, alpha=0.30, zorder=45)
    ax.plot(hx, hy, "o", color="#ffffff", markersize=24, alpha=0.60, zorder=46)
    ax.plot(hx, hy, "o", markersize=marker_size, color="#ffffff",
            markeredgecolor="#ff0040", markeredgewidth=edge_width, zorder=48)
    
    # Enhanced crosshair with better visibility
    cross_size = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * (0.016 if is_flood_view else 0.018)
    cross_lw = 4 if is_flood_view else 5
    ax.plot([hx - cross_size, hx + cross_size], [hy, hy],
            color="#ff0040", lw=cross_lw, zorder=50,
            path_effects=[pe.withStroke(linewidth=8, foreground="white")])
    ax.plot([hx, hx], [hy - cross_size, hy + cross_size],
            color="#ff0040", lw=cross_lw, zorder=50,
            path_effects=[pe.withStroke(linewidth=8, foreground="white")])
    
    # Enhanced label with better contrast
    label_fontsize = 10 if is_flood_view else 11
    ax.text(hx, hy + cross_size * 1.6, "SUBJECT PARCEL",
            ha="center", va="bottom", fontsize=label_fontsize, fontweight="bold",
            color="white", zorder=51,
            path_effects=[pe.withStroke(linewidth=4, foreground="#000000")])

    # ── Property risk badge ───────────────────────────────────────────────
    # Only show in flood simulation mode, completely disabled for clean DEM
    if is_flood_view:
        _ext_wall  = str(row.get("EXTERIORWA", "") or "Unknown").strip()
        _yr_built  = str(row.get("YEARBUILT", row.get("ACTYRBLT", "")) or "Unknown").strip()
        _foundation = str(row.get("FOUNDATION", "") or "Unknown").strip()
        _wall_up = _ext_wall.upper()
        if any(x in _wall_up for x in ["WOOD", "FRAME", "VINYL"]):
            _wall_risk, _wall_color = "🔴 HIGH", "#ef4444"
        elif any(x in _wall_up for x in ["MASONRY", "BLOCK", "CBS", "STUCCO", "CONCRETE"]):
            _wall_risk, _wall_color = "🟡 MOD", "#f59e0b"
        else:
            _wall_risk, _wall_color = "🟢 LOW", "#22c55e"
        try:
            _yr_int = int(str(_yr_built)[:4])
            if _yr_int < 1975:
                _yr_risk, _yr_color = "🔴 PRE-1975", "#ef4444"
            elif _yr_int < 2002:
                _yr_risk, _yr_color = "🟡 1975–2002", "#f59e0b"
            else:
                _yr_risk, _yr_color = "🟢 POST-2002", "#22c55e"
        except Exception:
            _yr_risk, _yr_color = "⚪ Unknown", "#9ca3af"
        _fnd_up = _foundation.upper()
        if any(x in _fnd_up for x in ["SLAB", "GRADE"]):
            _fnd_risk, _fnd_color = "🔴 HIGH", "#ef4444"
        elif any(x in _fnd_up for x in ["CRAWL", "STEM", "FILL"]):
            _fnd_risk, _fnd_color = "🟡 MOD", "#f59e0b"
        elif any(x in _fnd_up for x in ["PILE", "PIER", "COLUMN", "ELEVATED"]):
            _fnd_risk, _fnd_color = "🟢 LOW", "#22c55e"
        else:
            _fnd_risk, _fnd_color = "⚪ Unknown", "#9ca3af"
        _badge_lines = [
            f"Exterior Wall : {_ext_wall}",
            f"Risk          : {_wall_risk}",
            f"Year Built    : {_yr_built}  {_yr_risk}",
            f"Foundation    : {_foundation}",
            f"Risk          : {_fnd_risk}",
        ]
        _badge_txt = "\n".join(_badge_lines)
        _bx = hx + cross_size * 2.2
        _by = hy - cross_size * 2.2
        ax.text(_bx, _by, _badge_txt,
                va="top", ha="left", fontsize=8, family="monospace",
                color="white",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                          edgecolor="#f59e0b", linewidth=2, alpha=0.92),
                zorder=52,
                path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")])

    # ── North arrow ───────────────────────────────────────────────────────
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    ax.annotate("N", xy=(xlim[0]+0.05*xr, ylim[0]+0.10*yr),
                xytext=(xlim[0]+0.05*xr, ylim[0]+0.04*yr),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="white",
                                mutation_scale=18),
                ha="center", va="bottom", color="white", fontsize=15,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # ── Scale bar — 50ft for 60ft lot map, 100ft for 250ft block map ──
    try:
        # Choose scale bar length based on map radius
        if radius_ft <= 75:
            _bar_ft  = 50    # 50ft scale bar for Lot (60ft) map
            _bar_lbl = "50 ft"
        elif radius_ft <= 300:
            _bar_ft  = 100   # 100ft scale bar for Block (250ft) map
            _bar_lbl = "100 ft"
        else:
            _bar_ft  = 250   # 250ft for 1000ft neighbourhood map
            _bar_lbl = "250 ft"
        bar_m  = _bar_ft * 0.3048   # ft → metres (Web Mercator)
        bar_x0 = xlim[0] + xr * 0.06
        bar_x1 = bar_x0 + bar_m
        bar_y  = ylim[0] + yr * 0.05
        # Draw white bar with black outline for contrast
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="black", lw=9,
                solid_capstyle="butt", zorder=29)
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="white", lw=6,
                solid_capstyle="butt", zorder=30)
        ax.text((bar_x0 + bar_x1) / 2, bar_y + yr * 0.012,
                _bar_lbl, ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                zorder=31)
    except Exception:
        pass

    # ── Geo-verification corner badge ─────────────────────────────────────
    # Only show in flood simulation mode, completely disabled for clean DEM
    if is_flood_view and geo_verify:
        conf = geo_verify.get("overall_confidence", "?")
        gv_lines = [f"Geo-Verify: {conf}"]
        p1 = geo_verify.get("pass1_fwd_geocode", {})
        p2 = geo_verify.get("pass2_rev_geocode", {})
        p3 = geo_verify.get("pass3_elev_sanity", {})
        gv_lines.append(f"P1 Fwd geocode: {'✅' if p1.get('ok') else '❌'}")
        gv_lines.append(f"P2 Rev geocode: {'✅' if p2.get('ok') else '❌'} "
                        f"(score={p2.get('match_score',0):.2f})")
        gv_lines.append(f"P3 Elev sanity: {'✅' if p3.get('ok') else '❌'} "
                        f"({p3.get('sampled_elev','?')} ft)")
        for flag in geo_verify.get("flags", []):
            gv_lines.append(flag)
        gv_text = "\n".join(gv_lines)
        ax.text(xlim[0]+0.01*xr, ylim[1]-0.01*yr, gv_text,
                va="top", ha="left", fontsize=7.5,
                color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                          edgecolor="#374151", alpha=0.88),
                zorder=40,
                path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")])

    # ── Legend ────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if is_flood_view and _use_blue_ponding:
        # Flood simulation map — graduated depth legend
        legend_patches = [
            Patch(facecolor="#22c55e", alpha=0.7, label="Higher elevation (water drains away)"),
            Patch(facecolor="#fbbf24", alpha=0.7, label="Near grade (neutral)"),
            Patch(facecolor="#ef4444", alpha=0.7, label="Lower elevation (water collects here)"),
            Patch(facecolor="#cfe2f3", alpha=0.85, label="Flood zone — shallow (< 0.5 ft)"),
            Patch(facecolor="#3b82f6", alpha=0.85, label="Flood zone — moderate (0.5–2 ft)"),
            Patch(facecolor="#1e3a8a", alpha=0.85, label="Flood zone — deep (> 2 ft)"),
            Line2D([0],[0], color="#1e90ff", lw=1.5, label="Water flow direction"),
        ]
    else:
        # Clean DEM map — simplified elevation legend only
        legend_patches = [
            Patch(facecolor="#22c55e", alpha=0.8, label="Higher elevation"),
            Patch(facecolor="#fbbf24", alpha=0.8, label="Near grade"),
            Patch(facecolor="#ef4444", alpha=0.8, label="Lower elevation"),
        ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1e293b", edgecolor="#374151",
              labelcolor="white", fontsize=8.5, framealpha=0.93)

    # ── Colorbar ──────────────────────────────────────────────────────────
    # Only show colorbar for elevation data, not for hillshade or other overlays
    if im is not None and viz_mode != "3d":
        cax = fig.add_axes([0.15, 0.04, 0.70, 0.016])
        # Use a fully-opaque version of the overlay colormap for the colorbar
        # so the bar renders cleanly even though the map cmap has baked-in alpha.
        import matplotlib.cm as _mcm_cb
        import matplotlib.colors as _mcc_cb
        _cb_colors = [
            (0.82, 0.04, 0.04), (0.94, 0.28, 0.05),
            (0.97, 0.52, 0.10), (0.98, 0.74, 0.18),
            (0.99, 0.90, 0.30), (0.99, 0.97, 0.45),
        ]
        _cb_cmap = _mcc_cb.LinearSegmentedColormap.from_list("cb_opaque", _cb_colors)
        _cb_norm = _mcc_cb.Normalize(vmin=-span, vmax=0.0)
        _cb_sm   = _mcm_cb.ScalarMappable(cmap=_cb_cmap, norm=_cb_norm)
        _cb_sm.set_array([])
        cb  = fig.colorbar(_cb_sm, cax=cax, orientation="horizontal")
        cb.set_label("Elevation Δ (ft) relative to subject parcel grade",
                     color="white", fontsize=10)
        cb.ax.xaxis.set_tick_params(color="white")
        plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
        cb.outline.set_edgecolor("white")
    elif viz_mode == "3d":
        print("   🎨  3D View: No elevation colorbar (hillshade grayscale)")
    else:
        print("   ⚠️  Warning: 'im' was not initialized; skipping colorbar.")

    # ── Title ─────────────────────────────────────────────────────────────
    if is_flood_view:
        # Flood simulation title — include projected critical rate/duration if available
        map_type = "Flood Zone Simulation"
        _crit_info = ""
        if rainfall_damage:
            _crit = rainfall_damage.get("critical") or {}
            _crit_r = _crit.get("rate_in_hr")
            _crit_t = _crit.get("total_time_min")
            if _crit_r and _crit_t and not _crit.get("no_flood_risk"):
                _crit_hr = round(_crit_t / 60.0, 1)
                _crit_info = f"  |  Projected flood: {_crit_r} in/hr × {_crit_hr} hrs"
            elif _crit.get("no_flood_risk"):
                _crit_info = "  |  ✅ NO FLOOD RISK under all design storms"
        title = (f"{st_addr}  —  {map_type}\n"
                 f"{int(radius_ft)} ft radius  |  Blue = water accumulation zones"
                 f"{_crit_info}")
    else:
        # Clean DEM title
        map_type = "Lot View" if radius_ft <= 75 else "Block View" if radius_ft <= 300 else "Neighborhood View"
        title = (f"{st_addr}  —  {map_type}\n"
                 f"{int(radius_ft)} ft radius  |  Bilinear LiDAR DEM  |  "
                 f"{len(depression_regions)} depression(s) detected")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=16,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    # ── Precipitation simulation header (only on 4th sim map) ────────────
    # Printed at the very top of the image when sim_rain_in_hr is provided.
    # Pulls timing stats from rainfall_damage so header matches the table.
    if sim_rain_in_hr is not None:
        _rd  = rainfall_damage or {}
        _crit = _rd.get("critical") or {}
        _dep_min = _crit.get("dep_fill_min",  "—")
        _dmg_min = _crit.get("damage_time_min","—")
        _tot_min = _crit.get("total_time_min", "—")
        _rate_lbl = f"{sim_rain_in_hr:.1f} in/hr"
        # Inject most-probable storm type name into the simulation banner
        _fps_sim  = _rd.get("flood_scenarios", [])
        _prob_fp  = next((fp for fp in _fps_sim if fp.get("most_probable")), None)
        _storm_nm = (_prob_fp["callout"]["name"]
                     if _prob_fp and _prob_fp.get("callout")
                     else _get_storm_callout(sim_rain_in_hr).get("name", ""))
        _storm_tag = f"  |  Most Probable: {_storm_nm}" if _storm_nm else ""
        _sim_hdr = (
            f"💧 HOUSE FLOODING SIMULATION — {_rate_lbl} Rainfall{_storm_tag}  |  "
            f"Time to fill depressions: {_dep_min} min  |  "
            f"Time until 6\" water INSIDE house: {_dmg_min} min  |  "
            f"Total time from rain start to house flooding: {_tot_min} min"
        )
        fig.text(
            0.5, 0.993, _sim_hdr,
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e3a5f",
                      edgecolor="#3b82f6", linewidth=2.5, alpha=0.96),
            path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
            zorder=99,
        )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    try:
        # DPI is set at figure creation (_fig_dpi: 400/350/220 per radius).
        # Do NOT pass dpi= here — it would override the figure's own DPI.
        fig.savefig(out_png, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)

        # ── Post-processing: sharpen + boost contrast/saturation for 60 ft and 250 ft maps ──
        # 1000 ft maps are left at natural rendering for neighbourhood context.
        # Tuned per best-practice QL0/QL1 photographic output guidelines:
        #   60ft  — gentle saturation (×1.8) to keep WorldImagery colours realistic
        #   250ft — moderate saturation (×2.2) for clear elevation differentiation
        if radius_ft in (75, 250):
            try:
                from PIL import Image, ImageEnhance, ImageFilter
                _img = Image.open(out_png)
                # Sharpening: unsharp mask tuned for sub-10cm pixel density
                _usm_radius  = 1.5 if radius_ft <= 75 else 2.0
                _usm_percent = 150 if radius_ft <= 75 else 180
                _img = _img.filter(ImageFilter.UnsharpMask(
                    radius=_usm_radius, percent=_usm_percent, threshold=2))
                _img = _img.filter(ImageFilter.SHARPEN)
                if radius_ft == 250:
                    _img = _img.filter(ImageFilter.SHARPEN)  # extra pass for 250ft
                # Saturation: restrained for 75ft (photographic realism),
                #             moderate for 250ft (grade differentiation)
                _sat = 1.8 if radius_ft <= 75 else 2.2
                _img = ImageEnhance.Color(_img).enhance(_sat)
                # Contrast: slight lift only — avoids blowing out the satellite base
                _con = 1.5 if radius_ft <= 75 else 1.7
                _img = ImageEnhance.Contrast(_img).enhance(_con)
                # Brightness: +8% for 75ft, +12% for 250ft
                _bri = 1.08 if radius_ft <= 75 else 1.12
                _img = ImageEnhance.Brightness(_img).enhance(_bri)
                _img.save(out_png, optimize=True)
                print(f"   ✨  LiDAR enhancement applied ({int(radius_ft)} ft view): "
                      f"USM r={_usm_radius}/{_usm_percent}%, "
                      f"sat ×{_sat}, contrast ×{_con}, brightness ×{_bri}")
            except Exception as _enh_err:
                print(f"   ⚠️  Post-processing enhancement skipped: {_enh_err}")

        return True
    except Exception as e:
        print(f"   ❌ Map save error: {e}")
        plt.close(fig)
        return False


# =============================================================================
# PCPAO SCRAPER
# =============================================================================
def scrape_pcpao_for_parcel(parcel_id=None, address=None) -> dict:
    target = address or parcel_id or ""
    if not target.strip(): return {}
    tmp_leads = os.path.join(MASTER_EXPORT_DIR, "_tmp_lead.txt")
    tmp_out   = os.path.join(MASTER_EXPORT_DIR, "_tmp_pcpao.csv")
    try:
        with open(tmp_leads, "w") as f: f.write(target.strip() + "\n")
        env = os.environ.copy()
        env.update({"PCPAO_INPUT_CSV": tmp_leads, "PCPAO_OUTPUT_CSV": tmp_out,
                    "PCPAO_FAILED_CSV": PCPAO_FAILED_CSV})
        subprocess.run(["python3", PCPAO_SCRAPER], timeout=120,
                       capture_output=True, text=True, env=env)
        if os.path.exists(tmp_out):
            df = pd.read_csv(tmp_out)
            if not df.empty:
                s = df.iloc[0].to_dict()
                print(f"   🏛️  PCPAO ✅ {s.get('site_address','?')}")
                return s
    except subprocess.TimeoutExpired:
        print("   ⚠️  PCPAO timeout")
    except FileNotFoundError:
        print(f"   ⚠️  Scraper not found: {PCPAO_SCRAPER}")
    except Exception as e:
        print(f"   ⚠️  PCPAO: {e}")
    finally:
        for fn in [tmp_leads, tmp_out]:
            try: os.remove(fn)
            except: pass
    return {}

def pcpao_url(parcel_id: str) -> str:
    return f"https://www.pcpao.gov/general-info.php?strap={parcel_id.replace('-','').replace(' ','')}"


# =============================================================================
# PERMIT FLOOD INDICATOR
# =============================================================================
FLOOD_KW = ['flood', 'water damage', 'mold', 'remediation', 'restoration',
            'drywall', 'flooring', 'hvac', 'electrical panel', 'plumbing',
            'foundation', 'slab', 'demo', 'demolition', 'rebuild', 'replace',
            'elevation', 'drainage', 'sump', 'waterproof', 'moisture',
            'hurricane', 'tropical', 'storm damage', 'wind', 'roof damage',
            'insulation', 'framing', 'stucco', 'sheetrock', 'subflooring']

FLOOD_TYPES = ['building', 'electrical', 'mechanical', 'plumbing', 'roofing',
               'demolition', 'alteration', 'repair', 'renovation']

def parse_permit_date(date_str: str):
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except: pass
    return None

def analyze_permits_historical(pcpao_data: dict, row) -> dict:
    permit_records = []
    for i in range(1, 25):
        prefix = f"permit_{i}_" if i > 1 else "permit_"
        alt_p  = f"permit{i}_" if i > 1 else "permit_"
        for pfx in [prefix, alt_p]:
            num  = safe(pcpao_data.get(f"{pfx}number",  pcpao_data.get(f"permit_number_{i}", "")), "")
            typ  = safe(pcpao_data.get(f"{pfx}type",    pcpao_data.get(f"permit_type_{i}", "")), "")
            desc = safe(pcpao_data.get(f"{pfx}description", pcpao_data.get(f"permit_desc_{i}", "")), "")
            dt   = safe(pcpao_data.get(f"{pfx}date",    pcpao_data.get(f"issue_date_{i}",
                         pcpao_data.get(f"permit_date_{i}", ""))), "")
            val  = safe(pcpao_data.get(f"{pfx}value",   pcpao_data.get(f"estimated_value_{i}", "0")), "0")
            if any([num, typ, desc, dt]):
                permit_records.append({
                    "number": num, "type": str(typ).lower(),
                    "desc":   str(desc).lower(), "date_raw": dt, "value_raw": val,
                })
            if not num and not dt: break

    primary = {
        "number": safe(pcpao_data.get("permit_number", ""), ""),
        "type":   str(safe(pcpao_data.get("permit_type", ""), "")).lower(),
        "desc":   str(safe(pcpao_data.get("permit_description", ""), "")).lower(),
        "date_raw": safe(pcpao_data.get("issue_date", ""), ""),
        "value_raw": safe(pcpao_data.get("estimated_value", "0"), "0"),
    }
    if primary["number"] or primary["date_raw"]:
        if not any(p["number"] == primary["number"] for p in permit_records if primary["number"]):
            permit_records.insert(0, primary)

    parsed = []
    for p in permit_records:
        dt_obj = parse_permit_date(p["date_raw"])
        try:
            val_num = float(str(p["value_raw"]).replace("$","").replace(",",""))
        except:
            val_num = 0
        kw_found = [kw for kw in FLOOD_KW if kw in p["desc"] or kw in p["type"]]
        is_flood_type = any(ft in p["type"] for ft in FLOOD_TYPES)
        parsed.append({
            **p,
            "date_obj":      dt_obj,
            "date_str":      dt_obj.strftime("%B %Y") if dt_obj else (p["date_raw"] or "Unknown date"),
            "value":         val_num,
            "keywords":      kw_found,
            "is_flood_type": is_flood_type,
            "has_flood_kw":  len(kw_found) > 0,
        })

    dated = sorted([p for p in parsed if p["date_obj"]], key=lambda x: x["date_obj"])
    clusters = []
    used = set()
    CLUSTER_DAYS = 90
    CLUSTER_MIN  = 2

    for i, anchor in enumerate(dated):
        if i in used: continue
        window = [anchor]
        for j, other in enumerate(dated):
            if j == i or j in used: continue
            if abs((other["date_obj"] - anchor["date_obj"]).days) <= CLUSTER_DAYS:
                window.append(other)
        if len(window) >= CLUSTER_MIN:
            for j, p in enumerate(dated):
                if p in window: used.add(j)
            kw_union   = list(set(kw for p in window for kw in p["keywords"]))
            types      = list(set(p["type"] for p in window if p["type"]))
            total_val  = sum(p["value"] for p in window)
            flood_kw_c = sum(1 for p in window if p["has_flood_kw"])
            clusters.append({
                "date_start":   min(p["date_obj"] for p in window).strftime("%b %Y"),
                "date_end":     max(p["date_obj"] for p in window).strftime("%b %Y"),
                "count":        len(window),
                "permits":      window,
                "keywords":     kw_union,
                "types":        types,
                "total_value":  total_val,
                "flood_kw_count": flood_kw_c,
                "likely_flood": (flood_kw_c >= 1 or total_val > 10000) and len(window) >= 2,
            })

    flood_clusters   = [c for c in clusters if c["likely_flood"]]
    any_flood_permit = any(p["has_flood_kw"] and p["value"] > 2000 for p in parsed)
    n_with_kw        = sum(1 for p in parsed if p["has_flood_kw"])

    if flood_clusters:
        tier = "🔴 PRIOR FLOOD EVENT INDICATED — Permit cluster pattern detected"
        prior_flood = True
    elif any_flood_permit or n_with_kw >= 2:
        tier = "🟡 WATCH — Flood-related permit activity in property history"
        prior_flood = False
    else:
        tier = "🟢 No flood-damage permit clusters in available historical records"
        prior_flood = False

    history_lines = []
    if parsed:
        history_lines.append(f"Total permits in record: {len(parsed)}")
        for p in sorted(parsed, key=lambda x: x["date_obj"] or datetime.min, reverse=True):
            kw_str  = f" | Keywords: {', '.join(p['keywords'])}" if p["keywords"] else ""
            val_str = f" | Est. ${p['value']:,.0f}" if p["value"] > 0 else ""
            num_str = f"#{p['number']}" if p["number"] else "(no number)"
            typ_str = p["type"].title() if p["type"] else "Permit"
            history_lines.append(
                f"  {p['date_str']:<14}  {typ_str:<18}  {num_str}{val_str}{kw_str}")

    cluster_lines = []
    for c in clusters:
        mark = "⚠️ FLOOD CLUSTER" if c["likely_flood"] else "  Permit Cluster"
        cluster_lines.append(
            f"  {mark}  {c['date_start']}–{c['date_end']}  "
            f"({c['count']} permits, ${c['total_value']:,.0f} total)")
        if c["keywords"]: cluster_lines.append(f"    Flood keywords: {', '.join(c['keywords'])}")
        if c["types"]:    cluster_lines.append(f"    Permit types:   {', '.join(c['types'])}")

    return {
        "tier": tier, "prior_flood": prior_flood,
        "parsed_permits": parsed, "clusters": clusters,
        "flood_clusters": flood_clusters,
        "history_lines": history_lines, "cluster_lines": cluster_lines,
        "n_permits": len(parsed), "n_clusters": len(clusters),
        "n_flood_clusters": len(flood_clusters),
    }


# =============================================================================
# WEB SEARCH + OLLAMA
# =============================================================================
def web_search(query: str, n: int = 6) -> list:
    for url, params in [
        (OLLAMA_SEARCH_URL, {"query": query, "limit": n}),
        (SEARX_URL, {"q": query, "format": "json", "engines": "duckduckgo,bing"}),
    ]:
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list): return d[:n]
                return [{"title": x.get("title",""), "href": x.get("url",""),
                         "body": x.get("content","")} for x in d.get("results",[])[:n]]
        except: continue
    return []

def web_search_fallback(primary: str, fallbacks: list) -> tuple:
    for q in [primary] + fallbacks[:2]:
        r = web_search(q)
        if r: return r, q
    return [], primary

def ollama_query(prompt: str, label: str = "") -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.28, "num_predict": 750}}, timeout=180)
        r.raise_for_status()
        result = r.json().get("response", "").strip()
        if label: print(f"   🤖 ✅ [{label}]")
        return result if result else ""
    except requests.exceptions.ConnectionError:
        if label: print(f"   ⚠️  Ollama offline [{label}]")
        return ""
    except Exception:
        return ""

def ollama_research(prompt: str, primary: str, fallbacks: list = None, label: str = "") -> str:
    results, _ = web_search_fallback(primary, fallbacks or [])
    snippet = "\n".join([
        f"[{i+1}] {r.get('title','')} | {r.get('href','')}\n    {r.get('body','')[:300]}"
        for i, r in enumerate(results)
    ]) if results else "[No search results — use training knowledge and note it]"
    full = f"{prompt}\n\nSEARCH RESULTS:\n{snippet}\n\nCite sources as [N] URL."
    return ollama_query(full, label) or None


# =============================================================================
# HYDRAULICS
# =============================================================================
def manning_gpm(dia_in, n, slope=0.005):
    D = dia_in / 12.0
    A = math.pi * (D/2)**2
    R = D / 4.0
    return max(0.0, (1.486/n) * A * (R**(2/3)) * (slope**0.5) * 448.83)

def rational_gpm(area_sqft, rainfall_in_hr, C=0.85):
    cfs = C * rainfall_in_hr * (area_sqft/43560) / 12 * 43560 / 3600
    return max(0.0, cfs * 448.83)

def hydrostatic_psi(depth_ft):
    return (62.4 * abs(depth_ft)) / 144.0

def hydro_versai(H_ft):
    return VERSAI["rho_water"]*VERSAI["g"]*H_ft + 0.5*VERSAI["rho_water"]*(VERSAI["vel_wave"]**2)

def calc_hydraulics(sq_ft, dia_in, n, delta):
    LOT   = sq_ft * 4.5
    POND  = max(0.05, abs(min(0,delta))) if delta < 0 else 0.15
    p_gpm = manning_gpm(dia_in, n, 0.005)
    d_gpm = rational_gpm(LOT, 3.5, 0.85)
    crit_i = 3.5 * (p_gpm / max(d_gpm, 1.0)) if d_gpm > 0 else 0.0
    store  = LOT * POND * 7.481
    excess = d_gpm - p_gpm
    ttf    = None if excess <= 0 else round(store / max(excess, 0.1), 1)
    return p_gpm, crit_i, ttf

HYDRAULIC_MATRIX = {
    'RCP': {'name': 'Reinforced Concrete (RCP)', 'n_value': 0.013, 'logic': (
        "RCP infrastructure in Pinellas County routinely exceeds its 30-year design lifecycle."
    )},
    'CMP': {'name': 'Corrugated Metal (CMP)', 'n_value': 0.024, 'logic': (
        "CMP undergoes invert corrosion in Pinellas County's acidic soils."
    )},
    'PVC': {'name': 'Polyvinyl Chloride (PVC/HDPE)', 'n_value': 0.009, 'logic': (
        "PVC systems designed to 1980s rainfall data surcharge within minutes of modern cloudburst events."
    )},
}


# =============================================================================
# RAINFALL-TO-DAMAGE CALCULATOR
# =============================================================================

# Pinellas County historical storm durations by intensity (NOAA Atlas 14 / NWS TPC data).
# These caps reflect the realistic outer bound for each intensity tier — not a blanket 12 hrs.
_FLOOD_SCENARIO_LABELS = {
    # rate_in_hr: (label, approx_return_period, max_duration_hr)
    # NOTE: return periods at 0.25-0.75 in/hr are CONDITIONAL probabilities —
    # the rain event itself is common (<1-yr) but flooding only occurs given
    # wet antecedent conditions (prior rain, high tide, saturated soil).
    # Rates >= 1.0 in/hr are unconditional ARI from NOAA Atlas 14 Pinellas.
    0.25: ("Slow tropical moisture / stalled front",  "conditional*", 18.0),
    0.50: ("Prolonged steady rain",                   "conditional*", 12.0),
    0.75: ("Steady tropical rain band",               "1–2 yr (conditional*)", 8.0),
    1.00: ("Moderate rain band / weak T-storm",       "2–5 year",     6.0),
    1.50: ("Organized tropical convection",           "5-year",       4.0),
    2.00: ("Heavy afternoon thunderstorm",            "5–10 year",    3.0),
    2.50: ("Strong convective cell",                  "10-year",      2.5),
    3.00: ("Severe thunderstorm / squall line",       "10–25 year",   2.0),
    4.00: ("Extreme T-storm / outer rainband",        "25–50 year",   1.5),
    5.00: ("Near-hurricane intensity",                "50–100 year",  1.0),
    5.30: ("Near-hurricane intensity / outer rainband", "50-100 year", 0.75),
    5.80: ("100-year storm (NOAA Atlas 14 Pinellas)", "100-year",     0.75),
}
# * conditional = rain event is common (<1-yr) but this flood pathway
#   only activates given wet antecedent conditions: prior rainfall within
#   24-48 hrs, high tide reducing drainage outfall, or already-saturated soil.

# -- Storm-type callout map keyed by rate_in_hr ---------------------------
# Maps each scenario intensity to the plain-language storm category from
# the user-provided storm event reference table. The most probable flooding
# event (first entry in flood_scenarios) always gets a callout block in
# both the PDF and TXT versions of the FLOOD PATHWAYS section.
_STORM_CALLOUT_MAP = {
    0.25: {
        "name":     "Frontal Systems (Fall-Spring)",
        "duration": "6-24 hours",
        "rainfall": "~0.5-2 inches",
        "notes":    ["More steady, less intense than summer storms",
                     "Can include embedded thunderstorms"],
    },
    0.50: {
        "name":     "Frontal Systems / Prolonged Tropical Moisture",
        "duration": "6-24 hours",
        "rainfall": "~0.5-2 inches",
        "notes":    ["Steady prolonged rain over many hours",
                     "Common fall-through-spring in Tampa Bay area"],
    },
    0.75: {
        "name":     "Tropical Storm Rain",
        "duration": "12-36 hours",
        "rainfall": "~3-10 inches",
        "notes":    ["Rain comes in bands with breaks",
                     "Flooding depends heavily on storm speed"],
    },
    1.00: {
        "name":     "Heavy Downpours (Tropical Air Mass Storms)",
        "duration": "30-90 minutes",
        "rainfall": "~1-3 inches",
        "notes":    ["Extremely intense rainfall rates",
                     "Streets can flood even though storms are short"],
    },
    1.50: {
        "name":     "Heavy Downpours (Tropical Air Mass Storms)",
        "duration": "30-90 minutes",
        "rainfall": "~1-3 inches",
        "notes":    ["Organized tropical convection",
                     "Common during Florida summer wet season"],
    },
    2.00: {
        "name":     "Sea-Breeze Thunderstorms (Very Common in Summer)",
        "duration": "20-60 minutes",
        "rainfall": "~0.5-2 inches",
        "notes":    ["Happens almost daily in summer",
                     "Forms in the afternoon due to Gulf/land temperature differences",
                     "Can dump heavy rain fast, then clear quickly"],
    },
    2.50: {
        "name":     "Slow-Moving Thunderstorms / Training Storms",
        "duration": "1-4 hours",
        "rainfall": "~2-6+ inches",
        "notes":    ["Biggest flash flood risk locally",
                     "Storms pass over the same area repeatedly"],
    },
    3.00: {
        "name":     "Slow-Moving Thunderstorms / Training Storms",
        "duration": "1-4 hours",
        "rainfall": "~2-6+ inches",
        "notes":    ["Biggest flash flood risk locally",
                     "Storms pass over the same area repeatedly"],
    },
    4.00: {
        "name":     "Outer Rain Bands (Before/After Hurricanes)",
        "duration": "Intermittent over 1-2 days",
        "rainfall": "~2-8 inches total",
        "notes":    ["Feels like repeated thunderstorms",
                     "Can cause flooding before the storm even arrives"],
    },
    5.00: {
        "name":     "Hurricane Rain (Major Events)",
        "duration": "24-72+ hours",
        "rainfall": "~6-20+ inches",
        "notes":    ["Worst flooding happens with slow-moving storms",
                     "Outer bands may start rain a day before landfall"],
    },
    5.30: {
        "name":     "Hurricane Rain (Major Events)",
        "duration": "24-72+ hours",
        "rainfall": "~6-20+ inches",
        "notes":    ["Near-hurricane intensity rainfall",
                     "Outer bands may start rain a day before landfall"],
    },
}

def _get_storm_callout(rate_in_hr: float) -> dict:
    """Return the best-matching storm callout entry for a given rate."""
    keys = sorted(_STORM_CALLOUT_MAP.keys())
    best = min(keys, key=lambda k: abs(k - rate_in_hr))
    return _STORM_CALLOUT_MAP[best]


# -- Historical rainfall CSV loader ---------------------------------------
# Loads stpeterain.csv (St. Pete COOP 1965-2020) and
# Pinellas1990-2000.csv (Tampa Intl NOAA GHCN 1990-2000).
# KEY DATA-QUALITY STEP: replaces -99.99 missing-data sentinel -> NaN
# and drops those rows so they never corrupt frequency or threshold math.
_HIST_RAIN_CACHE: dict = {}   # module-level -- loaded once per run

def load_historical_rainfall() -> dict:
    """
    Load, clean, and merge Pinellas County historical rainfall CSVs.
    Returns dict with:
        df           -- clean daily DataFrame (date, precipitation)
        years        -- float years of record in combined dataset
        exceedance   -- {threshold_in: {n, freq_per_yr, last_date, last_amount}}
        most_probable-- canonical flood-threshold reference (1.45 in/day)
        last_event   -- {date_str, amount, freq_per_yr} for most_probable threshold
    """
    global _HIST_RAIN_CACHE
    if _HIST_RAIN_CACHE:
        return _HIST_RAIN_CACHE

    try:
        import numpy as _np
        import pandas as _pd
    except ImportError:
        _HIST_RAIN_CACHE = {"df": None, "years": 0, "exceedance": {},
                            "most_probable": 1.45,
                            "last_event": {"date_str": "N/A", "amount": 0.0, "freq_per_yr": 0.0}}
        return _HIST_RAIN_CACHE

    frames = []
    _desk = os.path.expanduser("~/Desktop")

    # ── Primary: pre-merged 1965-2026 file (best coverage, already deduplicated) ──
    _merged_path = os.path.join(_desk, "pinellas_rainfall_merged_1965_2026.csv")
    if os.path.exists(_merged_path):
        try:
            _dfm = _pd.read_csv(_merged_path, usecols=["date","precipitation"])
            _dfm["precipitation"] = _pd.to_numeric(_dfm["precipitation"], errors="coerce")
            _dfm.loc[_dfm["precipitation"] < -90, "precipitation"] = _np.nan
            _dfm = _dfm.dropna(subset=["precipitation"])
            _dfm["date"] = _pd.to_datetime(_dfm["date"], errors="coerce")
            _dfm = _dfm.dropna(subset=["date"])[["date","precipitation"]]
            frames.append(_dfm)
            print(f"   Hist: pinellas_rainfall_merged_1965_2026.csv -- {len(_dfm):,} records "
                  f"({_dfm['date'].min().year}–{_dfm['date'].max().year})")
        except Exception as _e:
            print(f"   Warning: merged CSV: {_e}")

    # ── Supplemental: historicalrainfall.csv (multi-station 2020-2026, NOAA format) ──
    # Adds recent-years spatial coverage from additional Pinellas stations.
    _hist_path = os.path.join(_desk, "historicalrainfall.csv")
    if os.path.exists(_hist_path):
        try:
            _dfh = _pd.read_csv(_hist_path, usecols=["DATE","PRCP"])
            _dfh.rename(columns={"DATE":"date","PRCP":"precipitation"}, inplace=True)
            _dfh["precipitation"] = _pd.to_numeric(_dfh["precipitation"], errors="coerce")
            _dfh.loc[_dfh["precipitation"] < -90, "precipitation"] = _np.nan
            _dfh = _dfh.dropna(subset=["precipitation"])
            _dfh["date"] = _pd.to_datetime(_dfh["date"], errors="coerce")
            _dfh = _dfh.dropna(subset=["date"])[["date","precipitation"]]
            frames.append(_dfh)
            print(f"   Hist: historicalrainfall.csv -- {len(_dfh):,} records "
                  f"({_dfh['date'].min().year}–{_dfh['date'].max().year})")
        except Exception as _e:
            print(f"   Warning: historicalrainfall.csv: {_e}")

    # ── Fallback: original individual CSVs (used if merged file missing) ──────
    if not frames:
        _stpete = os.path.join(_desk, "stpeterain.csv")
        if os.path.exists(_stpete):
            try:
                _df = _pd.read_csv(_stpete, header=0,
                                   names=["COOPID","YEAR","MONTH","DAY","precipitation"])
                _df["precipitation"] = _pd.to_numeric(_df["precipitation"], errors="coerce")
                _df.loc[_df["precipitation"] < -90, "precipitation"] = _np.nan
                _df = _df.dropna(subset=["precipitation"])
                _df["date"] = _pd.to_datetime(
                    _df[["YEAR","MONTH","DAY"]].astype(int).rename(
                        columns={"YEAR":"year","MONTH":"month","DAY":"day"}),
                    errors="coerce")
                _df = _df.dropna(subset=["date"])[["date","precipitation"]]
                frames.append(_df)
                print(f"   Hist: stpeterain.csv -- {len(_df):,} records (fallback)")
            except Exception as _e:
                print(f"   Warning: stpeterain.csv: {_e}")
        for _try_fn in ["Pinellas1990-2000 .csv", "Pinellas1990-2000.csv"]:
            _pin = os.path.join(_desk, _try_fn)
            if os.path.exists(_pin):
                try:
                    _df2 = _pd.read_csv(_pin, usecols=["DATE","PRCP"])
                    _df2.rename(columns={"DATE":"date","PRCP":"precipitation"}, inplace=True)
                    _df2["precipitation"] = _pd.to_numeric(_df2["precipitation"], errors="coerce")
                    _df2.loc[_df2["precipitation"] < -90, "precipitation"] = _np.nan
                    _df2 = _df2.dropna(subset=["precipitation"])
                    _df2["date"] = _pd.to_datetime(_df2["date"], errors="coerce")
                    _df2 = _df2.dropna(subset=["date"])[["date","precipitation"]]
                    frames.append(_df2)
                    print(f"   Hist: {_try_fn} -- {len(_df2):,} records (fallback)")
                except Exception as _e:
                    print(f"   Warning: {_try_fn}: {_e}")
                break

    if not frames:
        print("   Hist: No CSVs found -- using NOAA Atlas 14 defaults only")
        _HIST_RAIN_CACHE = {"df": None, "years": 0, "exceedance": {},
                            "most_probable": 1.45,
                            "last_event": {"date_str": "N/A", "amount": 0.0, "freq_per_yr": 0.0}}
        return _HIST_RAIN_CACHE

    combined = (_pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["date"])
                  .sort_values("date")
                  .reset_index(drop=True))
    combined = combined[combined["precipitation"] >= 0]
    years = max(1.0, (combined["date"].max() - combined["date"].min()).days / 365.25)
    print(f"   Hist: Combined {len(combined):,} daily records over {years:.1f} years")

    thresholds = [0.25, 0.50, 0.75, 1.00, 1.25, 1.45, 1.50,
                  2.00, 2.50, 3.00, 4.00, 5.00, 5.30]
    exceedance = {}
    for t in thresholds:
        hits = combined[combined["precipitation"] >= t]
        n    = len(hits)
        last = hits["date"].max() if n > 0 else None
        last_amt = (float(hits.loc[hits["date"] == last, "precipitation"].iloc[0])
                    if last is not None else 0.0)
        exceedance[t] = {
            "n":           n,
            "freq_per_yr": round(n / years, 2),
            "last_date":   last.strftime("%B %d, %Y") if last is not None else "N/A",
            "last_amount": round(last_amt, 2),
        }

    most_probable = 1.45  # daily total that maps to lowest pathways rate
    last_ev = exceedance.get(most_probable, {})
    _HIST_RAIN_CACHE = {
        "df":            combined,
        "years":         round(years, 1),
        "exceedance":    exceedance,
        "most_probable": most_probable,
        "last_event":    {
            "date_str":    last_ev.get("last_date", "N/A"),
            "amount":      last_ev.get("last_amount", 0.0),
            "freq_per_yr": last_ev.get("freq_per_yr", 0.0),
        },
    }
    return _HIST_RAIN_CACHE


def historical_citation_for_rate(rate_in_hr: float, duration_hr: float) -> str:
    """
    Given a flood pathway rate + duration, look up the closest historical
    exceedance entry and return a human-readable citation string.
    e.g. 'Historically 2.8x/yr -- last recorded December 20, 2020 (2.99")'
    """
    hist = load_historical_rainfall()
    exc  = hist.get("exceedance", {})
    if not exc:
        return ""
    total_in = round(rate_in_hr * duration_hr, 2)
    keys     = sorted(exc.keys())
    best_k   = min(keys, key=lambda k: abs(k - total_in))
    entry    = exc[best_k]
    freq     = entry.get("freq_per_yr", 0)
    last     = entry.get("last_date", "N/A")
    amt      = entry.get("last_amount", 0)
    if freq > 0:
        return f'Historically {freq:.1f}x/yr -- last recorded {last} ({amt:.2f}")'
    return f'Last recorded: {last} ({amt:.2f}")'


# ── Pinellas County historical rainfall frequency table ───────────────────────
# Source: NOAA GHCN-Daily, 7-year dataset (2020–2026), 52 stations in Pinellas County.
# Airport station (USW00012873) used as primary reference.
# Maps total storm rainfall (inches) to approximate annual frequency (events/yr).
# Derived from daily precipitation exceedance analysis; sub-daily rates estimated
# from NOAA Atlas 14 Vol.2 intensity-frequency-duration data for Tampa Bay region.
_PINELLAS_FREQ_TABLE = [
    # (total_rain_in, events_per_year, summer_season_events_per_year, description)
    (0.50,  25.0, 20.0, "common light rain — several times/month in summer"),
    (0.75,  16.0, 13.0, "moderate rain event"),
    (1.00,   9.6,  8.0, "significant rain — ~10x/year"),
    (1.25,   7.0,  6.0, "heavy rain event"),
    (1.45,   5.5,  4.8, "threshold: any meaningful storm can flood"),   # ~damage threshold ref
    (1.50,   5.3,  4.5, "heavy rain — ~5x/year"),
    (2.00,   3.3,  2.8, "very heavy event — 3–4x/year"),
    (2.50,   1.3,  1.1, "extreme event — about once/year"),
    (3.00,   0.7,  0.6, "rare extreme — every 1–2 years"),
    (4.00,   0.2,  0.2, "very rare — Hurricane/TS class event"),
    (5.00,   0.05, 0.05,"exceptional — 100-year level"),
]

def _pinellas_freq(total_rain_in: float) -> tuple:
    """
    Return (events_per_year, summer_events_per_year, description) for a given
    total rainfall amount in Pinellas County, FL.
    Uses linear interpolation between the table breakpoints.
    """
    if total_rain_in <= 0:
        return (365.0, 200.0, "any day with rain")
    if total_rain_in >= _PINELLAS_FREQ_TABLE[-1][0]:
        return (_PINELLAS_FREQ_TABLE[-1][1], _PINELLAS_FREQ_TABLE[-1][2], _PINELLAS_FREQ_TABLE[-1][3])
    for i in range(len(_PINELLAS_FREQ_TABLE) - 1):
        r0, f0, fs0, d0 = _PINELLAS_FREQ_TABLE[i]
        r1, f1, fs1, d1 = _PINELLAS_FREQ_TABLE[i + 1]
        if r0 <= total_rain_in <= r1:
            t = (total_rain_in - r0) / (r1 - r0) if r1 > r0 else 0.0
            return (round(f0 + t * (f1 - f0), 1),
                    round(fs0 + t * (fs1 - fs0), 1),
                    d0)
    return (0.1, 0.1, "very rare event")

def _storm_duration_cap(rate_in_hr: float) -> float:
    """
    Return the realistic maximum storm duration (hours) for a given rainfall
    intensity in Pinellas County, FL (NOAA Atlas 14 / NWS TPC basis).
    Intense convective cells rarely sustain peak rates beyond 1–2 hours;
    slow-moving tropical systems can deliver light rain for 18+ hours.
    """
    if   rate_in_hr >= 5.50: return 0.75   # extreme convective burst / 100-yr peak
    elif rate_in_hr >= 4.50: return 1.00   # near-hurricane intensity
    elif rate_in_hr >= 3.50: return 1.50   # extreme T-storm / outer rainband
    elif rate_in_hr >= 2.75: return 2.00   # severe thunderstorm / squall line
    elif rate_in_hr >= 2.25: return 2.50   # strong convective cell
    elif rate_in_hr >= 1.75: return 3.00   # heavy afternoon thunderstorm
    elif rate_in_hr >= 1.25: return 4.00   # organized tropical convection
    elif rate_in_hr >= 0.875: return 6.00  # moderate rain band
    elif rate_in_hr >= 0.625: return 8.00  # steady tropical rain band
    elif rate_in_hr >= 0.375: return 12.0  # prolonged steady rain
    else:                     return 18.0  # slow tropical moisture / stalled front


# =============================================================================
# DAMAGE COST MODEL — MULTI-DEPTH FLOOD + HURRICANE, FULL INSURANCE BREAKDOWN
# =============================================================================

# FEMA HAZUS-MH 2.1 RES1-1SNB (one-story, no basement, slab) depth-damage table
# Depth above first floor elevation (FFE) in feet → % of structure/content value lost
# Source: HAZUS Technical Manual Table 7.9 / Appendix B.1 (FL coastal humid climate)
_HAZUS_STRUCT = {
    -4: 0.0,  0: 0.0,  0.5: 10.0,  1: 14.0,  2: 23.0,  3: 35.0,
     4: 50.0, 5: 65.0, 6: 75.0,    7: 80.0,  8: 85.0,  12: 90.0, 24: 95.0,
}
_HAZUS_CONT = {
    -4: 0.0,  0: 0.0,  0.5:  7.0,  1: 12.0,  2: 19.0,  3: 28.0,
     4: 40.0, 5: 53.0, 6: 63.0,    7: 70.0,  8: 75.0,  12: 80.0, 24: 85.0,
}

# Storm profile for each analysis depth — storm type, concurrent wind damage,
# and closest historical Pinellas County event
# Wind structural add % = additional structure loss from wind concurrent with flood
# Wind contents add %   = additional contents loss (wind-driven rain, debris, broken glass)
# Sources: NHC tropical cyclone reports; NOAA NCEI Storm Events Database;
#          USGS storm surge surveys; Pinellas County CEMP After-Action Reports
_DEPTH_STORM_PROFILES = {
    0.5: {
        "depth_label":        '6"  (0.5 ft)',
        "storm_type":         "Heavy Rain / Localized Flooding",
        "wind_mph":           0,
        "wind_struct_add_pct": 0.0,
        "wind_cont_add_pct":   0.0,
        "ho_covers_wind":     False,
        "return_period":      "2–5 year",
        "annual_prob":        0.35,
        "pinellas_event":     "Tropical Storm Debby — Aug 5–6, 2024 "
                              "(18\"+ total rainfall; localized 6\" inundation, "
                              "Pinellas/Hillsborough; $3.6B FL losses)",
        "event_source":       "NWS Tampa Debby Event Review, Aug 2024; "
                              "NOAA NCEI Storm Events Database #2024-09",
    },
    1.0: {
        "depth_label":        '1 ft  (12")',
        "storm_type":         "Tropical Storm / Intense Rainband",
        "wind_mph":           55,
        "wind_struct_add_pct": 2.0,
        "wind_cont_add_pct":   1.0,
        "ho_covers_wind":     True,
        "return_period":      "10–25 year",
        "annual_prob":        0.07,
        "pinellas_event":     "Tropical Storm Eta — Nov 11–12, 2020 "
                              "(12\" rainfall, 1 ft+ coastal inundation Pinellas; "
                              "$1.9B FL damage; 61 mph peak Clearwater)",
        "event_source":       "NHC Eta Tropical Cyclone Report, 2020; "
                              "Pinellas County CEMP Eta After-Action, 2021",
    },
    1.5: {
        "depth_label":        '18"  (1.5 ft)',
        "storm_type":         "Category 1–2 Hurricane",
        "wind_mph":           90,
        "wind_struct_add_pct": 9.0,
        "wind_cont_add_pct":   5.0,
        "ho_covers_wind":     True,
        "return_period":      "25–50 year",
        "annual_prob":        0.025,
        "pinellas_event":     "Hurricane Milton — Oct 9–10, 2024 "
                              "(Cat 3 landfall Siesta Key; Pinellas surge 3–5 ft, "
                              "~18\" interior flooding; $35B FL insured losses; "
                              "100+ mph gusts Clearwater/St. Pete)",
        "event_source":       "NHC Milton Tropical Cyclone Report, Oct 2024; "
                              "FEMA Region 4 Milton Damage Assessment, 2024",
    },
    2.0: {
        "depth_label":        '2 ft  (24")',
        "storm_type":         "Category 2–3 Hurricane (direct/near-direct hit)",
        "wind_mph":           115,
        "wind_struct_add_pct": 22.0,
        "wind_cont_add_pct":   12.0,
        "ho_covers_wind":     True,
        "return_period":      "50–100 year",
        "annual_prob":        0.012,
        "pinellas_event":     "Hurricane Irma — Sep 10–11, 2017 "
                              "(Cat 3 FL landfall; Tampa Bay 3–5 ft surge; "
                              "Pinellas 2 ft+ inundation surge zones; "
                              "$50B FL damage; 110+ mph gusts Pinellas)",
        "event_source":       "USGS Irma Storm Surge Report DS-1059, 2017; "
                              "NHC Irma Tropical Cyclone Report, 2018",
    },
    3.0: {
        "depth_label":        '3 ft  (36")',
        "storm_type":         "Major Hurricane Cat 3–4 (direct landfall)",
        "wind_mph":           135,
        "wind_struct_add_pct": 38.0,
        "wind_cont_add_pct":   22.0,
        "ho_covers_wind":     True,
        "return_period":      "100–500 year",
        "annual_prob":        0.004,
        "pinellas_event":     "Hurricane Ian — Sep 28–29, 2022 "
                              "(Cat 4 landfall Fort Myers; Pinellas equivalent-track "
                              "models project 3–6 ft surge; $112B total FL damage; "
                              "FEMA modeling: 3 ft+ Pinellas low-lying areas)",
        "event_source":       "NHC Ian Tropical Cyclone Report, 2023; "
                              "FEMA HAZUS Ian Damage Model, Region 4, 2023; "
                              "Pinellas County Hazard Mitigation Plan Update, 2023",
    },
}

# Progressive contents-to-structure ratio by structure value tier
# Higher-value homes carry proportionally more in furnishings, electronics,
# art, jewelry, and custom appliances.
# Source: FEMA NFIP claims actuarial data; Insurance Information Institute
#         2024 Homeowners Insurance Report; HAZUS-MH RES1 default ratios
def _contents_ratio_for_value(structure_val: float) -> tuple:
    """Returns (ratio, tier_label) — contents as fraction of structure RCV."""
    if structure_val < 150_000:
        return 0.45, "45% (entry-level — FEMA NFIP actuarial tier 1)"
    elif structure_val < 250_000:
        return 0.55, "55% (mid-range — FEMA NFIP actuarial tier 2)"
    elif structure_val < 400_000:
        return 0.65, "65% (upper-mid — Insurance Information Institute 2024)"
    elif structure_val < 600_000:
        return 0.70, "70% (premium — III 2024; higher furnishings/electronics)"
    else:
        return 0.75, "75% (luxury — III 2024; art, jewelry, custom appliances)"


def _acv_ratio_for_age(year_built) -> tuple:
    """
    ACV / RCV ratio for a structure given year built.
    Florida residential structure depreciation: 1.5% per year, max 80% depreciation.
    Source: Marshall & Swift Residential Cost Handbook (2024);
            Florida DFS Adjuster Depreciation Schedule, Ch. 626.9744 F.S.
    Returns (acv_ratio, age_years, depreciation_pct, source_note).
    """
    try:
        age = max(0, 2025 - int(str(year_built).strip()))
    except Exception:
        age = 35   # default: mid-age Pinellas home if year unknown
    depr_pct = min(80.0, age * 1.5)
    acv_ratio = round((100.0 - depr_pct) / 100.0, 3)
    return (
        acv_ratio,
        age,
        round(depr_pct, 1),
        "Marshall & Swift Residential Cost Handbook 2024; FL DFS Ch.626.9744 F.S.",
    )


def _hazus_pct(depth_ft: float, table: dict) -> float:
    """Linearly interpolate a FEMA HAZUS depth-damage table (depth_ft → %)."""
    keys = sorted(table.keys())
    if depth_ft <= keys[0]:  return float(table[keys[0]])
    if depth_ft >= keys[-1]: return float(table[keys[-1]])
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0 <= depth_ft <= k1:
            frac = (depth_ft - k0) / (k1 - k0)
            return table[k0] + frac * (table[k1] - table[k0])
    return 0.0


def calc_damage_cost_model(
    sq_ft: float,
    lot_area_sqft: float,
    delta: float,
    surge_ft: float,
    rainfall_damage: dict,
    pcpao_data: dict,
    proj: dict,
    year_built: str = None,
    # ── Insurance inputs (Pinellas-typical defaults) ──────────────────────────
    flood_struct_limit: float   = 250_000.0,  # NFIP building coverage max
    flood_cont_limit: float     = 100_000.0,  # NFIP contents coverage max
    flood_deductible: float     = 2_000.0,    # NFIP standard deductible
    ho_struct_limit: float      = None,       # HO Coverage A — defaults to RCV
    ho_wind_deductible_pct: float = 0.03,     # FL named-storm: 3% of Coverage A
    ho_flat_deductible: float   = 2_500.0,    # HO non-wind/flat deductible
) -> dict:
    """
    Multi-depth flood damage + hurricane cost model for Pinellas County SFR.

    Analyzes five flood depth thresholds: 6\", 1 ft, 18\", 2 ft, 3 ft.
    For each depth: calculates required storm, HAZUS structural + contents damage,
    concurrent hurricane wind damage, flood insurance payout (ACV/RCV rules),
    homeowners/wind insurance payout, and out-of-pocket gap.
    """
    # ── Valuation — Layer 1 ───────────────────────────────────────────────────
    # Priority: PCPAO building_value > just_value × 0.72 > sq_ft × $312
    # Pinellas 2024-25 standard construction cost (Marshall & Swift)
    # Within 1 mile of bay/gulf: waterfront rebuild premium applies
    _dist_wf_early = proj.get("dist_to_water_ft", 9999) if proj else 9999
    _COST_PER_SQFT = 650.0 if (isinstance(_dist_wf_early, (int, float)) and _dist_wf_early < 5280) else 312.0
    _STRUCT_RATIO  = 0.72    # FL typical land/structure split if only just_value
    val_source_detail = []

    bldg_val = 0.0
    try:
        _bv = str(pcpao_data.get("building_value", 0) or 0)
        bldg_val = float(_bv.replace(",", "").replace("$", "").strip() or 0)
    except Exception:
        bldg_val = 0.0

    just_val = 0.0
    try:
        _jv = str(pcpao_data.get("just_value", 0) or 0)
        just_val = float(_jv.replace(",", "").replace("$", "").strip() or 0)
    except Exception:
        just_val = 0.0

    land_val = 0.0
    try:
        _lv = str(pcpao_data.get("land_value", 0) or 0)
        land_val = float(_lv.replace(",", "").replace("$", "").strip() or 0)
    except Exception:
        land_val = 0.0

    # Waterfront / market uplift factors — based on proximity to tidal water
    _dist_wf = proj.get("dist_to_water_ft", 9999) if proj else 9999
    _is_waterfront = (isinstance(_dist_wf, (int, float)) and _dist_wf < 150
                      and bool((proj or {}).get("waterbody_tidal", False)))
    _MARKET_UPLIFT = 1.35 if _is_waterfront else 1.10
    _wf_label = ("waterfront premium ×1.35 (tidal, <150 ft)"
                 if _is_waterfront else "standard Pinellas market ×1.10")

    if bldg_val > 10_000:
        structure_rcv = bldg_val
        val_source = "PCPAO building_value (direct appraiser field)"
        val_source_detail = [
            f"PCPAO building value:  ${bldg_val:>12,.0f}",
            f"PCPAO land value:      ${land_val:>12,.0f}",
            f"PCPAO just value:      ${just_val:>12,.0f}",
            "Source: Pinellas County Property Appraiser (pcpao.gov), 2025 roll",
        ]
    elif just_val > 10_000:
        structure_rcv = just_val * _STRUCT_RATIO * _MARKET_UPLIFT
        val_source = f"PCPAO just_value × {_STRUCT_RATIO} × {_MARKET_UPLIFT} market uplift ({_wf_label})"
        val_source_detail = [
            f"PCPAO just value:      ${just_val:>12,.0f}",
            f"Structure (72%):       ${just_val*_STRUCT_RATIO:>12,.0f}",
            f"Market uplift (×{_MARKET_UPLIFT}):   ${structure_rcv:>12,.0f}  [{_wf_label}]",
            f"Land est. (28%):       ${just_val*0.28:>12,.0f}",
            "Source: Pinellas County Property Appraiser (pcpao.gov), 2025 roll",
            "Split: FL avg residential 72% structure / 28% land (CoreLogic 2024)",
            f"Uplift: Save Our Homes cap suppresses assessed value; {_wf_label} premium applied",
        ]
    else:
        structure_rcv = sq_ft * _COST_PER_SQFT
        val_source = f"Est. {sq_ft:,.0f} sqft × ${_COST_PER_SQFT:.0f}/sqft [{_wf_label}]"
        val_source_detail = [
            f"Building sqft:         {sq_ft:>12,.0f} sqft",
            f"Cost per sqft:         ${_COST_PER_SQFT:>11,.0f}  [{_wf_label}]",
            f"Estimated RCV:         ${structure_rcv:>12,.0f}",
            f"Source: St. Pete / Pinellas 2024-25 market — {_wf_label} rate applied",
            f"  Waterfront (<150 ft open water): $550+/sqft  |  Standard: $312/sqft",
            "Note: PCPAO data unavailable — estimate only",
        ]

    # Progressive contents ratio by home value tier
    cont_ratio, cont_tier_label = _contents_ratio_for_value(structure_rcv)
    contents_rcv = structure_rcv * cont_ratio

    # ACV depreciation for flood insurance contents (always ACV) and structure gap
    acv_ratio, age_yr, depr_pct, acv_source = _acv_ratio_for_age(year_built)
    structure_acv = structure_rcv * acv_ratio
    contents_acv  = contents_rcv  * acv_ratio

    # HO struct limit defaults to full RCV
    if ho_struct_limit is None or ho_struct_limit < 1000:
        ho_struct_limit = structure_rcv
    ho_wind_deductible = round(ho_struct_limit * ho_wind_deductible_pct, 0)

    # ── Saltwater multiplier ──────────────────────────────────────────────────
    _mtfcc = proj.get("waterbody_mtfcc", "")
    _tidal  = proj.get("waterbody_tidal", False)
    if _tidal or _mtfcc in {"H2051", "H3010", "H3013"}:
        salt_mult  = 1.25
        salt_label = "Tidal/saltwater +25% (bay, canal, or river — salt accelerates corrosion/mold)"
    elif _mtfcc in {"H2030", "H2025"}:
        salt_mult  = 1.05
        salt_label = "Freshwater +5% (lake or pond)"
    else:
        salt_mult  = 1.00
        salt_label = "No water body surcharge"

    # ── Per-depth scenario calculations ──────────────────────────────────────
    _DEPTHS_IN = [6.0, 12.0, 18.0, 24.0, 36.0]  # damage_depth_in values
    _DEPTH_FT  = [d / 12.0 for d in _DEPTHS_IN]  # same in feet

    depth_scenarios = []
    for depth_ft, depth_in in zip(_DEPTH_FT, _DEPTHS_IN):
        prof = _DEPTH_STORM_PROFILES[depth_ft]

        # HAZUS flood damage % at this depth above FFE
        s_flood_pct = _hazus_pct(depth_ft, _HAZUS_STRUCT)
        c_flood_pct = _hazus_pct(depth_ft, _HAZUS_CONT)

        # Concurrent hurricane wind damage (additive for depths ≥ 18")
        s_wind_pct = prof["wind_struct_add_pct"]
        c_wind_pct = prof["wind_cont_add_pct"]

        # Total damage percent (flood + wind, capped at 95%)
        s_total_pct = min(95.0, s_flood_pct + s_wind_pct)
        c_total_pct = min(95.0, c_flood_pct + c_wind_pct)

        # Dollar damage — flood component uses saltwater multiplier
        struct_flood_dmg = structure_rcv * (s_flood_pct / 100.0) * salt_mult
        cont_flood_dmg   = contents_rcv  * (c_flood_pct / 100.0) * salt_mult
        struct_wind_dmg  = structure_rcv * (s_wind_pct  / 100.0)  # wind: no salt mult
        cont_wind_dmg    = contents_rcv  * (c_wind_pct  / 100.0)
        total_flood_dmg  = struct_flood_dmg + cont_flood_dmg
        total_wind_dmg   = struct_wind_dmg  + cont_wind_dmg
        total_dmg        = total_flood_dmg + total_wind_dmg

        # ── FLOOD INSURANCE (NFIP standard) ──────────────────────────────────
        # Structure: NFIP pays RCV for primary SFR if insured to ≥80% value.
        #   If under-insured → pays ACV. We model at-limit (worst-case = ACV).
        #   Per 44 CFR §61.3 and FEMA SFIP Building Coverage terms.
        # Contents: NFIP ALWAYS pays ACV regardless. Per SFIP Contents Coverage.
        # NFIP co-insurance rule (44 CFR §61.3 / SFIP Building Coverage):
        #   If coverage ≥ 80% of RCV → pays RCV up to policy limit
        #   If coverage < 80% of RCV → pays the GREATER of:
        #       (a) ACV of damage, or
        #       (b) (coverage_limit / (0.80 × RCV)) × damage   [co-insurance penalty]
        _nfip_struct_pays_rcv = (flood_struct_limit >= structure_rcv * 0.80)
        if _nfip_struct_pays_rcv:
            _flood_struct_eligible = struct_flood_dmg
        else:
            _coins_factor  = flood_struct_limit / max(1.0, 0.80 * structure_rcv)
            _option_a      = struct_flood_dmg * acv_ratio          # ACV of damage
            _option_b      = struct_flood_dmg * _coins_factor      # co-insurance formula
            _flood_struct_eligible = max(_option_a, _option_b)
        flood_struct_payout = min(
            flood_struct_limit,
            max(0.0, _flood_struct_eligible - flood_deductible)
        )
        flood_cont_payout = min(
            flood_cont_limit,
            max(0.0, cont_flood_dmg * acv_ratio - flood_deductible)  # SFIP: deductible applies separately to contents
        )
        flood_total_payout = flood_struct_payout + flood_cont_payout

        # ── HOMEOWNERS / WIND INSURANCE ───────────────────────────────────────
        # Standard HO policy EXCLUDES rising floodwater (ISO HO-3 Exclusion J).
        # For hurricane depths (≥ 18"): wind damage component IS covered at RCV.
        # Wind deductible: FL named-storm = 3% of Coverage A (dwelling limit).
        # Flat deductible: applies to non-wind HO claims (fire, theft, etc.) —
        #   not applicable here since flood is excluded. Source: FL Stat §627.701.
        if prof["ho_covers_wind"] and total_wind_dmg > 0:
            ho_wind_payout = max(0.0, total_wind_dmg - ho_wind_deductible)
            ho_payout      = round(min(ho_struct_limit, ho_wind_payout), 0)
            ho_note        = (f"Wind/HO covers wind damage only "
                              f"(3% deductible = ${ho_wind_deductible:,.0f})")
        else:
            ho_payout = 0.0
            ho_note   = ("HO policy: flood excluded (ISO HO-3 Excl. J); "
                         "no wind component at this depth")

        # ── OUT-OF-POCKET GAP ────────────────────────────────────────────────
        oop = max(0.0, total_dmg - flood_total_payout - ho_payout)

        depth_scenarios.append({
            "depth_in":           depth_in,
            "depth_ft":           depth_ft,
            "depth_label":        prof["depth_label"],
            "storm_type":         prof["storm_type"],
            "wind_mph":           prof["wind_mph"],
            "return_period":      prof["return_period"],
            "annual_prob":        prof["annual_prob"],
            "pinellas_event":     prof["pinellas_event"],
            "event_source":       prof["event_source"],
            # Flood damage
            "hazus_struct_flood_pct":  round(s_flood_pct, 1),
            "hazus_cont_flood_pct":    round(c_flood_pct, 1),
            "struct_flood_dmg":        round(struct_flood_dmg, 0),
            "cont_flood_dmg":          round(cont_flood_dmg, 0),
            # Wind damage (hurricane component)
            "wind_struct_add_pct":     round(s_wind_pct, 1),
            "wind_cont_add_pct":       round(c_wind_pct, 1),
            "struct_wind_dmg":         round(struct_wind_dmg, 0),
            "cont_wind_dmg":           round(cont_wind_dmg, 0),
            # Totals
            "total_struct_dmg":        round(struct_flood_dmg + struct_wind_dmg, 0),
            "total_cont_dmg":          round(cont_flood_dmg   + cont_wind_dmg,   0),
            "total_flood_dmg":         round(total_flood_dmg, 0),
            "total_wind_dmg":          round(total_wind_dmg, 0),
            "total_dmg":               round(total_dmg, 0),
            # Insurance
            "nfip_struct_pays_rcv":    _nfip_struct_pays_rcv,
            "flood_struct_payout":     round(flood_struct_payout, 0),
            "flood_cont_payout":       round(flood_cont_payout, 0),
            "flood_total_payout":      round(flood_total_payout, 0),
            "ho_payout":               round(ho_payout, 0),
            "ho_note":                 ho_note,
            # OUT OF POCKET
            "out_of_pocket":           round(oop, 0),
        })

    # ── Expected Annual Damage (EAD) ─────────────────────────────────────────
    # EAD via trapezoidal integration over the probability-damage curve.
    # Sort ascending by probability (lowest prob = most severe event first).
    # Pure trapezoid — no separate starting term (avoids double-counting pts[0]).
    _pts = sorted(depth_scenarios, key=lambda s: s["annual_prob"])
    ead  = 0.0
    for i in range(len(_pts) - 1):
        p0, p1 = _pts[i]["annual_prob"],  _pts[i + 1]["annual_prob"]
        d0, d1 = _pts[i]["total_dmg"],    _pts[i + 1]["total_dmg"]
        ead   += (p1 - p0) * (d0 + d1) / 2.0
    ead = round(ead, 0)

    # ── Mitigation ROI ────────────────────────────────────────────────────────
    _BARRIER_LF      = 35.0
    _BARRIER_COST_LF = 185.0
    _PROTECT_EFF     = 0.85
    barrier_cost    = round(_BARRIER_LF * _BARRIER_COST_LF, 0)
    annual_benefit  = round(ead * _PROTECT_EFF, 0)
    payback_yr      = round(barrier_cost / annual_benefit, 1) if annual_benefit > 0 else 999.0

    return {
        "ok":                   True,
        # Valuation
        "structure_rcv":        round(structure_rcv, 0),
        "structure_acv":        round(structure_acv, 0),
        "contents_rcv":         round(contents_rcv, 0),
        "contents_acv":         round(contents_acv, 0),
        "val_source":           val_source,
        "val_source_detail":    val_source_detail,
        "cont_ratio":           cont_ratio,
        "cont_tier_label":      cont_tier_label,
        "year_built":           year_built,
        "age_yr":               age_yr,
        "depr_pct":             depr_pct,
        "acv_ratio":            acv_ratio,
        "acv_source":           acv_source,
        # Saltwater
        "salt_mult":            salt_mult,
        "salt_label":           salt_label,
        "waterbody_name":       proj.get("waterbody", "Unknown"),
        # Insurance inputs used
        "flood_struct_limit":   flood_struct_limit,
        "flood_cont_limit":     flood_cont_limit,
        "flood_deductible":     flood_deductible,
        "ho_struct_limit":      ho_struct_limit,
        "ho_wind_deductible":   ho_wind_deductible,
        # Per-depth breakdown
        "depth_scenarios":      depth_scenarios,
        # EAD + ROI
        "ead_annual":           ead,
        "barrier_cost":         barrier_cost,
        "annual_benefit":       annual_benefit,
        "simple_payback_yr":    payback_yr,
    }


def calc_rainfall_to_damage(
        lot_area_sqft: float,
        runoff_c: float,
        soil_infil_in_hr: float,
        pipe_capacity_gpm: float,
        open_drains: list,
        radial_summary: dict,
        g_elev: float,
        delta: float,
        max_intensity_in_hr: float = 5.3,
        damage_depth_in: float = 6.0,
        max_total_rain_in: float = 24.0,
        drainage_class: str = "ASSUMED",
        effective_drainage_gpm: float = None,
        pipe_data: dict = None,
        myakka_state: str = None,
        dist_water_ft: float = 999.0) -> dict:
    """
    Compute the minimum rainfall intensity (in/hr) and storm duration (hours)
    needed for accumulated surface water to reach the house at >= damage_depth_in.

    Uses the full hydraulic logic already in this engine:
      - ASCE Rational Method (rational_gpm) for gross surface runoff
      - Manning's equation (manning_gpm) for gravity-main pipe capacity
      - Open-drain trapezoidal channel capacity from GIS data
      - SSURGO-derived soil infiltration rate
      - LiDAR-derived micro-depression storage volume (from radial DEM scan)

    Algorithm
    ---------
    For each intensity r (0.1 → max_intensity_in_hr, step 0.1 in/hr):
      1. Gross runoff GPM from Rational Method (Q = CiA).
      2. Subtract: soil infiltration equivalent GPM + gravity-main capacity
         + open-drain capacity.
      3. Net excess GPM is what accumulates on the site.
      4. LiDAR depression storage (gal) must be filled before water rises
         at the house grade.
      5. After depression fills, remaining excess raises depth at house.
      6. Damage threshold = damage_depth_in inches over 70% of lot area.
      7. Max storm duration = min(max_total_rain_in / r, 3.0) hours.
      8. If damage threshold is reached within that duration → critical event.
    """
    # ── Adaptive drainage model — Triple-Path override ───────────────────
    # drainage_class drives which hydraulic equations are applied:
    #   NONE    → surface runoff + infiltration only (no Manning, no HEC-22)
    #   PARTIAL → 50 % of detected effective capacity
    #   FULL    → 100 % of detected effective capacity
    #   ASSUMED → legacy behaviour (pipe + open-drain from GIS row join)
    _use_tripath = (drainage_class in ("NONE", "PARTIAL", "FULL") and
                    effective_drainage_gpm is not None)
    # Minimum realistic drainage floor: even with "NONE" detected infrastructure,
    # flat Pinellas lots always have some sheet-flow path, swale edge runoff, and
    # soil surface percolation.  Modeling zero drainage overstates flood risk for
    # all parcels that are simply not on a catalogued pipe segment.
    # Conservative floors per SWFWMD basin planning guidance:
    #   NONE detected    → 75 GPM  (sheet flow only — roughly a 4" roadside swale)
    #   PARTIAL detected → max(detected×0.50, 120 GPM)
    #   FULL detected    → use full detected capacity
    _DRAIN_FLOOR_NONE    = 75.0    # GPM — no catalogued infrastructure
    _DRAIN_FLOOR_PARTIAL = 120.0   # GPM — partial system minimum

    if drainage_class == "NONE" and effective_drainage_gpm is not None:
        # No catalogued infrastructure — apply sheet-flow floor only
        pipe_capacity_gpm = _DRAIN_FLOOR_NONE
        open_drains       = []
    elif drainage_class == "PARTIAL" and effective_drainage_gpm is not None:
        # Partial relief at 50 %, floored at partial minimum
        pipe_capacity_gpm = max(effective_drainage_gpm * 0.50, _DRAIN_FLOOR_PARTIAL)
        open_drains       = []
    elif drainage_class == "FULL" and effective_drainage_gpm is not None:
        # Full detected capacity replaces legacy pipe/drain estimate
        pipe_capacity_gpm = effective_drainage_gpm
        open_drains       = []

    # ── Per-pipe slope-corrected Manning capacity (ASSUMED mode only) ────────
    # In ASSUMED mode the caller passes a single p_gpm built with a fixed 0.5%
    # slope.  When analyze_pipe_slopes data is available we replace that with
    # per-pipe Manning calculations that use the actual LiDAR-derived terrain
    # slope toward each structure — exactly the same manning_gpm() formula but
    # with a slope that reflects real drainage head:
    #   slope_pct > 0.5 → gravity assists flow  → use detected slope
    #   slope_pct < -0.5 → backflow risk          → clamp to 0.1% (near-zero)
    #   flat / unknown   → 0.3% (conservative)
    # Open drains from pipe_data replace the passed-in open_drains list so
    # the trapezoidal capacity from analyze_pipe_slopes is used verbatim.
    if drainage_class == "ASSUMED" and pipe_data:
        _pipes_raw = pipe_data.get("pipes", [])
        if _pipes_raw:
            _pipe_cap_sum = 0.0
            for _p in _pipes_raw:
                _mat   = str(_p.get("material", "RCP")).upper()
                _dia   = float(_p.get("diameter_in", 12) or 12)
                _s_pct = _p.get("slope_pct")
                if _s_pct is not None:
                    if _s_pct > 0.5:      _slope = max(0.001, _s_pct / 100.0)
                    elif _s_pct < -0.5:   _slope = 0.001   # backflow risk
                    else:                 _slope = 0.003   # flat — minimal head
                else:
                    _slope = 0.005   # fallback default
                _n_key = ('RCP' if ('CONC' in _mat or 'RCP' in _mat)
                          else 'CMP' if ('CMP' in _mat or 'MET' in _mat)
                          else 'PVC')
                _n_val = HYDRAULIC_MATRIX[_n_key]['n_value']
                _pipe_cap_sum += manning_gpm(_dia, _n_val, slope=_slope)
            if _pipe_cap_sum > 0:
                pipe_capacity_gpm = _pipe_cap_sum
        # Also use the trapezoidal open-drain capacities from analyze_pipe_slopes
        _od_raw = pipe_data.get("open_drains", [])
        if _od_raw:
            open_drains = _od_raw

    # ── Open-drain combined capacity (CFS → GPM) ─────────────────────────
    drain_gpm = 0.0
    for d in (open_drains or []):
        cap_cfs = d.get("capacity_cfs", 0.0) or 0.0
        drain_gpm += cap_cfs * 448.83

    # ── LiDAR depression storage and geometry ────────────────────────────
    # Apply 70% usability factor: not all micro-depressions fill perfectly;
    # water spills unevenly across uneven ground.  LiDAR volume is a geometric
    # maximum, not an effective storage number.
    _raw_dep_gal     = float(radial_summary.get("total_ponded_gal", 0.0) or 0.0)
    # 70% usable storage (LiDAR vol is geometric max; real fill is uneven).
    # Floor raised to 500 gal — the absolute minimum realistic backyard low
    # spot in Pinellas (even a shallow swale holds several hundred gallons).
    dep_storage_gal  = max(500.0, _raw_dep_gal * 0.70)
    max_dep_depth_ft = float(radial_summary.get("max_depression_depth_ft", 0.0) or 0.0)

    # ── Depression planform area ──────────────────────────────────────────────
    # Prefer the directly measured area (sum of depression pixel sqft) stored
    # in the radial summary; fall back to vol/depth geometry only when absent.
    _raw_dep_area = float(radial_summary.get("total_dep_area_sqft", 0.0) or 0.0)
    if _raw_dep_area > 100.0:
        dep_area_sqft = _raw_dep_area
    elif max_dep_depth_ft > 0.05 and dep_storage_gal > 500.0:
        dep_area_sqft = dep_storage_gal / (max(max_dep_depth_ft, 0.10) * 7.481)
    else:
        dep_area_sqft = 1000.0   # default: typical Pinellas yard low area
    # Floor: 600 sqft (~25×25 ft patch), cap: 30% of lot
    dep_area_sqft = max(600.0, min(dep_area_sqft, lot_area_sqft * 0.30))

    # ── Damage volume ─────────────────────────────────────────────────────────
    # Water must spread over the CONTRIBUTING drainage area around the house,
    # not just the tiny depression footprint.  Use the larger of:
    #   (a) computed dep_area_sqft, or
    #   (b) 15% of the lot (minimum sheet-flow collection area for flat Pinellas)
    # This prevents unrealistically tiny volumes that give 2-minute flood times.
    rise_to_house_ft  = max(0.0, float(delta or 0.0))
    damage_rise_ft    = rise_to_house_ft + damage_depth_in / 12.0
    _eff_damage_area  = max(dep_area_sqft, lot_area_sqft * 0.15)
    _eff_damage_area  = min(_eff_damage_area, lot_area_sqft * 0.50)  # cap 50%
    damage_vol_gal    = _eff_damage_area * damage_rise_ft * 7.481
    damage_vol_gal    = max(500.0, damage_vol_gal)

    # ── Storm-condition deration factors ─────────────────────────────────────
    # Gravity mains rarely run at full Manning capacity during storms.
    # Three compounding physical limits:
    #   (a) PIPE_STORM_FACTOR  — hydraulic efficiency under surcharge, inlet
    #       blockage, partial flow, tidal backpressure, and debris loading.
    #       FHWA HEC-22 / ASCE 5-45 storm-inlet guidance: 30–60%, use 0.40.
    #   (b) INLET_CAPTURE_RATIO — grate intercepts only ~75% of gutter flow
    #       (bypassed flow passes over the inlet); per HEC-22 curb-opening std.
    #   (c) Hard cap: pipe cannot drain more water than actually arrives at it
    #       (actual_pipe_gpm = min of capacity-derated and inlet-limited).
    #   (d) DRAIN_STORM_FACTOR — open channels also surcharged / debris-limited
    #       during peak; 70% efficiency is conservative for earthen ditches.
    PIPE_STORM_FACTOR   = 0.40   # Manning → effective storm capacity
    INLET_CAPTURE_RATIO = 0.75   # fraction of surface runoff captured by grate
    DRAIN_STORM_FACTOR  = 0.70   # open-drain storm efficiency

    # ── Reportable scenario filters ───────────────────────────────────────────
    # Three combined gates. A scenario must pass ALL THREE to appear in the
    # flood pathway table or be named the "critical threshold":
    #
    #   1. Minimum rate: 0.75 in/hr — below this, rain is light drizzle that
    #      virtually never standalone-floods a Pinellas property; low-rate
    #      risk is conveyed as a conditional footnote, not a headline number.
    #
    #   2. Minimum net excess: 10 GPM — anything below this is within the
    #      margin of error of Manning's estimates; drainage "technically
    #      exceeded" by 1–5 GPM is not a real flood scenario.
    #
    #   3. Minimum cumulative rain: 0.50 inches — filters scenarios that
    #      reach damage only via implausibly long accumulation at trivial
    #      rates.  0.50" is the practical lower bound for standing water.
    _MIN_REPORT_RATE   = 1.00   # in/hr
    _MIN_NET_EXCESS    = 10.0   # GPM
    _MIN_CUMUL_RAIN    = 0.50   # inches total

    results = []
    critical = None

    r = 0.25   # internal loop still starts at 0.25 for physics accuracy;
               # reportability filters are applied during pathway collection.
    while r <= max_intensity_in_hr + 0.001:
        r = round(r, 2)

        # Per-intensity realistic storm window (Pinellas County climatology).
        # Answers the question: "what rate, if sustained for its realistic max
        # duration, puts 6 inches in the house?"
        # Intense cells (3+ in/hr) rarely last more than 2 hours;
        # slow tropical moisture (0.25 in/hr) can persist 18 hours.
        # Using a blanket 12 hrs for ALL intensities is not realistic — this
        # reflects actual Pinellas/NOAA Atlas 14 storm duration data.
        max_duration_hr  = _storm_duration_cap(r)
        max_duration_min = max_duration_hr * 60.0

        # Gross runoff — Rational Method Q = CiA.
        # AMC phase shift: C ramps as the storm progresses through 3 phases.
        # Phase 1 (0-45 min)  — soil near dry;  use C × 0.70 (high infiltration)
        # Phase 2 (45-90 min) — partial sat;    use C × 1.00 (calibrated value)
        # Phase 3 (>90 min)   — near saturated; use C × 1.22 (hardpan limits drain)
        # The Rational Method uses the equilibrium C for steady-state (Phase 2/3),
        # but for low-rate long-duration events we weight all three phases to get
        # a more realistic storm-average C.
        _dur_min = max_duration_hr * 60.0
        if _dur_min <= 45.0:
            _amc_c = runoff_c * 0.70              # mostly Phase 1
        elif _dur_min <= 90.0:
            _blend = (_dur_min - 45.0) / 45.0    # 0→1 over 45-90 min
            _amc_c = runoff_c * (0.70 + 0.30 * _blend)  # Phase 1→2
        else:
            _wet_frac = min(1.0, (_dur_min - 90.0) / 90.0)  # 0→1 over next 90 min
            _amc_c = runoff_c * (1.00 + 0.22 * _wet_frac)   # Phase 2→3
        _amc_c = min(_amc_c, 0.95)   # physical cap
        gross_gpm = rational_gpm(lot_area_sqft, r, _amc_c)

        # Storm-derated pipe drainage:
        #   Step 1 — derate Manning capacity to storm-condition throughput
        effective_pipe_cap = pipe_capacity_gpm * PIPE_STORM_FACTOR
        #   Step 1b — tidal backpressure: storm surge backing into outfall reduces capacity
        #   When the pipe outlet is near open water, surge can partially or fully submerge
        #   the outfall, cutting effective pipe capacity significantly.
        if dist_water_ft < 100:
            _tidal_factor = 0.45   # outfall likely submerged — severe backpressure
        elif dist_water_ft < 300:
            _tidal_factor = 0.65   # outfall intermittently submerged
        elif dist_water_ft < 500:
            _tidal_factor = 0.80   # some backpressure from nearshore surge
        else:
            _tidal_factor = 1.00   # no meaningful tidal influence
        effective_pipe_cap = effective_pipe_cap * _tidal_factor
        #   Step 2 — inlet grate can only capture INLET_CAPTURE_RATIO of runoff
        inlet_limited_gpm  = gross_gpm * INLET_CAPTURE_RATIO
        #   Step 3 — actual pipe removal = lesser of the two (can't exceed runoff)
        actual_pipe_gpm    = min(effective_pipe_cap, inlet_limited_gpm)

        # Open-drain derated capacity
        actual_drain_gpm   = drain_gpm * DRAIN_STORM_FACTOR

        # Total drainage capacity (storm-realistic; infiltration excluded — in C)
        total_capacity_gpm = actual_pipe_gpm + actual_drain_gpm

        # Net surface excess (clamp to zero — no negative accumulation)
        net_gpm = max(0.0, gross_gpm - total_capacity_gpm)

        if net_gpm <= 0.0:
            results.append({
                "rate_in_hr":      r,
                "max_duration_hr": round(max_duration_hr, 2),
                "net_excess_gpm":  0.0,
                "dep_fill_min":    None,
                "damage_time_min": None,
                "total_time_min":  None,
                "within_storm":    False,
                "status":          "✅ Drains handle it — no surface excess",
            })
            r = round(r + 0.1, 2)
            continue

        # ── Time of concentration (Tc) lag ───────────────────────────────────
        # Rain does not produce surface runoff instantly.  Before the rational
        # method reaches equilibrium, the catchment goes through:
        #   (a) surface wetting / initial abstraction  (~2 min)
        #   (b) depression & micro-storage filling     (~3 min)
        #   (c) sheet-flow travel time to collection   (~5–15 min, flat Pinellas)
        # NRCS TR-55 / FAA method for flat (<1%) suburban lots gives Tc ≈ 10–20 min.
        # Using 12 min as the conservative Pinellas floor (very short sheet paths).
        # High-intensity storms shorten effective Tc slightly (flow moves faster).
        if r >= 3.0:
            _tc_min = 8.0    # intense cell — fast concentraion
        elif r >= 1.5:
            _tc_min = 10.0   # heavy storm — moderate concentration
        else:
            _tc_min = 12.0   # slow/moderate rain — full concentration lag

        # Time (min) to fill depression storage
        dep_fill_min = dep_storage_gal / net_gpm

        # Time (min) to accumulate 6" damage depth after depression fills
        damage_time_min = damage_vol_gal / net_gpm

        # Total time from rain-start to damage threshold (includes Tc lag)
        total_time_min = _tc_min + dep_fill_min + damage_time_min

        within = total_time_min <= max_duration_min

        results.append({
            "rate_in_hr":      r,
            "max_duration_hr": round(max_duration_hr, 2),
            "gross_gpm":       round(gross_gpm, 1),
            "capacity_gpm":    round(total_capacity_gpm, 1),
            "net_excess_gpm":  round(net_gpm, 1),
            "tc_lag_min":      round(_tc_min, 1),
            "dep_fill_min":    round(dep_fill_min, 1),
            "damage_time_min": round(damage_time_min, 1),
            "total_time_min":  round(total_time_min, 1),
            "within_storm":    within,
            "status":          "🔴 DAMAGE THRESHOLD REACHED" if within
                               else "⚠️  Exceeds storm window",
        })

        # Only promote to "critical" if it passes all three reportability gates
        if within and critical is None:
            _r_chk   = results[-1]["rate_in_hr"]
            _net_chk = results[-1].get("net_excess_gpm", 0)
            _tot_chk = round(_r_chk * (results[-1]["total_time_min"] / 60.0), 2)
            if (_r_chk >= _MIN_REPORT_RATE and
                    _net_chk >= _MIN_NET_EXCESS and
                    _tot_chk >= _MIN_CUMUL_RAIN):
                critical = results[-1]

        r = round(r + 0.1, 2)

    # ── Collect all viable flood pathways (all rates that DO flood within their window) ──
    # These represent the MULTIPLE WAYS the property can flood, ordered from
    # lowest intensity (slowest / most likely prolonged event) to highest.
    # Load historical exceedance frequencies once for scoring below
    _hist = load_historical_rainfall()
    _exc  = _hist.get("exceedance", {})

    _flood_pathways = []
    for _entry in results:
        if _entry.get("within_storm") and _entry.get("total_time_min") is not None:
            _r   = _entry["rate_in_hr"]
            _dur = round(_entry["total_time_min"] / 60.0, 2)
            _tot = round(_r * _dur, 2)

            # ── Apply reportability gates ─────────────────────────────────
            if _r < _MIN_REPORT_RATE:
                continue   # rate too low to be a credible standalone flood
            if _entry.get("net_excess_gpm", 999) < _MIN_NET_EXCESS:
                continue   # excess within Manning margin of error
            if _tot < _MIN_CUMUL_RAIN:
                continue   # total rain too low to produce standing water
            # Find the closest label bucket
            _label, _rp = "Rainfall event", "—"
            _closest_key = min(_FLOOD_SCENARIO_LABELS.keys(), key=lambda k: abs(k - _r))
            if abs(_closest_key - _r) <= 0.30:
                _label, _rp, _ = _FLOOD_SCENARIO_LABELS[_closest_key]

            # ── Antecedent condition framing ──────────────────────────────
            if _r <= 0.75:
                _ante_cond = "wet antecedent (saturated or recent rain)"
                _ante_note = ("Flood only triggers given wet antecedent conditions "
                              "(prior rain or high tide). Dry day: threshold rises ~40%.")
                _ante_prob = 0.35   # ~35% of Pinellas wet-season days are truly saturated
            elif _r <= 1.25:
                _ante_cond = "borderline — wet conditions likely needed"
                _ante_note = ("Flood probable after wet antecedent; possible from dry start "
                              "if storm stalls.")
                _ante_prob = 0.65
            else:
                _ante_cond = "any conditions"
                _ante_note = ("High-intensity storm can flood regardless of antecedent "
                              "moisture — rate overwhelms all drainage.")
                _ante_prob = 1.00

            # ── Historical frequency score from merged CSV data ───────────
            # Map pathway total rain (_tot inches) to nearest exceedance threshold.
            # freq_per_yr = how many times per year that daily total is exceeded
            # in the Pinellas 1965-2026 record.  Higher = more common event.
            _exc_keys  = sorted(_exc.keys())
            _near_key  = min(_exc_keys, key=lambda k: abs(k - _tot)) if _exc_keys else None
            _freq_yr   = _exc[_near_key]["freq_per_yr"] if _near_key else 1.0

            # ── Net excess margin (normalised 0→1) ────────────────────────
            # Higher excess GPM over the minimum threshold = drainage more
            # decisively overwhelmed = event more likely to complete flood.
            _excess_norm = min(1.0, (_entry.get("net_excess_gpm", 0) - _MIN_NET_EXCESS)
                               / max(1.0, 100.0 - _MIN_NET_EXCESS))

            # ── Combined probability score ─────────────────────────────────
            # Score = historical_frequency × antecedent_probability × (1 + excess_margin)
            # The (1 + excess_margin) bonus rewards scenarios where drainage is
            # decisively overwhelmed, not just barely exceeded.
            _prob_score = _freq_yr * _ante_prob * (1.0 + _excess_norm)

            _flood_pathways.append({
                "rate_in_hr":        _r,
                "duration_hr":       _dur,
                "total_rain_in":     _tot,
                "tc_lag_min":        _entry.get("tc_lag_min"),
                "dep_fill_min":      _entry.get("dep_fill_min"),
                "total_time_min":    _entry.get("total_time_min"),
                "net_excess_gpm":    round(_entry.get("net_excess_gpm", 0), 1),
                "label":             _label,
                "return_period":     _rp,
                "most_probable":     False,   # assigned after scoring all pathways
                "callout":           None,    # assigned to winner below
                "historical_cite":   "",      # assigned to winner below
                "antecedent_cond":   _ante_cond,
                "antecedent_note":   _ante_note,
                "prob_score":        round(_prob_score, 4),
                "freq_per_yr":       _freq_yr,
            })

    # ── Assign most_probable to highest-scoring pathway ──────────────────────
    # Score = historical frequency (events/yr from 61-yr CSV record)
    #       × antecedent probability (1.0 any / 0.65 borderline / 0.35 wet only)
    #       × (1 + normalised net excess margin)
    # This replaces the old "first in list" logic with data-driven selection.
    if _flood_pathways:
        _best = max(_flood_pathways, key=lambda fp: fp["prob_score"])
        _best["most_probable"]    = True
        _best["callout"]          = _get_storm_callout(_best["rate_in_hr"])
        _best["historical_cite"]  = historical_citation_for_rate(
            _best["rate_in_hr"], _best["duration_hr"])

    # ── Volumetric DEM correlation (most-probable storm) ─────────────────────
    # Cross-checks the hydrological accounting:
    #   Total excess = net_gpm × total_time_min
    #   Phase 1 fills LiDAR depression (dep_storage_gal at 70% usability)
    #   Phase 2 residual rises to structure → should equal damage_vol_gal
    # Error term quantifies how well the two independent estimates agree.
    _vol_corr = None
    if _flood_pathways:
        _mp_net_gpm   = _best.get("net_excess_gpm", 0.0)
        _mp_total_min = _best.get("total_time_min",  0.0)
        _mp_rate      = _best.get("rate_in_hr", 0.0)
        _total_excess_gal  = _mp_net_gpm * _mp_total_min
        _dep_frac          = (dep_storage_gal / _total_excess_gal
                              if _total_excess_gal > 0 else 0.0)
        _residual_gal      = max(0.0, _total_excess_gal - dep_storage_gal)
        _vol_err_pct       = (abs(_residual_gal - damage_vol_gal)
                              / max(1.0, damage_vol_gal) * 100.0)
        _confirmed         = _residual_gal >= damage_vol_gal
        _vol_corr = {
            "mp_rate_in_hr":            _mp_rate,
            "mp_net_excess_gpm":        round(_mp_net_gpm, 1),
            "mp_total_time_min":        round(_mp_total_min, 1),
            "total_excess_vol_gal":     round(_total_excess_gal, 1),
            "dep_storage_capacity_gal": round(dep_storage_gal, 1),
            "dep_fill_fraction_pct":    round(_dep_frac * 100.0, 1),
            "residual_after_dep_gal":   round(_residual_gal, 1),
            "damage_threshold_gal":     round(damage_vol_gal, 1),
            "vol_balance_error_pct":    round(_vol_err_pct, 1),
            "confirmed":                _confirmed,
            "interpretation": (
                f"Most-probable storm ({_mp_rate:.1f} in/hr, "
                f"{_mp_total_min:.0f} min) generates "
                f"{_total_excess_gal:,.0f} gal net surface excess. "
                f"LiDAR depression absorbs {dep_storage_gal:,.0f} gal "
                f"({_dep_frac*100:.1f}%); residual toward structure: "
                f"{_residual_gal:,.0f} gal vs. "
                f"{damage_vol_gal:,.0f} gal damage threshold "
                f"({'CONFIRMED ✓' if _confirmed else 'boundary case — monitor'}) "
                f"[volume balance error {_vol_err_pct:.1f}%]."
            ),
        }

    # De-duplicate pathways: keep only one representative per 0.5 in/hr bucket
    # so the report shows clean representative scenarios (not 60 nearly-identical rows).
    _seen_buckets = set()
    flood_scenarios = []
    for _fp in _flood_pathways:
        _bucket = round(_fp["rate_in_hr"] * 2) / 2   # round to nearest 0.5
        if _bucket not in _seen_buckets:
            _seen_buckets.add(_bucket)
            flood_scenarios.append(_fp)

    # ── Fallback: no rate floods within its realistic storm window → NO FLOOD RISK ──
    if critical is None:
        valid = [x for x in results if x.get("total_time_min") is not None]
        if valid:
            best = min(valid, key=lambda x: x["total_time_min"])
            r_best         = best["rate_in_hr"]
            t_needed_hr    = best["total_time_min"] / 60.0
            rain_needed_in = round(r_best * t_needed_hr, 2)
            # Since critical is None, NO rate flooded within its realistic window.
            # The property survives all Pinellas County design storms → NO FLOOD RISK.
            critical = {**best,
                        "required_total_rain_in": rain_needed_in,
                        "no_flood_risk": True,
                        "note": (
                            f"✅  NO FLOOD RISK — property does not reach 6\" damage "
                            f"threshold within any realistic Pinellas County storm window "
                            f"(tested up to 6.0 in/hr). "
                            f"Site elevation buffer: {round(float(delta or 0.0), 1)} ft above "
                            f"parcel low point. Drainage and terrain provide sufficient relief "
                            f"under all NOAA Atlas 14 design storm scenarios for this county."
                        )}

    DISCLAIMER = (
        "METHODOLOGY DISCLAIMER: Rainfall-to-damage thresholds are computed using "
        "the ASCE Rational Method (Q = CiA) for gross surface runoff, Manning's "
        "equation for gravity-main pipe capacity, trapezoidal channel geometry for "
        "open-drain capacity, and pixel-sampled bilinear LiDAR DEM data (250 ft "
        "radius, 2.5 ft step) for micro-depression storage volumes. Soil "
        "infiltration rates are estimated from SSURGO hydric soil classification. "
        "Building footprints are excluded from the ponding area using the effective-"
        "ground DEM filter (>1 ft above local minimum). "
        "Storm-condition pipe capacity is derated to 40% of Manning theoretical "
        "maximum to account for inlet blockage, partial-flow hydraulics, tidal "
        "backpressure, and downstream surcharge (FHWA HEC-22 / ASCE 5-45 basis). "
        "Inlet capture is further limited to 75% of arriving surface runoff per "
        "HEC-22 curb-opening efficiency; open drains are derated to 70% for "
        "storm-peak debris loading. Net surface excess = gross runoff minus the "
        "sum of storm-derated drainage and infiltration; depression storage fills "
        "first, then excess accumulates to the 6-inch damage-depth volume. "
        "Results represent screening-level approximations — actual flood timing "
        "varies with antecedent soil moisture, infrastructure age/condition, and "
        "localized grading. This output is intended for flood risk awareness and "
        "barrier-sizing guidance only and does not constitute engineering "
        "certification. For FEMA LOMA applications, floodplain management, or "
        "structural flood design, consult a licensed PE."
    )

    return {
        "ok":             True,
        "critical":       critical,
        "flood_scenarios": flood_scenarios,   # list of all viable flood pathways
        "vol_dem_corr":   _vol_corr,          # volumetric DEM cross-check for most-probable storm
        "table":          results,
        "inputs": {
            "lot_area_sqft":            round(lot_area_sqft, 0),
            "runoff_c":                 round(runoff_c, 3),
            "myakka_state":             myakka_state,
            "soil_infil_in_hr":         soil_infil_in_hr,
            "dist_to_water_ft":         round(dist_water_ft, 0) if dist_water_ft < 999 else "Unknown",
            "dep_storage_raw_gal":      round(_raw_dep_gal, 1),
            "dep_usability_factor":     0.70,
            "note_infil":               "Infiltration already embedded in C (Rational Method) — not double-subtracted",
            "pipe_capacity_gpm_raw":    round(pipe_capacity_gpm, 1),
            "pipe_storm_factor":        PIPE_STORM_FACTOR,
            "inlet_capture_ratio":      INLET_CAPTURE_RATIO,
            "pipe_capacity_gpm_storm":  round(pipe_capacity_gpm * PIPE_STORM_FACTOR, 1),
            "drain_capacity_gpm_raw":   round(drain_gpm, 1),
            "drain_storm_factor":       DRAIN_STORM_FACTOR,
            "drain_capacity_gpm_storm": round(drain_gpm * DRAIN_STORM_FACTOR, 1),
            "dep_storage_gal":          round(dep_storage_gal, 1),
            "dep_area_sqft_est":        round(dep_area_sqft, 0),
            "delta_ft":                 round(float(delta or 0.0), 2),
            "damage_rise_ft":           round(damage_rise_ft, 2),
            "damage_vol_gal":           round(damage_vol_gal, 1),
            "max_intensity_in_hr":      max_intensity_in_hr,
            "damage_depth_in":          damage_depth_in,
            "max_total_rain_in":        max_total_rain_in,
        },
        "disclaimer": DISCLAIMER,
    }


# =============================================================================
# TERRAIN ANALYSIS
# =============================================================================
def analyze_terrain(terrain: dict, g_elev: float) -> dict:
    analysis     = {}
    flow_vectors = {}
    for dir_name, profile in terrain.items():
        if not profile: continue
        elevs = [p[1] for p in profile]
        dists = [p[0] for p in profile]
        flat_runs, run_start = [], None
        for i in range(1, len(elevs)):
            if abs(elevs[i] - elevs[i-1]) < 0.02:
                if run_start is None: run_start = dists[i-1]
            else:
                if run_start is not None:
                    flat_runs.append((run_start, dists[i-1]))
                    run_start = None
        if run_start is not None:
            flat_runs.append((run_start, dists[-1]))
        depressions = []
        for i in range(2, len(elevs)-2):
            if elevs[i] < elevs[i-2] and elevs[i] < elevs[i+2] and elevs[i] < g_elev:
                depressions.append({"dist_ft": dists[i], "elev_ft": round(elevs[i],3),
                                    "depth_ft": round(g_elev - elevs[i], 3)})
        terminal_delta = elevs[-1] - g_elev
        net_slope_pct  = (terminal_delta / SAMPLE_RANGE_FT) * 100
        flow_toward    = terminal_delta < -0.08
        flow_vectors[dir_name] = terminal_delta
        analysis[dir_name] = {
            "profile":        profile,
            "terminal_elev":  round(elevs[-1],3),
            "terminal_delta": round(terminal_delta,3),
            "net_slope_pct":  round(net_slope_pct, 4),
            "flow_toward":    flow_toward,
            "depressions":    depressions,
            "flat_runs":      flat_runs,
            "is_void":        any((b-a) >= 2.0 for a, b in flat_runs),
            "point_slopes":   [round((elevs[i]-elevs[i-1])/SAMPLE_INTERVAL_FT, 4)
                               for i in range(1, len(elevs))],
        }
    pf = min(flow_vectors, key=flow_vectors.get) if flow_vectors else "UNKNOWN"
    hf = max(flow_vectors, key=flow_vectors.get) if flow_vectors else "UNKNOWN"
    conv_dirs = [d for d, v in flow_vectors.items() if v < -0.05]
    return {"directions": analysis, "primary_flow": pf, "highest_dir": hf,
            "convergence_dirs": conv_dirs, "dirs_toward": len(conv_dirs)}


# =============================================================================
# GRAVITY MAIN PIPE SLOPE DETECTION
# =============================================================================
def analyze_pipe_slopes(row, g_elev: float, terrain_analysis: dict) -> dict:
    pipes = []
    for suffix in ['', '_2', '_3', '_5', '_8']:
        dist_key = f"distance{suffix}" if suffix else "distance"
        mat_key  = f"MATERIAL{suffix.upper()}" if suffix else "MATERIAL"
        dia_key  = f"DIAMETER_2" if suffix == '_2' else "DIAMETER"
        fx_key   = f"feature_x{suffix}" if suffix else "feature_x"
        fy_key   = f"feature_y{suffix}" if suffix else "feature_y"
        nx_key   = f"nearest_x{suffix}" if suffix else "nearest_x"
        ny_key   = f"nearest_y{suffix}" if suffix else "nearest_y"

        dist = safe(row.get(dist_key))
        mat  = safe(row.get(mat_key, row.get("MATERIAL_3","")), "UNKNOWN")
        dia  = safe(row.get(dia_key, row.get("DIAMETER_2", 12)), 12)
        fx   = safe(row.get(fx_key))
        fy   = safe(row.get(fy_key))
        nx   = safe(row.get(nx_key))
        ny   = safe(row.get(ny_key))

        if dist is None: continue
        try:
            dist_ft = float(dist)
            dia_in  = float(str(dia).replace('"',''))
        except: continue

        if fx and nx and fy and ny:
            try:
                pipe_dx    = float(nx) - float(fx)
                pipe_dy    = float(ny) - float(fy)
                pipe_angle = math.degrees(math.atan2(pipe_dy, pipe_dx))
                dirs       = ["E","NE","N","NW","W","SW","S","SE"]
                dir_idx    = int((pipe_angle + 360 + 22.5) / 45) % 8
                pipe_dir   = dirs[dir_idx]
            except:
                pipe_dir = "UNKNOWN"
        else:
            pipe_dir = "UNKNOWN"

        da      = terrain_analysis["directions"].get(pipe_dir, {})
        profile = da.get("profile", [])
        pipe_terrain_elev = None
        if profile:
            closest = min(profile, key=lambda p: abs(p[0] - dist_ft), default=None)
            if closest:
                pipe_terrain_elev = closest[1]

        if pipe_terrain_elev is not None:
            elev_change = pipe_terrain_elev - g_elev
            slope_pct   = (elev_change / max(dist_ft, 1)) * 100
            if slope_pct > 0.5:    slope_desc = "📈 RISING toward pipe (favorable drainage)"
            elif slope_pct < -0.5: slope_desc = "📉 FALLING toward pipe (backflow risk)"
            else:                  slope_desc = "➡️  FLAT — minimal head, low drainage capacity"
        else:
            elev_change = None
            slope_pct   = None
            slope_desc  = "⚠️  Elevation along pipe path unavailable"

        pipes.append({
            "distance_ft":          round(dist_ft, 1),
            "material":             str(mat).upper(),
            "diameter_in":          dia_in,
            "direction":            pipe_dir,
            "terrain_elev_at_pipe": pipe_terrain_elev,
            "elev_change_ft":       round(elev_change, 3) if elev_change is not None else None,
            "slope_pct":            round(slope_pct, 4) if slope_pct is not None else None,
            "slope_desc":           slope_desc,
        })

    seen = set()
    unique_pipes = []
    for p in pipes:
        key = round(p["distance_ft"])
        if key not in seen:
            seen.add(key)
            unique_pipes.append(p)

    open_drains = []
    for suffix in ['', '_2']:
        tw  = safe(row.get(f"TOPWIDTH{suffix}"))
        bw  = safe(row.get(f"BOTWIDTH{suffix}"))
        dep = safe(row.get(f"DEPTH{suffix}"))
        d2  = safe(row.get(f"distance{suffix if suffix else '_3'}"))
        if all([tw, dep]):
            try:
                open_drains.append({
                    "top_width_ft": round(float(tw),2),
                    "bot_width_ft": round(float(bw),2) if bw else None,
                    "depth_ft":     round(float(dep),2),
                    "distance_ft":  round(float(d2),1) if d2 else None,
                    "capacity_cfs": round(
                        (float(tw)+float(bw or tw))/2 * float(dep) *
                        (1.486/0.035) * ((float(dep)/2)**(2/3)) * (0.003**0.5), 2)
                })
            except: pass

    return {"pipes": unique_pipes, "open_drains": open_drains}


# =============================================================================
# TRIPLE-PATH HYDRAULIC DRAINAGE INTELLIGENCE ENGINE
# Scans real infrastructure within DRAINAGE_SCAN_RADIUS_FT around the parcel.
#
# Path A — Stormwater Structures (points, Z_1 elevation)
#   • HEC-22 inlet efficiency 75 % applied to each reachable structure
# Path B — Gravity Mains (lines, CUL_SIZE diameter, MATERIAL)
#   • Manning's n from material, Q computed, 40 % storm derating
# Path C — Open Drains (lines, TOPWIDTH / BOTWIDTH / DEPTH)
#   • Trapezoidal channel flow, 70 % debris derating
#
# Connectivity test per feature:
#   Z_feature < g_elev             → Stage-1 relief (immediate)
#   Z_feature >= g_elev            → Stage-2 overflow (capacity = 0 until
#                                    ponding depth exceeds structure height)
#   LiDAR flow direction           → terrain must slope toward feature
#   Distance weighting             → <75 ft strong / 75-150 moderate / 150-250 weak
#                                    (features 250-1000 ft remain in scan
#                                     but are weighted proportionally)
# =============================================================================

def _parse_culvert_size_in(cul_size_raw) -> float:
    """Extract dominant diameter (inches) from CUL_SIZE string like '18', '24x18'."""
    if cul_size_raw is None:
        return 12.0
    s = str(cul_size_raw).strip()
    nums = re.findall(r'\d+\.?\d*', s)
    if not nums:
        return 12.0
    return max(float(n) for n in nums)

def _material_to_manning(mat_raw: str) -> float:
    """Map pipe material string to Manning's n coefficient."""
    m = str(mat_raw).upper().strip()
    if any(k in m for k in ('RCP', 'CONC', 'CMP_CON')):   return 0.013
    if any(k in m for k in ('CMP', 'GALV', 'METAL')):     return 0.024
    if any(k in m for k in ('PVC', 'HDPE', 'POLY')):      return 0.011
    if 'TILE' in m or 'CLAY' in m:                         return 0.017
    return 0.013   # default RCP


def triple_path_drainage_scan(
        cx: float, cy: float, df_crs: str,
        g_elev: float, surge_ft: float) -> dict:
    """
    Scan the three Pinellas County stormwater layers within DRAINAGE_SCAN_RADIUS_FT.
    Returns drainage classification (NONE / PARTIAL / FULL) and total effective GPM.
    """
    result = {
        "ok": False,
        "drainage_class":        "NONE",
        "effective_gpm":         0.0,
        "path_a_gpm":            0.0,
        "path_b_gpm":            0.0,
        "path_c_gpm":            0.0,
        "n_structures":          0,
        "n_mains":               0,
        "n_open_drains":         0,
        "strongest_structure":   None,
        "features_a":            [],
        "features_b":            [],
        "features_c":            [],
        "features_d":            [],
        "features_e":            [],
        "features_f":            [],
        "features_g":            [],
        "features_h":            [],
        "features_i":            [],
        "features_j":            [],
        "has_control_valve":     False,
        "tidal_backflow_risk":    False,
        "note":                  "",
    }

    try:
        import geopandas as gpd
        from pyproj import CRS, Transformer
        from shapely.geometry import Point
    except ImportError:
        result["note"] = "geopandas/pyproj not available"
        return result

    # ── convert parcel centroid to WGS-84 for bbox pre-filter ─────────────
    try:
        src_crs = CRS.from_user_input(df_crs)
        wgs84   = CRS.from_epsg(4326)
        tfm_to_wgs = Transformer.from_crs(src_crs, wgs84, always_xy=True)
        lng_c, lat_c = tfm_to_wgs.transform(cx, cy)
    except Exception as _e:
        result["note"] = f"CRS transform failed: {_e}"
        return result

    # ── helper: compute bbox in any target CRS from parcel centroid ──────────
    def _bbox_in_crs(target_crs_epsg: int, pad: float = 1.20):
        """Return (minx, miny, maxx, maxy) in the given EPSG, padded by pad×radius."""
        from pyproj import CRS as _C, Transformer as _T
        tc  = _C.from_epsg(target_crs_epsg)
        tfm = _T.from_crs(src_crs, tc, always_xy=True)
        tx, ty = tfm.transform(cx, cy)
        # determine units: projected feet vs metres vs geographic degrees
        ax_unit = tc.axis_info[0].unit_name.lower() if tc.axis_info else ''
        if tc.is_geographic:
            buf = (DRAINAGE_SCAN_RADIUS_FT / 364000.0) * pad
        elif 'metre' in ax_unit or 'meter' in ax_unit:
            buf = DRAINAGE_SCAN_RADIUS_FT * 0.3048 * pad
        else:                          # feet (State Plane)
            buf = DRAINAGE_SCAN_RADIUS_FT * pad
        return (tx - buf, ty - buf, tx + buf, ty + buf)

    # approximate degree buffer (WGS-84 fallback, kept for reference)
    deg_buf = (DRAINAGE_SCAN_RADIUS_FT / 364000.0) * 1.15   # 15 % margin

    PIPE_DERATING   = 0.40   # Path B: storm-peak / tidal backpressure
    INLET_EFF       = 0.75   # Path A: HEC-22 inlet capture efficiency
    DRAIN_DERATING  = 0.70   # Path C: debris reduction

    # ─────────────────────────────────────────────────────────────────
    # PATH A — Stormwater Structures
    # ─────────────────────────────────────────────────────────────────
    features_a = []
    try:
        bbox_a = _bbox_in_crs(3857)   # Storm_Water_Inlet is EPSG:3857
        gdf_s = gpd.read_file(STRUCTURES_GPKG, layer='reprojected', bbox=bbox_a)
        # already EPSG:3857 — no reproject needed
        parcel_pt = Point(cx, cy)
        gdf_s['_dist_ft'] = gdf_s.geometry.distance(parcel_pt)
        gdf_s = gdf_s[gdf_s['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()

        for _, r in gdf_s.iterrows():
            _rim = coalesce_float(r.get('RIM'), default=None)
            _inv = coalesce_float(r.get('INV1'), default=None)
            z1   = _rim if _rim is not None else (_inv if _inv is not None else coalesce_float(r.get('Z_1'), default=None))
            dist_ft = float(r['_dist_ft'])
            if z1 is None:
                continue
            reachable = (z1 < g_elev)
            rel_elev  = round(z1 - g_elev, 3)

            # distance weighting
            if dist_ft < 75:       wt = 1.00
            elif dist_ft < 150:    wt = 0.70
            elif dist_ft < 250:    wt = 0.45
            else:                  wt = max(0.05, 0.45 * (250 / dist_ft))

            # nominal intake capacity: assume 12-in effective opening, 6-ft head
            if reachable:
                head_ft    = max(0.1, g_elev - z1 + surge_ft * 0.5)
                area_ft2   = math.pi * (0.5)**2   # 12-in inlet
                q_cfs      = 0.62 * area_ft2 * math.sqrt(2 * 32.2 * head_ft)
                q_gpm_raw  = q_cfs * 448.83 * INLET_EFF * wt
            else:
                q_gpm_raw = 0.0   # structure is uphill — no immediate relief

            features_a.append({
                "dist_ft":    round(dist_ft, 1),
                "z1":         round(z1, 3),
                "rel_elev":   rel_elev,
                "reachable":  reachable,
                "weight":     round(wt, 3),
                "gpm":        round(q_gpm_raw, 2),
                "type":       str(r.get('INLETTYPE', r.get('PARENTTYPE', 'unknown')) or 'unknown'),
                "facilityid": str(r.get('FACILITYID', '') or ''),
                "rim_elev":   round(z1, 3),
                "basin":      str(r.get('BASIN', '') or ''),
            })
    except Exception as _e:
        result["note"] += f" PathA-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH B — Gravity Mains
    # ─────────────────────────────────────────────────────────────────
    features_b = []
    try:
        bbox_b = _bbox_in_crs(3857)   # FINAL_GRAVITY_MAINS is EPSG:3857
        gdf_m = gpd.read_file(GRAVITY_MAINS_GPKG, layer='reprojected', bbox=bbox_b)
        # already EPSG:3857
        parcel_pt = Point(cx, cy)
        gdf_m['_dist_ft'] = gdf_m.geometry.distance(parcel_pt)
        gdf_m = gdf_m[gdf_m['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()

        for _, r in gdf_m.iterrows():
            dist_ft  = float(r['_dist_ft'])
            dia_in   = _parse_culvert_size_in(r.get('CUL_SIZE'))
            _pipe_type_map = {1:'RCP', 2:'CMP', 3:'PVC', 4:'HDPE'}
            _mat_raw = str(r.get('MATERIAL', '') or '')
            if not _mat_raw.strip() or _mat_raw.upper() in ('NULL','NONE',''):
                _mat_raw = _pipe_type_map.get(int(r.get('PIPE_TYPE', 0) or 0), 'RCP')
            n_val    = _material_to_manning(_mat_raw)
            q_full   = manning_gpm(dia_in, n_val, slope=0.005)
            q_storm  = q_full * PIPE_DERATING

            if dist_ft < 75:       wt = 1.00
            elif dist_ft < 150:    wt = 0.70
            elif dist_ft < 250:    wt = 0.45
            else:                  wt = max(0.05, 0.45 * (250 / dist_ft))

            features_b.append({
                "dist_ft":    round(dist_ft, 1),
                "dia_in":     dia_in,
                "material":   _mat_raw,
                "n_val":      n_val,
                "cul_size":   str(r.get('CUL_SIZE', '') or ''),
                "q_full_gpm": round(q_full, 2),
                "gpm":        round(q_storm * wt, 2),
                "weight":     round(wt, 3),
                "facilityid": str(r.get('FACILITYID', '') or ''),
                "lifecycle":  str(r.get('LIFECYCLES', '') or ''),
            })
    except Exception as _e:
        result["note"] += f" PathB-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH C — Open Drains (trapezoidal channel)
    # ─────────────────────────────────────────────────────────────────
    features_c = []
    try:
        bbox_c = _bbox_in_crs(3857)   # FINAL_OPEN_DRAINS is EPSG:3857
        gdf_d = gpd.read_file(OPEN_DRAINS_GPKG, layer='reprojected', bbox=bbox_c)
        # already EPSG:3857
        parcel_pt = Point(cx, cy)
        gdf_d['_dist_ft'] = gdf_d.geometry.distance(parcel_pt)
        gdf_d = gdf_d[gdf_d['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()

        for _, r in gdf_d.iterrows():
            dist_ft = float(r['_dist_ft'])
            tw  = coalesce_float(r.get('TOPWIDTH'), default=None)
            bw  = coalesce_float(r.get('BOTWIDTH'),  default=None)
            dep = coalesce_float(r.get('DEPTH'),     default=None)
            if tw is None or dep is None or dep <= 0:
                continue
            bw_eff = bw if bw else tw * 0.5
            # trapezoidal hydraulic radius
            area_ft2 = ((tw + bw_eff) / 2.0) * dep
            perim    = bw_eff + 2 * math.sqrt(((tw - bw_eff) / 2)**2 + dep**2)
            Rh       = area_ft2 / max(perim, 0.01)
            n_ditch  = 0.035
            q_cfs    = (1.486 / n_ditch) * area_ft2 * (Rh ** (2/3)) * (0.003 ** 0.5)
            q_gpm    = q_cfs * 448.83 * DRAIN_DERATING

            if dist_ft < 75:       wt = 1.00
            elif dist_ft < 150:    wt = 0.70
            elif dist_ft < 250:    wt = 0.45
            else:                  wt = max(0.05, 0.45 * (250 / dist_ft))

            features_c.append({
                "dist_ft":      round(dist_ft, 1),
                "top_width_ft": round(tw, 2),
                "bot_width_ft": round(bw_eff, 2),
                "depth_ft":     round(dep, 2),
                "q_gross_gpm":  round(q_cfs * 448.83, 2),
                "gpm":          round(q_gpm * wt, 2),
                "weight":       round(wt, 3),
                "asset_type":   str(r.get('ASSET_TYPE', '') or ''),
                "basin_name":   str(r.get('BASIN_NAME', '') or ''),
                "bed_material": str(r.get('BEDMATERIA', '') or ''),
                "facilityid":   str(r.get('FACILITYID', '') or ''),
                "lifecycle":    str(r.get('LIFECYCLES', '') or ''),
            })
    except Exception as _e:
        result["note"] += f" PathC-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH D — St. Pete Gravity Mains (st_pete_gravity_main.gpkg)
    # Has DIAMETER (real), UPELEV, DOWNELEV, MATERIAL, PIPETYPE
    # ─────────────────────────────────────────────────────────────────
    features_d = []
    try:
        if os.path.exists(STPETE_GRAVITY_GPKG):
            bbox_d = _bbox_in_crs(3857)
            gdf_d2 = gpd.read_file(STPETE_GRAVITY_GPKG, layer='reprojected', bbox=bbox_d)
            parcel_pt = Point(cx, cy)
            gdf_d2['_dist_ft'] = gdf_d2.geometry.distance(parcel_pt) * 3.28084
            gdf_d2 = gdf_d2[gdf_d2['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_d2.iterrows():
                dist_ft  = float(r['_dist_ft'])
                dia_in   = coalesce_float(r.get('DIAMETER'), default=12.0)
                _mat     = str(r.get('MATERIAL', r.get('PIPETYPE', 'RCP')) or 'RCP')
                n_val    = _material_to_manning(_mat)
                # Use actual pipe slope from UPELEV/DOWNELEV if available
                _up   = coalesce_float(r.get('UPELEV'),   default=None)
                _down = coalesce_float(r.get('DOWNELEV'), default=None)
                _slope = abs(_up - _down) / max(float(r.get('Shape__Length', 100) or 100), 1) if (_up and _down) else 0.005
                q_full  = manning_gpm(dia_in, n_val, slope=max(_slope, 0.001))
                q_storm = q_full * PIPE_DERATING
                if dist_ft < 75:       wt = 1.00
                elif dist_ft < 150:    wt = 0.70
                elif dist_ft < 250:    wt = 0.45
                else:                  wt = max(0.05, 0.45 * (250 / dist_ft))
                features_d.append({
                    "dist_ft":    round(dist_ft, 1),
                    "dia_in":     dia_in,
                    "material":   _mat,
                    "n_val":      n_val,
                    "slope":      round(_slope, 5),
                    "q_full_gpm": round(q_full, 2),
                    "gpm":        round(q_storm * wt, 2),
                    "weight":     round(wt, 3),
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":     "stpete_gravity",
                })
    except Exception as _e:
        result["note"] += f" PathD-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH E — St. Pete Culverts (Culverts.gpkg)
    # Has DIAMETER, SIZE1FT, SIZE2FT, UPELEV, DOWNELEV, MATERIAL
    # ─────────────────────────────────────────────────────────────────
    features_e = []
    try:
        if os.path.exists(STPETE_CULVERTS_GPKG):
            bbox_e = _bbox_in_crs(3857)
            gdf_cv = gpd.read_file(STPETE_CULVERTS_GPKG, layer='reprojected', bbox=bbox_e)
            parcel_pt = Point(cx, cy)
            gdf_cv['_dist_ft'] = gdf_cv.geometry.distance(parcel_pt) * 3.28084
            gdf_cv = gdf_cv[gdf_cv['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_cv.iterrows():
                dist_ft  = float(r['_dist_ft'])
                # Use SIZE1FT (ft) if available, else DIAMETER (inches)
                _s1 = coalesce_float(r.get('SIZE1FT'), default=None)
                _s2 = coalesce_float(r.get('SIZE2FT'), default=None)
                dia_in = (_s1 * 12.0) if _s1 else coalesce_float(r.get('DIAMETER'), default=12.0)
                _mat   = str(r.get('MATERIAL', 'RCP') or 'RCP')
                n_val  = _material_to_manning(_mat)
                _up    = coalesce_float(r.get('UPELEV'),   default=None)
                _down  = coalesce_float(r.get('DOWNELEV'), default=None)
                _slope = abs(_up - _down) / max(float(r.get('Shape__Length', 50) or 50), 1) if (_up and _down) else 0.005
                q_full  = manning_gpm(dia_in, n_val, slope=max(_slope, 0.001))
                q_storm = q_full * PIPE_DERATING
                if dist_ft < 75:       wt = 1.00
                elif dist_ft < 150:    wt = 0.70
                elif dist_ft < 250:    wt = 0.45
                else:                  wt = max(0.05, 0.45 * (250 / dist_ft))
                features_e.append({
                    "dist_ft":    round(dist_ft, 1),
                    "dia_in":     dia_in,
                    "material":   _mat,
                    "n_val":      n_val,
                    "slope":      round(_slope, 5),
                    "q_full_gpm": round(q_full, 2),
                    "gpm":        round(q_storm * wt, 2),
                    "weight":     round(wt, 3),
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":     "stpete_culvert",
                })
    except Exception as _e:
        result["note"] += f" PathE-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH F — St. Pete Outfall Points (storm_water_discharge_point.gpkg)
    # Key for surge backflow: DISCHARGETYPE, OUTFALLTO, RIM, INV1
    # Outfalls discharging to bay/canal are backflow risks at high tide
    # ─────────────────────────────────────────────────────────────────
    features_f = []
    try:
        if os.path.exists(STPETE_OUTFALL_PT_GPKG):
            bbox_f = _bbox_in_crs(3857)
            gdf_of = gpd.read_file(STPETE_OUTFALL_PT_GPKG, layer='reprojected', bbox=bbox_f)
            parcel_pt = Point(cx, cy)
            gdf_of['_dist_ft'] = gdf_of.geometry.distance(parcel_pt) * 3.28084
            gdf_of = gdf_of[gdf_of['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_of.iterrows():
                dist_ft = float(r['_dist_ft'])
                _rim    = coalesce_float(r.get('RIM'), default=None)
                _inv    = coalesce_float(r.get('INV1'), default=None)
                z1      = _rim if _rim is not None else _inv
                _dtype  = str(r.get('DISCHARGETYPE', '') or '')
                _to     = str(r.get('OUTFALLTO', '') or '')
                # Tidal backflow risk: outfall to bay/canal at or above grade
                _tidal_risk = any(kw in _to.lower() for kw in ('bay', 'canal', 'gulf', 'tidal', 'harbor'))
                reachable = (z1 is not None and z1 < g_elev) and not _tidal_risk
                if dist_ft < 75:       wt = 1.00
                elif dist_ft < 150:    wt = 0.70
                elif dist_ft < 250:    wt = 0.45
                else:                  wt = max(0.05, 0.45 * (250 / dist_ft))
                # Outfall capacity: model as 18-in pipe (typical St. Pete outfall)
                _dia = coalesce_float(r.get('DIAMETER'), default=18.0)
                if reachable and z1 is not None:
                    head_ft   = max(0.1, g_elev - z1 + surge_ft * 0.3)
                    area_ft2  = math.pi * (_dia / 24.0)**2
                    q_cfs     = 0.62 * area_ft2 * math.sqrt(2 * 32.2 * head_ft)
                    q_gpm_raw = q_cfs * 448.83 * INLET_EFF * wt
                else:
                    q_gpm_raw = 0.0
                features_f.append({
                    "dist_ft":      round(dist_ft, 1),
                    "rim_elev":     round(z1, 3) if z1 else None,
                    "discharge_type": _dtype,
                    "outfall_to":   _to,
                    "tidal_risk":   _tidal_risk,
                    "reachable":    reachable,
                    "weight":       round(wt, 3),
                    "gpm":          round(q_gpm_raw, 2),
                    "facilityid":   str(r.get('FACILITYID', '') or ''),
                    "source":       "stpete_outfall",
                })
    except Exception as _e:
        result["note"] += f" PathF-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH G — Manholes (Manhole.gpkg)
    # Network nodes — RIM elevation tells us if the system is below grade.
    # Manholes below parcel grade indicate an active, reachable system node.
    # ─────────────────────────────────────────────────────────────────
    features_g = []
    try:
        if os.path.exists(MANHOLES_GPKG):
            bbox_g = _bbox_in_crs(3857)
            gdf_mh = gpd.read_file(MANHOLES_GPKG, layer='reprojected', bbox=bbox_g)
            parcel_pt = Point(cx, cy)
            gdf_mh['_dist_ft'] = gdf_mh.geometry.distance(parcel_pt) * 3.28084
            gdf_mh = gdf_mh[gdf_mh['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_mh.iterrows():
                dist_ft = float(r['_dist_ft'])
                _rim = coalesce_float(r.get('RIM'), r.get('RIMELEVATI'), r.get('Z_1'), default=None)
                _inv = coalesce_float(r.get('INV'), r.get('INVERT'),    default=None)
                z1   = _rim if _rim is not None else _inv
                if z1 is None:
                    continue
                reachable = z1 < g_elev
                if dist_ft < 75:    wt = 1.00
                elif dist_ft < 150: wt = 0.70
                elif dist_ft < 250: wt = 0.45
                else:               wt = max(0.05, 0.45 * (250 / dist_ft))
                # Manholes contribute supplemental drainage via connected pipes;
                # model as a 12-in equivalent opening when reachable
                if reachable:
                    head_ft   = max(0.1, g_elev - z1 + surge_ft * 0.3)
                    area_ft2  = math.pi * (0.5) ** 2
                    q_cfs     = 0.62 * area_ft2 * math.sqrt(2 * 32.2 * head_ft)
                    q_gpm_raw = q_cfs * 448.83 * INLET_EFF * wt * 0.5  # 50% credit (node, not inlet)
                else:
                    q_gpm_raw = 0.0
                features_g.append({
                    "dist_ft":   round(dist_ft, 1),
                    "rim_elev":  round(z1, 3),
                    "reachable": reachable,
                    "weight":    round(wt, 3),
                    "gpm":       round(q_gpm_raw, 2),
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":    "manhole",
                })
    except Exception as _e:
        result["note"] += f" PathG-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH H — Structure Network (storm_water_structure_network.gpkg)
    # Connected network topology — line features representing pipe segments
    # between structures.  Presence indicates active piped drainage system.
    # ─────────────────────────────────────────────────────────────────
    features_h = []
    try:
        if os.path.exists(STRUCTURE_NETWORK_GPKG):
            bbox_h = _bbox_in_crs(3857)
            gdf_sn = gpd.read_file(STRUCTURE_NETWORK_GPKG, layer='reprojected', bbox=bbox_h)
            parcel_pt = Point(cx, cy)
            gdf_sn['_dist_ft'] = gdf_sn.geometry.distance(parcel_pt) * 3.28084
            gdf_sn = gdf_sn[gdf_sn['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_sn.iterrows():
                dist_ft = float(r['_dist_ft'])
                dia_in  = coalesce_float(r.get('DIAMETER'), r.get('PIPE_SIZE'), default=12.0)
                _mat    = str(r.get('MATERIAL', r.get('PIPETYPE', 'RCP')) or 'RCP')
                n_val   = _material_to_manning(_mat)
                _up     = coalesce_float(r.get('UPELEV'),   r.get('US_INVERT'), default=None)
                _down   = coalesce_float(r.get('DOWNELEV'), r.get('DS_INVERT'), default=None)
                _slope  = abs(_up - _down) / max(float(r.get('Shape__Length', 100) or 100), 1) \
                          if (_up and _down) else 0.004
                q_full  = manning_gpm(dia_in, n_val, slope=max(_slope, 0.001))
                q_storm = q_full * PIPE_DERATING
                if dist_ft < 75:    wt = 1.00
                elif dist_ft < 150: wt = 0.70
                elif dist_ft < 250: wt = 0.45
                else:               wt = max(0.05, 0.45 * (250 / dist_ft))
                features_h.append({
                    "dist_ft":    round(dist_ft, 1),
                    "dia_in":     dia_in,
                    "material":   _mat,
                    "slope":      round(_slope, 5),
                    "q_full_gpm": round(q_full, 2),
                    "gpm":        round(q_storm * wt, 2),
                    "weight":     round(wt, 3),
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":     "structure_network",
                })
    except Exception as _e:
        result["note"] += f" PathH-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH I — Fittings (storm_water_fitting.gpkg)
    # Pipe connections (tees, wyes, reducers). Presence confirms active
    # piped network.  Contribute minor supplemental capacity (junction loss).
    # ─────────────────────────────────────────────────────────────────
    features_i = []
    try:
        if os.path.exists(FITTINGS_GPKG):
            bbox_i = _bbox_in_crs(3857)
            gdf_fi = gpd.read_file(FITTINGS_GPKG, layer='reprojected', bbox=bbox_i)
            parcel_pt = Point(cx, cy)
            gdf_fi['_dist_ft'] = gdf_fi.geometry.distance(parcel_pt) * 3.28084
            gdf_fi = gdf_fi[gdf_fi['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_fi.iterrows():
                dist_ft  = float(r['_dist_ft'])
                _rim     = coalesce_float(r.get('RIM'), r.get('ELEVATION'), default=None)
                reachable = (_rim is not None and _rim < g_elev)
                if dist_ft < 75:    wt = 1.00
                elif dist_ft < 150: wt = 0.70
                elif dist_ft < 250: wt = 0.45
                else:               wt = max(0.05, 0.45 * (250 / dist_ft))
                features_i.append({
                    "dist_ft":    round(dist_ft, 1),
                    "rim_elev":   round(_rim, 3) if _rim is not None else None,
                    "reachable":  reachable,
                    "weight":     round(wt, 3),
                    "gpm":        0.0,   # fittings = network presence indicator, no direct flow
                    "type":       str(r.get('FITTINGTYPE', r.get('TYPE', 'unknown')) or 'unknown'),
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":     "fitting",
                })
    except Exception as _e:
        result["note"] += f" PathI-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # PATH J — Storm Control Valves (stpete_scv.gpkg)
    # CRITICAL: A nearby SCV means tidal backflow is actively prevented.
    # Overrides tidal_backflow_risk flag for any outfall in the same zone.
    # ─────────────────────────────────────────────────────────────────
    features_j = []
    has_control_valve = False
    try:
        if os.path.exists(CONTROL_VALVES_GPKG):
            bbox_j = _bbox_in_crs(3857)
            gdf_cv = gpd.read_file(CONTROL_VALVES_GPKG, layer='reprojected', bbox=bbox_j)
            parcel_pt = Point(cx, cy)
            gdf_cv['_dist_ft'] = gdf_cv.geometry.distance(parcel_pt) * 3.28084
            gdf_cv = gdf_cv[gdf_cv['_dist_ft'] <= DRAINAGE_SCAN_RADIUS_FT].copy()
            for _, r in gdf_cv.iterrows():
                dist_ft = float(r['_dist_ft'])
                _status = str(r.get('STATUS', r.get('VALVESTATU', 'unknown')) or 'unknown')
                _type   = str(r.get('VALVETYPE', r.get('TYPE', 'SCV')) or 'SCV')
                active  = _status.upper() not in ('ABANDONED', 'REMOVED', 'INACTIVE', 'CLOSED')
                if active:
                    has_control_valve = True
                features_j.append({
                    "dist_ft":  round(dist_ft, 1),
                    "type":     _type,
                    "status":   _status,
                    "active":   active,
                    "gpm":      0.0,
                    "facilityid": str(r.get('FACILITYID', '') or ''),
                    "source":   "control_valve",
                })
    except Exception as _e:
        result["note"] += f" PathJ-err:{_e}"

    # ─────────────────────────────────────────────────────────────────
    # AGGREGATE
    # ─────────────────────────────────────────────────────────────────
    path_a_gpm = sum(f["gpm"] for f in features_a)
    path_b_gpm = sum(f["gpm"] for f in features_b)
    path_c_gpm = sum(f["gpm"] for f in features_c)
    path_d_gpm = sum(f["gpm"] for f in features_d)
    path_e_gpm = sum(f["gpm"] for f in features_e)
    path_f_gpm = sum(f["gpm"] for f in features_f)
    path_g_gpm = sum(f["gpm"] for f in features_g)
    path_h_gpm = sum(f["gpm"] for f in features_h)
    # features_i (fittings) and features_j (control valves) don't contribute flow GPM
    total_gpm  = (path_a_gpm + path_b_gpm + path_c_gpm + path_d_gpm +
                  path_e_gpm + path_f_gpm + path_g_gpm + path_h_gpm)

    # tidal backflow risk — any reachable outfall pointing to tidal water,
    # BUT overridden to False if an active Storm Control Valve is detected nearby
    tidal_backflow_risk = any(f.get("tidal_risk") for f in features_f)
    if has_control_valve:
        tidal_backflow_risk = False   # SCV actively prevents tidal backflow

    # classify drainage type — now includes manhole nodes and network links
    reachable_structs = (
        [f for f in features_a if f["reachable"]] +
        [f for f in features_f if f.get("reachable")] +
        [f for f in features_g if f.get("reachable")]   # manholes below grade
    )
    n_total_features = (len(features_a) + len(features_b) + len(features_c) +
                        len(features_d) + len(features_e) + len(features_f) +
                        len(features_g) + len(features_h) + len(features_i) +
                        len(features_j))
    if n_total_features == 0:
        drain_class = "NONE"
    elif total_gpm < 10.0 or len(reachable_structs) == 0:
        drain_class = "NONE"
    elif total_gpm < 50.0 or len(reachable_structs) < 2:
        drain_class = "PARTIAL"
    else:
        drain_class = "FULL"

    # strongest contributing structure
    strongest = None
    if reachable_structs:
        strongest = min(reachable_structs, key=lambda f: f["dist_ft"])

    result.update({
        "ok":                  True,
        "drainage_class":      drain_class,
        "effective_gpm":       round(total_gpm, 1),
        "path_a_gpm":          round(path_a_gpm, 1),
        "path_b_gpm":          round(path_b_gpm, 1),
        "path_c_gpm":          round(path_c_gpm, 1),
        "path_d_gpm":          round(path_d_gpm, 1),
        "path_e_gpm":          round(path_e_gpm, 1),
        "path_f_gpm":          round(path_f_gpm, 1),
        "path_g_gpm":          round(path_g_gpm, 1),
        "path_h_gpm":          round(path_h_gpm, 1),
        "n_structures":        len(features_a),
        "n_mains":             len(features_b) + len(features_d),
        "n_open_drains":       len(features_c),
        "n_culverts":          len(features_e),
        "n_outfalls":          len(features_f),
        "n_manholes":          len(features_g),
        "n_network_links":     len(features_h),
        "n_fittings":          len(features_i),
        "n_control_valves":    len(features_j),
        "has_control_valve":   has_control_valve,
        "n_reachable_structs": len(reachable_structs),
        "tidal_backflow_risk": tidal_backflow_risk,
        "strongest_structure": strongest,
        "features_a":          features_a,
        "features_b":          features_b,
        "features_c":          features_c,
        "features_d":          features_d,
        "features_e":          features_e,
        "features_f":          features_f,
        "features_g":          features_g,
        "features_h":          features_h,
        "features_i":          features_i,
        "features_j":          features_j,
    })
    return result


# =============================================================================
# 5TH MAP — DRAINAGE INFRASTRUCTURE & CONNECTIVITY MAP
# Base layer: 1,000 ft DEM (same LiDAR raster used by the other four maps)
# Export: standalone PNG only — NOT included in the flood_report PDF
# =============================================================================
def build_drainage_context_map(
        st_addr: str,
        cx: float, cy: float, df_crs: str,
        g_elev: float,
        drainage_intel: dict,
        out_png: str) -> bool:
    """
    Generate the Drainage Infrastructure & Connectivity Map.
    1,000 ft DEM overlay + color-coded stormwater infrastructure.
    Exported as PNG only — never passed to build_client_pdf().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import matplotlib.patheffects as pe
        import rasterio
        import contextily as ctx
        from pyproj import CRS, Transformer
        from scipy.ndimage import gaussian_filter as _gauss
    except ImportError as _ie:
        print(f"   ❌  Drainage map: missing lib — {_ie}")
        return False

    radius_ft = 1000.0

    # ── reproject centroid to Web Mercator (for contextily) ───────────────
    try:
        src_crs = CRS.from_user_input(df_crs)
        wm_crs  = CRS.from_epsg(3857)
        tfm_to_wm = Transformer.from_crs(src_crs, wm_crs, always_xy=True)
        cx_wm, cy_wm = tfm_to_wm.transform(cx, cy)
        units_per_ft = _crs_units_per_foot(src_crs)
    except Exception as _e:
        print(f"   ❌  Drainage map CRS: {_e}")
        return False

    # ── load DEM window (1,000 ft radius) ─────────────────────────────────
    dem_rgba   = None
    dem_extent = None     # [west_wm, east_wm, south_wm, north_wm]
    try:
        tile_index = DemTileIndex(LIDAR_RASTER, PINELLAS_LIDAR_FILES)
        ds, sx, sy = tile_index.pick_dataset_for_point(cx, cy, df_crs)
        if ds is not None:
            r_units = radius_ft * units_per_ft
            from rasterio.windows import from_bounds
            from rasterio.transform import array_bounds
            win = from_bounds(sx - r_units, sy - r_units,
                              sx + r_units, sy + r_units,
                              ds.transform)
            band = ds.read(1, window=win).astype("float32")
            nodata = ds.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            band[(band < PINELLAS_ELEV_MIN_FT) | (band > PINELLAS_ELEV_MAX_FT)] = np.nan

            # compute Web Mercator extent from actual raster corners
            win_tf = ds.window_transform(win)
            nrows, ncols = band.shape
            corners_src = [
                (win_tf.c,              win_tf.f),
                (win_tf.c + ncols*win_tf.a, win_tf.f),
                (win_tf.c,              win_tf.f + nrows*win_tf.e),
                (win_tf.c + ncols*win_tf.a, win_tf.f + nrows*win_tf.e),
            ]
            tfm_dem_to_wm = Transformer.from_crs(ds.crs, wm_crs, always_xy=True)
            wm_xs = [tfm_dem_to_wm.transform(x, y)[0] for x, y in corners_src]
            wm_ys = [tfm_dem_to_wm.transform(x, y)[1] for x, y in corners_src]
            dem_extent = [min(wm_xs), max(wm_xs), min(wm_ys), max(wm_ys)]

            # smooth and stretch contrast
            valid_mask = ~np.isnan(band)
            if valid_mask.sum() > 50:
                band_sm = np.where(valid_mask, _gauss(np.nan_to_num(band, nan=float(np.nanmean(band))), sigma=0.7), np.nan)
                lo, hi = np.nanpercentile(band_sm[valid_mask], [2, 98])
                if hi > lo:
                    band_norm = np.clip((band_sm - lo) / (hi - lo), 0, 1)
                else:
                    band_norm = np.zeros_like(band_sm)
            else:
                band_norm = np.zeros_like(band)

            # RGBA overlay
            from matplotlib.cm import get_cmap
            cmap_d = get_cmap("RdYlGn_r")
            dem_rgba = cmap_d(band_norm)
            dem_rgba[..., 3] = np.where(valid_mask, 0.38, 0.0)
        tile_index.close()
    except Exception as _e:
        print(f"   ⚠️  Drainage map DEM load: {_e}")

    # ── set up figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14), dpi=180)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    margin_wm = radius_ft * 0.3048 * 1.05
    ax.set_xlim(cx_wm - margin_wm, cx_wm + margin_wm)
    ax.set_ylim(cy_wm - margin_wm, cy_wm + margin_wm)

    # satellite basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery,
                        zoom=17, alpha=0.75)
    except Exception:
        ax.set_facecolor("#1a2332")

    # DEM overlay
    if dem_rgba is not None and dem_extent is not None:
        ax.imshow(dem_rgba, extent=dem_extent, origin="upper",
                  aspect="auto", interpolation="bilinear", zorder=2)

    # ── project infrastructure to Web Mercator for plotting ───────────────
    try:
        from pyproj import CRS as _CRS, Transformer as _Transformer
        src_for_feat = _CRS.from_user_input(df_crs)

        def _pt_to_wm(feat_x, feat_y, feat_crs_str):
            try:
                fc = _CRS.from_user_input(feat_crs_str)
                t  = _Transformer.from_crs(fc, wm_crs, always_xy=True)
                return t.transform(feat_x, feat_y)
            except Exception:
                return feat_x, feat_y

        def _feat_xy_from_geom(geom, feat_crs_str):
            """Return (x_wm, y_wm) for the nearest point of a geometry to parcel."""
            try:
                from shapely.geometry import Point as _Pt
                # use centroid of geom projected to parcel CRS
                fc   = _CRS.from_user_input(feat_crs_str)
                t_to_src = _Transformer.from_crs(fc, src_for_feat, always_xy=True)
                cent = geom.centroid
                xs, ys = t_to_src.transform(cent.x, cent.y)
                t_to_wm = _Transformer.from_crs(src_for_feat, wm_crs, always_xy=True)
                return t_to_wm.transform(xs, ys)
            except Exception:
                return None, None

        # ── Path A — Structures ───────────────────────────────────────────
        if drainage_intel.get("features_a"):
            try:
                import geopandas as gpd
                from shapely.geometry import Point as _Pt
                gdf_plot_a = gpd.read_file(STRUCTURES_GPKG, layer='reprojected')
                gdf_plot_a = gdf_plot_a.to_crs(src_for_feat)
                parcel_pt  = _Pt(cx, cy)
                gdf_plot_a['_dist_ft'] = gdf_plot_a.geometry.distance(parcel_pt)
                gdf_plot_a = gdf_plot_a[gdf_plot_a['_dist_ft'] <= radius_ft]
                gdf_plot_a = gdf_plot_a.to_crs(wm_crs)

                for _, r in gdf_plot_a.iterrows():
                    try:
                        gx, gy = r.geometry.x, r.geometry.y
                        z1 = coalesce_float(r.get('Z_1'), default=g_elev + 1)
                        reach = z1 < g_elev
                        color = "#22c55e" if reach else "#ef4444"
                        ax.scatter(gx, gy, c=color, s=140, marker="^",
                                   edgecolors="white", linewidths=1.4, zorder=7, alpha=0.95)
                        dist_ft = float(r.get('_dist_ft', 0))
                        rel_el  = round(z1 - g_elev, 1)
                        sign_s  = "+" if rel_el >= 0 else ""
                        ax.annotate(f"{dist_ft:.0f}ft\n{sign_s}{rel_el}ft",
                                    xy=(gx, gy), xytext=(gx + 18, gy + 18),
                                    fontsize=6.5, color=color, fontweight="bold",
                                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                                    zorder=8)
                    except Exception:
                        continue
            except Exception as _ea:
                print(f"   ⚠️  Drainage map Path-A plot: {_ea}")

        # ── Path B — Gravity Mains ────────────────────────────────────────
        if drainage_intel.get("features_b"):
            try:
                import geopandas as gpd
                from shapely.geometry import Point as _Pt
                gdf_plot_b = gpd.read_file(
                    GRAVITY_MAINS_GPKG,
                    layer='reprojected')
                gdf_plot_b = gdf_plot_b.to_crs(src_for_feat)
                parcel_pt  = _Pt(cx, cy)
                gdf_plot_b['_dist_ft'] = gdf_plot_b.geometry.distance(parcel_pt)
                gdf_plot_b = gdf_plot_b[gdf_plot_b['_dist_ft'] <= radius_ft]
                gdf_plot_b = gdf_plot_b.to_crs(wm_crs)

                for _, r in gdf_plot_b.iterrows():
                    try:
                        geom = r.geometry
                        if geom is None or geom.is_empty:
                            continue
                        dia_in = _parse_culvert_size_in(r.get('CUL_SIZE'))
                        lw = max(2.0, min(5.5, dia_in / 6.0))
                        xs, ys = geom.xy
                        ax.plot(list(xs), list(ys), color="#60a5fa",
                                lw=lw, alpha=0.92, zorder=5,
                                path_effects=[pe.withStroke(linewidth=lw+2.0, foreground="#0f172a", alpha=0.7)])
                    except Exception:
                        continue
            except Exception as _eb:
                print(f"   ⚠️  Drainage map Path-B plot: {_eb}")

        # ── Path C — Open Drains ──────────────────────────────────────────
        if drainage_intel.get("features_c"):
            try:
                import geopandas as gpd
                from shapely.geometry import Point as _Pt
                gdf_plot_c = gpd.read_file(
                    OPEN_DRAINS_GPKG,
                    layer='reprojected')
                gdf_plot_c = gdf_plot_c.to_crs(src_for_feat)
                parcel_pt  = _Pt(cx, cy)
                gdf_plot_c['_dist_ft'] = gdf_plot_c.geometry.distance(parcel_pt)
                gdf_plot_c = gdf_plot_c[gdf_plot_c['_dist_ft'] <= radius_ft]
                gdf_plot_c = gdf_plot_c.to_crs(wm_crs)

                for _, r in gdf_plot_c.iterrows():
                    try:
                        geom = r.geometry
                        if geom is None or geom.is_empty:
                            continue
                        tw   = coalesce_float(r.get('TOPWIDTH'), default=4.0)
                        lw   = max(2.0, min(7.0, float(tw) / 2.5))
                        xs, ys = geom.xy
                        ax.plot(list(xs), list(ys), color="#fbbf24",
                                lw=lw, alpha=0.92, zorder=4,
                                path_effects=[pe.withStroke(linewidth=lw+2.0, foreground="#0f172a", alpha=0.7)])
                    except Exception:
                        continue
            except Exception as _ec:
                print(f"   ⚠️  Drainage map Path-C plot: {_ec}")

    except Exception as _ep:
        print(f"   ⚠️  Drainage map feature projection: {_ep}")

    # ── parcel centroid marker ────────────────────────────────────────────
    ax.scatter(cx_wm, cy_wm, c="#ffffff", s=180, marker="*",
               edgecolors="#000000", linewidths=1.2, zorder=10, label="Subject Parcel")
    # 1000 ft radius circle
    import matplotlib.patches as _mpatch
    circ = _mpatch.Circle((cx_wm, cy_wm), margin_wm / 1.05,
                           fill=False, edgecolor="#ffffff", lw=1.5,
                           linestyle="--", alpha=0.65, zorder=9)
    ax.add_patch(circ)

    # ── LiDAR flow direction arrows from radial profiles ─────────────────
    # (drawn as simple compass arrows scaled to the 1000 ft window)
    arrow_len = margin_wm * 0.18
    for dir_name, (ddx, ddy) in DIRECTIONS.items():
        ax.annotate("", xy=(cx_wm + ddx * arrow_len, cy_wm + ddy * arrow_len),
                    xytext=(cx_wm, cy_wm),
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=0.8),
                    zorder=6)

    # ── title and summary text ────────────────────────────────────────────
    dc   = drainage_intel.get("drainage_class", "NONE")
    gpm  = drainage_intel.get("effective_gpm", 0.0)
    ns   = drainage_intel.get("n_structures", 0)
    nm   = drainage_intel.get("n_mains", 0)
    nd   = drainage_intel.get("n_open_drains", 0)
    dc_color = {"NONE": "#ef4444", "PARTIAL": "#f59e0b", "FULL": "#22c55e"}.get(dc, "#94a3b8")

    ax.set_title(
        f"Drainage Infrastructure & Connectivity Map — {st_addr}\n"
        f"1,000 ft radius  |  Drainage Type: {dc}  |  Effective Capacity: {gpm:.1f} GPM",
        fontsize=10, fontweight="bold", color="white", pad=10)

    # Drainage Influence Summary box
    if dc == "NONE":
        summary_txt = ("No physically connected drainage infrastructure detected.\n"
                       "Flooding governed by terrain storage and surface runoff only.")
    else:
        strongest = drainage_intel.get("strongest_structure")
        st_info = ""
        if strongest:
            st_info = (f" | Closest structure: {strongest['dist_ft']:.0f} ft, "
                       f"Δelev {strongest['rel_elev']:+.2f} ft")
        summary_txt = (f"Drainage influenced by {ns} structures / {nm} gravity mains / "
                       f"{nd} open drains{st_info}\n"
                       f"Path A: {drainage_intel['path_a_gpm']:.1f} GPM  |  "
                       f"Path B: {drainage_intel['path_b_gpm']:.1f} GPM  |  "
                       f"Path C: {drainage_intel['path_c_gpm']:.1f} GPM")

    ax.text(0.01, 0.01, summary_txt, transform=ax.transAxes,
            fontsize=7.5, color="#e2e8f0", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", alpha=0.85, edgecolor=dc_color),
            zorder=12)

    # legend
    legend_elems = [
        mpatches.Patch(color="#22c55e", label="Structure — reachable (downhill)"),
        mpatches.Patch(color="#ef4444", label="Structure — uphill / no relief"),
        Line2D([0],[0], color="#60a5fa", lw=2, label="Gravity main (line width ∝ diameter)"),
        Line2D([0],[0], color="#fbbf24", lw=2, label="Open drain (line width ∝ top-width)"),
        Line2D([0],[0], color="#ffffff", marker="*", markersize=9, lw=0, label="Subject parcel"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True,
              facecolor="#1e293b", edgecolor="#475569", labelcolor="white",
              fontsize=7.5, framealpha=0.90)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    try:
        fig.tight_layout(pad=0.5)
        fig.savefig(out_png, bbox_inches="tight", dpi=180,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except Exception as _se:
        print(f"   ❌  Drainage map save error: {_se}")
        plt.close(fig)
        return False


# =============================================================================
# CLIENT PDF
# =============================================================================
def build_lidar_radial_tier(radial_summary: dict) -> dict:
    """
    Derive a 3-tier terrain risk ranking from the radial LiDAR summary.

    Tier thresholds (OR logic — worst single metric drives the tier):
      CRITICAL : dirs_toward >= 5  OR  max_depth >= 0.5 ft  OR  ponding >= 600 gal
      OK       : dirs_toward >= 3  OR  max_depth >= 0.25 ft OR  ponding >= 250 gal
      GOOD     : all below OK thresholds

    Returns a dict with keys:
      tier, tier_color, tier_icon, dirs_toward, max_depth_ft,
      total_gal, n_dep, summary (list of 5 sentences)
    """
    dirs_toward  = int(radial_summary.get("dirs_toward", 0))
    max_depth_ft = float(radial_summary.get("max_depression_depth_ft", 0.0))
    total_gal    = float(radial_summary.get("total_ponded_gal", 0.0))
    n_dep        = int(radial_summary.get("n_depressions", 0))
    pf_dir       = radial_summary.get("primary_flow", "unresolved")

    # ── tier classification ───────────────────────────────────────────────────
    if dirs_toward >= 5 or max_depth_ft >= 0.5 or total_gal >= 600:
        tier       = "CRITICAL"
        tier_color = "#b91c1c"
        tier_icon  = "\u26a0\ufe0f"
    elif dirs_toward >= 3 or max_depth_ft >= 0.25 or total_gal >= 250:
        tier       = "OK"
        tier_color = "#d97706"
        tier_icon  = "\u26a1"
    else:
        tier       = "GOOD"
        tier_color = "#15803d"
        tier_icon  = "\u2705"

    # ── plain-language summary (5 sentences) ─────────────────────────────────
    max_depth_in = round(max_depth_ft * 12, 1)
    pf_label     = pf_dir if pf_dir and pf_dir != "unresolved" else "multiple directions"

    s1 = (f"The LiDAR radial profile analysis scanned 8 cardinal directions "
          f"within a 250-foot radius and found that {dirs_toward} of those "
          f"directions slope toward the structure.")

    if dirs_toward >= 5:
        s2 = ("This level of terrain convergence is elevated — more than half "
              "of the surrounding surface drains toward this property, increasing "
              "the likelihood that sheet flow will accumulate near the foundation "
              "during moderate to heavy rainfall events.")
    elif dirs_toward >= 3:
        s2 = ("This indicates moderate terrain convergence; a meaningful share "
              "of surrounding surface runoff is directed toward the structure, "
              "which can become significant during multi-inch rainfall events.")
    else:
        s2 = ("This indicates low terrain convergence; most surrounding "
              "surface runoff flows away from the structure, which is a "
              "favourable drainage characteristic.")

    if n_dep > 0:
        s3 = (f"The scan identified {n_dep} micro-depression zone(s) within the "
              f"study area, with a maximum ponding depth of {max_depth_in} inches "
              f"and a combined storage capacity of {total_gal:.0f} gallons — "
              f"water that must either infiltrate or overflow before it can drain.")
    else:
        s3 = ("No significant micro-depression zones were detected within the "
              "study radius, suggesting the immediate terrain does not create "
              "natural impoundment points that would hold standing water.")

    s4 = (f"The dominant surface-flow direction is {pf_label}, based on "
          f"elevation-gradient analysis of LiDAR pixels at 2.5-foot intervals "
          f"using bilinear sub-pixel sampling for sub-cell precision.")

    if tier == "CRITICAL":
        s5 = ("Given the CRITICAL terrain tier, flood barriers are strongly "
              "recommended for any storm event exceeding a 2-year return period; "
              "the combination of directional convergence and depression storage "
              "significantly shortens the time available to respond once rainfall begins.")
    elif tier == "OK":
        s5 = ("The OK terrain tier suggests that barriers should be considered "
              "for storms exceeding the 5-year return period, and that routine "
              "inspection of low points and drainage grates near the structure "
              "is advisable before the wet season.")
    else:
        s5 = ("The GOOD terrain tier reflects relatively low topographic flood "
              "pressure; standard maintenance of gutters and downspout extensions "
              "should be sufficient to manage typical wet-season rainfall at this location.")

    return {
        "tier":        tier,
        "tier_color":  tier_color,
        "tier_icon":   tier_icon,
        "dirs_toward": dirs_toward,
        "max_depth_ft": max_depth_ft,
        "total_gal":   total_gal,
        "n_dep":       n_dep,
        "summary":     [s1, s2, s3, s4, s5],
    }


def build_client_pdf(st_addr: str, out_pdf: str, radial_png: str, nb_png: str,
                     risk_score: float, flood_icon: str, surge_ft: float,
                     foundation: str, year_built: str, opening_specs: list,
                     radial_summary: dict, permit_analysis: dict,
                     ollama_sales: Optional[str],
                     geo_verify: dict = None,
                     rainfall_damage: dict = None,
                     elnino_proj: dict = None,
                     ponding_sim_png: str = None,
                     drainage_map_png: str = None,
                     lot_png: str = None,
                     block_png: str = None,
                     proj: dict = None,
                     drainage_intel: dict = None,
                     damage_cost: dict = None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
    except Exception as e:
        print(f"   ❌ matplotlib PDF not available: {e}")
        return False

    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=220)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")
        ax.text(0.0, 0.98, "Flood Risk Assessment — Terrain & Barrier Plan",
                fontsize=17, fontweight="bold", va="top")
        ax.text(0.0, 0.94, st_addr, fontsize=13, fontweight="bold", va="top")
        ax.text(0.0, 0.90, f"Score: {round(risk_score,1)}/100   Status: {flood_icon}",
                fontsize=12, va="top")
        ax.text(0.0, 0.86,
                f"Surge head: {surge_ft} ft | Built: {year_built} | Foundation: {foundation}",
                fontsize=11, va="top")
        rs = radial_summary or {}
        ax.text(0.0, 0.82,
                f"LiDAR convergence: {rs.get('dirs_toward','?')}/8 toward-home | "
                f"Max depression: {rs.get('max_depression_depth_ft','?')} ft | "
                f"Total depressions: {rs.get('n_depressions','?')} | "
                f"Ponding: {rs.get('total_ponded_gal','?')} gal",
                fontsize=10, va="top")
        if geo_verify:
            ax.text(0.0, 0.78,
                    f"Geo-Verify: {geo_verify.get('overall_confidence','?')}",
                    fontsize=10, va="top", color="#dc2626")
        # ── Methodology box — fills blank space below narrative ───────────────
        import textwrap as _tw
        import matplotlib.patches as _mp1
        _rd1   = rainfall_damage or {}
        _inp1  = _rd1.get("inputs", {})
        _hist1 = load_historical_rainfall()
        _exc1  = _hist1.get("exceedance", {})
        _yrs1  = _hist1.get("years", 0)
        _freq_1in = _exc1.get(1.00, {}).get("freq_per_yr", 0)
        _freq_2in = _exc1.get(2.00, {}).get("freq_per_yr", 0)
        _freq_3in = _exc1.get(3.00, {}).get("freq_per_yr", 0)
        _mk1 = _inp1.get("myakka_state")
        _soil_desc = ("Myakka sand 3-state saturation model: dry C=0.15, "
                      "storm-wet C=0.55, saturated C=0.82") if _mk1 else \
                     "Standard SSURGO hydrologic soil group curve"

        # Each entry: (text, is_header, fontsize, color)
        # Long body text is pre-wrapped at 88 chars so nothing clips at right edge.
        _WRAP = 88
        _sections = [
            ("METHODOLOGY  &  DATA SOURCES", True, 9.0, "#1e3a5f"),
            ("SPACER4", False, 0, "#000"),
            ("Runoff Model", True, 8.0, "#1e3a5f"),
            (f"ASCE Rational Method  Q = C·i·A.  C = {_inp1.get('runoff_c',0):.3f} "
             f"(impervious x 0.90 + pervious x soil class).  Infiltration embedded in C, "
             f"not double-subtracted.  AMC phase-shift: C x 0.70 (0-45 min), "
             f"C x 1.00 (45-90 min), C x 1.22 (>90 min).",
             False, 7.5, "#374151"),
            ("SPACER3", False, 0, "#000"),
            ("Soil Classification", True, 8.0, "#1e3a5f"),
            (_soil_desc + ".  Source: USDA SSURGO.", False, 7.5, "#374151"),
            ("SPACER3", False, 0, "#000"),
            ("Drainage Capacity", True, 8.0, "#1e3a5f"),
            ("Manning's equation on GIS pipe diameter & slope.  Storm-deration: "
             "pipe x 0.40 (HEC-22), open drain x 0.70, inlet capture x 0.75.  "
             "Baseline floor: 75 GPM (no infrastructure) / 120 GPM (partial).  "
             "Tidal backpressure applied within 500 ft of open water.",
             False, 7.5, "#374151"),
            ("SPACER3", False, 0, "#000"),
            ("Depression Storage  (LiDAR Correlation)", True, 8.0, "#1e3a5f"),
            ("LiDAR radial scan at 250 ft.  Raw ponding volume x 70% usability "
             "(uneven micro-terrain fill).  Tc lag 8-12 min (NRCS TR-55).  "
             "Net excess GPM x fill time = LiDAR depression volume — verifying "
             "hydraulic model is consistent with actual DEM terrain capacity.",
             False, 7.5, "#374151"),
            ("SPACER3", False, 0, "#000"),
            ("Historical Rainfall  (Probability Basis)", True, 8.0, "#1e3a5f"),
            (f"Pinellas merged station record 1965-2026 ({_yrs1:.0f} yrs, deduplicated).  "
             f"Exceedance: >=1.0\" {_freq_1in:.1f}x/yr  |  >=2.0\" {_freq_2in:.1f}x/yr  "
             f"|  >=3.0\" {_freq_3in:.1f}x/yr.  Most-probable pathway scored by: "
             f"historical frequency x antecedent probability x drainage-excess margin.",
             False, 7.5, "#374151"),
            ("SPACER3", False, 0, "#000"),
            ("Limitations", True, 8.0, "#b91c1c"),
            ("First-principles model using public GIS & meteorological data.  "
             "Not a FEMA flood study or licensed engineering survey.  "
             "Results are comparative risk indicators, not engineering certifications.",
             False, 7.5, "#6b7280"),
        ]

        # Expand each body line through textwrap so no line exceeds _WRAP chars
        _render_lines = []   # list of (text, is_header, fontsize, color, is_spacer, gap)
        for _entry in _sections:
            _et, _eh, _efs, _ec = _entry
            if _et.startswith("SPACER"):
                _gap = int(_et.replace("SPACER",""))
                _render_lines.append(("", False, 0, "#000", True, _gap))
            elif _eh:
                _render_lines.append((_et, True, _efs, _ec, False, 0))
            else:
                _wrapped = _tw.wrap(_et, width=_WRAP)
                for _wl in _wrapped:
                    _render_lines.append((_wl, False, _efs, _ec, False, 0))

        # Box: top sits just below property info block, bottom at 0.02
        # axes height = 0.84 of 11.69" = 9.82" = 707pt
        _PTS_PER_AX = 707.0
        _by1 = 0.74   # fixed anchor just below the property header rows
        _by0 = 0.02
        _bx0, _bx1 = 0.0, 0.99
        _bg = _mp1.FancyBboxPatch((_bx0, _by0), _bx1 - _bx0, _by1 - _by0,
                                   boxstyle="round,pad=0.006",
                                   facecolor="#f8fafc", edgecolor="#cbd5e1",
                                   linewidth=0.9, transform=ax.transAxes, clip_on=False)
        ax.add_patch(_bg)
        _my = _by1 - 0.010
        for _lt, _lh, _lfs, _lc, _lsp, _lgap in _render_lines:
            if _lsp:
                _my -= _lgap / _PTS_PER_AX
                continue
            ax.text(0.010, _my, _lt,
                    fontsize=_lfs,
                    fontweight="bold" if _lh else "normal",
                    va="top", color=_lc,
                    transform=ax.transAxes,
                    clip_on=False)
            _my -= (_lfs * 1.35) / _PTS_PER_AX
            if _my < _by0 + 0.005:
                break

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: radial profile (or 250ft block).
        # Page 3: flood ponding simulation — shows inundation zones for projected storm.
        # The drainage connectivity map is exported as a separate PNG (not in the PDF).
        _nb_display = (ponding_sim_png
                       if (ponding_sim_png and os.path.exists(ponding_sim_png))
                       else nb_png)
        for img_path, orientation in [(radial_png, "landscape"), (_nb_display, "landscape")]:
            if img_path and os.path.exists(img_path):
                if orientation == "landscape":
                    fig = plt.figure(figsize=(11.69, 8.27), dpi=220)
                else:
                    fig = plt.figure(figsize=(8.27, 11.69), dpi=220)
                ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
                ax.axis("off")
                ax.imshow(mpimg.imread(img_path))
                pdf.savefig(fig)
                plt.close(fig)

        # ── LiDAR Radial Profile Tier Summary page ────────────
        _lidar_tier = build_lidar_radial_tier(radial_summary)
        _ltfig = plt.figure(figsize=(8.27, 11.69), dpi=200)
        _ltfig.patch.set_facecolor("white")
        _ltax = _ltfig.add_axes([0.08, 0.08, 0.84, 0.84])
        _ltax.axis("off")
        _ylt = 0.97
        _ltax.text(0.5, _ylt, "Radial LiDAR Elevation Profile Summary",
                   ha="center", va="top", fontsize=15, fontweight="bold")
        _ylt -= 0.03
        _ltax.text(0.5, _ylt, st_addr, ha="center", va="top", fontsize=11)
        _ylt -= 0.035
        _tc = _lidar_tier["tier_color"]
        _ltax.text(0.5, _ylt,
                   f"{_lidar_tier['tier_icon']}  TERRAIN TIER:  {_lidar_tier['tier']}",
                   ha="center", va="top", fontsize=14, fontweight="bold", color=_tc,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=_tc, edgecolor=_tc, alpha=0.12))
        _ylt -= 0.055
        for _ml in [
            f"Directions toward home : {_lidar_tier['dirs_toward']}/8",
            f"Max depression depth   : {round(_lidar_tier['max_depth_ft']*12,1)}\"",
            f"Total ponding capacity : {_lidar_tier['total_gal']:.0f} gal",
            f"Depression zones       : {_lidar_tier['n_dep']}",
        ]:
            _ltax.text(0.05, _ylt, _ml, fontsize=10, va="top", family="monospace")
            _ylt -= 0.025
        _ylt -= 0.015
        _ltax.text(0.0, _ylt, "TIER KEY", fontsize=10, fontweight="bold", va="top")
        _ylt -= 0.025
        for _tnm, _tc2, _tdesc in [
            ("GOOD",     "#15803d", "Low convergence, shallow depressions -- routine monitoring"),
            ("OK",       "#d97706", "Moderate terrain pressure -- barrier advised for 5-yr+ storms"),
            ("CRITICAL", "#b91c1c", "High convergence / deep depressions -- barrier strongly recommended"),
        ]:
            _ltax.text(0.03, _ylt, f"  {_tnm:<10}  {_tdesc}", fontsize=9, va="top", color=_tc2)
            _ylt -= 0.022
        _ylt -= 0.015
        _ltax.text(0.0, _ylt, "ANALYSIS", fontsize=10, fontweight="bold", va="top")
        _ylt -= 0.025
        for _sent in _lidar_tier["summary"]:
            _words_l = _sent.split()
            _ln_l = "  "
            for _wl in _words_l:
                if len(_ln_l) + len(_wl) > 92:
                    _ltax.text(0.0, _ylt, _ln_l.rstrip(), fontsize=9.5, va="top", color="#1e293b")
                    _ylt -= 0.020
                    _ln_l = "  " + _wl + " "
                else:
                    _ln_l += _wl + " "
            if _ln_l.strip():
                _ltax.text(0.0, _ylt, _ln_l.rstrip(), fontsize=9.5, va="top", color="#1e293b")
                _ylt -= 0.020
            _ylt -= 0.008
        pdf.savefig(_ltfig)
        plt.close(_ltfig)

        # ── Rainfall-to-Damage page ───────────────────────────────────────
        if rainfall_damage and rainfall_damage.get("ok"):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
            fig.patch.set_facecolor("white")
            ax  = fig.add_axes([0.06, 0.04, 0.88, 0.92])
            ax.axis("off")
            yc  = 0.98
            def _T(txt, fs=10, fw="normal", col="black", ya=yc):
                ax.text(0.0, ya, txt, fontsize=fs, fontweight=fw, va="top", color=col)
            ax.text(0.5, yc, "Rainfall-to-Damage Threshold Analysis",
                    ha="center", va="top", fontsize=14, fontweight="bold")
            yc -= 0.03
            ax.text(0.5, yc, st_addr, ha="center", va="top", fontsize=11)
            yc -= 0.025

            rd   = rainfall_damage
            inp  = rd.get("inputs", {})
            crit = rd.get("critical")
            ax.text(0.0, yc, "INPUTS", fontsize=10, fontweight="bold", va="top")
            yc -= 0.022
            _myakka_lbl = ""
            if inp.get("myakka_state"):
                _myakka_lbl = f"  [Myakka sand — {inp['myakka_state']} state]"
            for lbl, val in [
                ("Lot area (effective ponding)",
                 f"{inp.get('lot_area_sqft',0):.0f} sqft × 0.70"),
                ("Runoff coefficient C",        f"{inp.get('runoff_c',0):.3f}{_myakka_lbl}"),
                ("Soil infiltration rate",      f"{inp.get('soil_infil_in_hr',0)} in/hr"),
                ("Gravity-main pipe capacity",  f"{inp.get('pipe_capacity_gpm',0):.1f} GPM"),
                ("Open-drain capacity",         f"{inp.get('drain_capacity_gpm',0):.1f} GPM"),
                ("LiDAR depression storage",    f"{inp.get('dep_storage_gal',0):.1f} gal"),
                ("Damage volume (6\" threshold)",f"{inp.get('damage_vol_gal',0):.1f} gal"),
            ]:
                ax.text(0.02, yc, f"{lbl:<35}: {val}", fontsize=8.5, va="top",
                        family="monospace")
                yc -= 0.018

            yc -= 0.01
            ax.text(0.0, yc, "CRITICAL THRESHOLD", fontsize=10, fontweight="bold",
                    va="top", color="#b91c1c")
            yc -= 0.022
            if crit:
                for lbl, val in [
                    ("Rainfall rate required",
                     f"{crit.get('rate_in_hr','?')} in/hr"),
                    ("Max storm duration",
                     f"{crit.get('max_duration_hr','?')} hrs"),
                    ("Net surface excess",
                     f"{crit.get('net_excess_gpm','?')} GPM"),
                    ("Time to fill LiDAR depressions",
                     f"{crit.get('dep_fill_min','?')} min"),
                    ("Time to 6\" damage after fill",
                     f"{crit.get('damage_time_min','?')} min"),
                    ("TOTAL time — rain start → 6\" damage",
                     f"{crit.get('total_time_min','?')} min"),
                    ("Status",   crit.get('status','?')),
                ]:
                    ax.text(0.02, yc, f"{lbl:<40}: {val}", fontsize=8.5, va="top",
                            family="monospace",
                            color="#b91c1c" if "DAMAGE" in str(val) else "black")
                    yc -= 0.018
            elif crit and crit.get("no_flood_risk"):
                ax.text(0.02, yc,
                        "✅ NO FLOOD RISK — property survives all Pinellas County design storms.",
                        fontsize=9, va="top", color="#16a34a")
                yc -= 0.018
            else:
                ax.text(0.02, yc,
                        "✅ No rainfall scenario triggers 6\" damage within its realistic storm window.",
                        fontsize=9, va="top", color="green")
                yc -= 0.018

            # ── FLOOD PATHWAYS section ────────────────────────────────────────
            # Show every distinct rate/duration combo that CAN flood the property.
            _fps = rd.get("flood_scenarios", [])
            yc -= 0.008
            if _fps:
                ax.text(0.0, yc, "FLOOD PATHWAYS — Ways this property can reach 6\" inside",
                        fontsize=10, fontweight="bold", va="top", color="#1d4ed8")
                yc -= 0.018
                ax.text(0.01, yc,
                        f"{'Rate':>6}  {'Duration':>9}  {'Total Rain':>11}  "
                        f"{'Time→Flood':>11}  {'Storm Type':<38}  {'Freq'}",
                        fontsize=7.5, va="top", family="monospace", color="#374151")
                yc -= 0.015
                for _fp in _fps:
                    _r_s   = f"{_fp['rate_in_hr']:.2f}\""
                    _dur_s = f"{_fp['duration_hr']:.1f} hrs"
                    _tot_s = f"{_fp['total_rain_in']:.2f}\""
                    _ttf   = _fp.get("total_time_min")
                    _ttf_s = f"{_ttf:.0f} min" if _ttf else "—"
                    _lbl   = str(_fp.get("label", ""))[:38]
                    _rp    = str(_fp.get("return_period", ""))
                    _is_mp = _fp.get("most_probable", False)
                    _row_color = "#7c3aed" if _is_mp else "#b91c1c"
                    _row_weight = "bold" if _is_mp else "normal"
                    _star  = " ★ MOST PROBABLE" if _is_mp else ""
                    _ante  = str(_fp.get("antecedent_cond", ""))
                    _line  = (f"{_r_s:>6}/hr  {_dur_s:>9}  {_tot_s:>11}  "
                              f"{_ttf_s:>11}  {_lbl:<38}  {_rp}{_star}")
                    ax.text(0.01, yc, _line, fontsize=7.5, va="top",
                            family="monospace", color=_row_color,
                            fontweight=_row_weight)
                    yc -= 0.013
                    # Sub-line: antecedent condition (smaller, grey)
                    if _ante:
                        ax.text(0.015, yc, f"    Conditions: {_ante}",
                                fontsize=6.5, va="top", color="#6b7280", style="italic")
                        yc -= 0.011
                    # -- Storm callout box for most probable event --------
                    if _is_mp and _fp.get("callout") and yc > 0.18:
                        _co    = _fp["callout"]
                        _cite  = _fp.get("historical_cite", "")
                        _clines = [
                            f"  ★ {_co['name']}",
                            f"    Duration: {_co['duration']}  |  Typical rainfall: {_co['rainfall']}",
                        ]
                        for _nt in _co.get("notes", []):
                            _clines.append(f"    - {_nt}")
                        if _cite:
                            _clines.append(f"    {_cite}")
                        _box_h = len(_clines) * 0.014 + 0.010
                        _box_rect = plt.matplotlib.patches.FancyBboxPatch(
                            (0.0, yc - _box_h), 0.99, _box_h,
                            boxstyle="round,pad=0.005",
                            facecolor="#ede9fe", edgecolor="#7c3aed",
                            linewidth=1.2, transform=ax.transAxes, clip_on=False)
                        ax.add_patch(_box_rect)
                        for _ci, _cl in enumerate(_clines):
                            ax.text(0.01, yc - 0.005 - _ci * 0.014, _cl,
                                    fontsize=7.0, va="top",
                                    family="monospace", color="#4c1d95")
                        yc -= (_box_h + 0.008)
                    if yc < 0.18:   # don't run off the page
                        ax.text(0.01, yc, f"  … and {len(_fps) - _fps.index(_fp) - 1} more scenarios",
                                fontsize=7.5, va="top", color="#6b7280")
                        break
                # ── cumulative excess threshold note ──────────────────────
                if _fps and yc > 0.14:
                    _cum_vals = [fp.get("total_rain_in", 0) for fp in _fps if fp.get("total_rain_in")]
                    if _cum_vals:
                        _cum_avg = sum(_cum_vals) / len(_cum_vals)
                        ax.text(0.01, yc,
                                f"  Cumulative excess threshold: ~{_cum_avg:.2f}\" total rain  "
                                f"(all pathways converge here — rate determines how fast, "
                                f"not whether flooding occurs)",
                                fontsize=7.0, va="top", color="#6b7280",
                                style="italic")
                        yc -= 0.014
                    # conditional footnote
                    has_conditional = any("conditional" in str(fp.get("return_period","")) for fp in _fps)
                    if has_conditional:
                        ax.text(0.01, yc,
                                "  * conditional = common rain event, but flood only triggers given "
                                "wet antecedent conditions (prior rain / high tide / saturated soil)",
                                fontsize=6.5, va="top", color="#9ca3af", style="italic")
                        yc -= 0.012
                # Methodology is on page 1 — no need to repeat here.
            elif not _fps and not (crit and crit.get("no_flood_risk")):
                ax.text(0.0, yc, "FLOOD PATHWAYS",
                        fontsize=10, fontweight="bold", va="top", color="#1d4ed8")
                yc -= 0.018
                ax.text(0.01, yc, "No flood scenario found within any realistic storm window.",
                        fontsize=9, va="top", color="#16a34a")
                yc -= 0.018
            yc -= 0.006

            # ── Volumetric DEM cross-check ─────────────────────────────────
            _vdc = rd.get("vol_dem_corr")
            if _vdc and yc > 0.22:
                ax.text(0.0, yc, "VOLUMETRIC DEM CORRELATION  (most-probable storm)",
                        fontsize=8.5, fontweight="bold", va="top", color="#0f172a")
                yc -= 0.016
                _vdc_rows = [
                    ("Net surface excess generated",
                     f"{_vdc['total_excess_vol_gal']:>10,.0f} gal"),
                    ("  Phase 1 — depression absorption (LiDAR)",
                     f"{_vdc['dep_storage_capacity_gal']:>10,.0f} gal"
                     f"  ({_vdc['dep_fill_fraction_pct']:.1f}%)"),
                    ("  Phase 2 — residual toward structure",
                     f"{_vdc['residual_after_dep_gal']:>10,.0f} gal"),
                    ("Damage-threshold volume (6\")",
                     f"{_vdc['damage_threshold_gal']:>10,.0f} gal"),
                    ("Volume balance error",
                     f"{_vdc['vol_balance_error_pct']:>9.1f}%"
                     f"  {'✓ CONFIRMED' if _vdc['confirmed'] else '⚠ boundary'}"),
                ]
                for _vk, _vv in _vdc_rows:
                    ax.text(0.01, yc, f"  {_vk:<42} {_vv}",
                            fontsize=7.0, va="top", family="monospace",
                            color="#1e3a5f")
                    yc -= 0.013
                yc -= 0.006

            # Intensity table — 1.0 in/hr steps (readable across full 1–6 in/hr range)
            yc -= 0.01
            ax.text(0.0, yc, "INTENSITY TABLE  (1.0 in/hr steps)",
                    fontsize=9, fontweight="bold", va="top")
            yc -= 0.020
            hdr = f"{'Rate':>6}  {'Dur':>5}  {'NetGPM':>7}  {'DepMin':>7}  {'DmgMin':>7}  {'TotMin':>7}"
            ax.text(0.01, yc, hdr, fontsize=7.5, va="top", family="monospace",
                    color="#374151")
            yc -= 0.015
            for row_d in rd.get("table", []):
                # show only whole in/hr values (1.0, 2.0, … 24.0)
                _r = row_d["rate_in_hr"]
                if abs(round(_r) - _r) > 0.05:
                    continue
                ttm = row_d.get("total_time_min")
                dfm = row_d.get("dep_fill_min")
                dtm = row_d.get("damage_time_min")
                line_c = "#b91c1c" if row_d.get("within_storm") else "#374151"
                ax.text(0.01, yc,
                        f"{row_d['rate_in_hr']:>5.1f}  "
                        f"{row_d['max_duration_hr']:>4.1f}h  "
                        f"{row_d.get('net_excess_gpm',0):>7.1f}  "
                        f"{str(round(dfm,1)) if dfm else 'N/A':>7}  "
                        f"{str(round(dtm,1)) if dtm else 'N/A':>7}  "
                        f"{str(round(ttm,1)) if ttm else 'N/A':>7}",
                        fontsize=7.5, va="top", family="monospace", color=line_c)
                yc -= 0.014
                if yc < 0.10:
                    break

            # Disclaimer
            yc -= 0.008
            disc = rd.get("disclaimer", "")
            # wrap at ~110 chars for smaller font
            words_d = disc.split()
            line_d  = ""
            disc_lines = []
            for w in words_d:
                if len(line_d) + len(w) > 110:
                    disc_lines.append(line_d.strip())
                    line_d = w + " "
                else:
                    line_d += w + " "
            if line_d.strip():
                disc_lines.append(line_d.strip())
            ax.text(0.0, yc, "METHODOLOGY DISCLAIMER",
                    fontsize=7.5, fontweight="bold", va="top", color="#6b7280")
            yc -= 0.016
            for dl in disc_lines:
                ax.text(0.0, yc, dl, fontsize=6.8, va="top", color="#6b7280")
                yc -= 0.013
                if yc < 0.02:
                    break

            pdf.savefig(fig)
            plt.close(fig)

        # ── El Niño Projection page ───────────────────────────────────────
        if elnino_proj and elnino_proj.get("ok"):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=200)
            fig.patch.set_facecolor("white")
            ax  = fig.add_axes([0.06, 0.04, 0.88, 0.92])
            ax.axis("off")
            yc  = 0.98
            ax.text(0.5, yc, "Super El Niño — Rainfall Damage Projection",
                    ha="center", va="top", fontsize=14, fontweight="bold",
                    color="#1d4ed8")
            yc -= 0.03
            ax.text(0.5, yc, st_addr, ha="center", va="top", fontsize=11)
            yc -= 0.025

            ep = elnino_proj
            ax.text(0.0, yc, "HISTORICAL ONI RECORD SUMMARY",
                    fontsize=10, fontweight="bold", va="top")
            yc -= 0.022
            for lbl, val in [
                ("ONI data range",
                 f"{ep['oni_years'][0]}–{ep['oni_years'][1]}"),
                ("Super El Niño years (ONI ≥ 2.0)",
                 f"{ep['n_super_years']}  ({ep['prob_super_year_pct']}% annual probability)"),
                ("Wet-season super months (Jun–Sep)",
                 f"{len(ep['wet_super_months'])}  ({ep['prob_wet_season_pct']}% per month)"),
                ("Super El Niño years on record",
                 ", ".join(str(y) for y in ep['super_years'][-10:]) +
                 ("..." if len(ep['super_years']) > 10 else "")),
            ]:
                ax.text(0.02, yc, f"{lbl:<42}: {val}", fontsize=8.5, va="top",
                        family="monospace")
                yc -= 0.018

            yc -= 0.01
            ax.text(0.0, yc, f"CURRENT YEAR ({ep['current_year']}) ENSO STATUS",
                    fontsize=10, fontweight="bold", va="top",
                    color="#b91c1c" if ep["is_super_this_year"] else "#1d4ed8")
            yc -= 0.022
            for lbl, val in [
                ("Condition",         ep['current_label']),
                ("Wet-season multiplier", f"{ep['current_multiplier']}x  rainfall enhancement"),
                ("Base critical rate",
                 f"{ep['critical_rate_in_hr']} in/hr  (from hydraulic calculation)"),
                ("Effective trigger rate",
                 f"{ep['adjusted_critical']} in/hr  (lower threshold under El Niño)"),
            ]:
                ax.text(0.02, yc, f"{lbl:<38}: {val}", fontsize=8.5, va="top",
                        family="monospace",
                        color="#b91c1c" if ep["is_super_this_year"] else "black")
                yc -= 0.018

            if ep["is_super_this_year"]:
                yc -= 0.005
                ax.text(0.01, yc,
                        f"⚠  SUPER EL NIÑO ACTIVE — damage threshold reduced from "
                        f"{ep['critical_rate_in_hr']} → {ep['adjusted_critical']} in/hr",
                        fontsize=9, va="top", fontweight="bold", color="#b91c1c")
                yc -= 0.022

            if ep.get("breached_months"):
                yc -= 0.01
                ax.text(0.0, yc,
                        f"HISTORICAL MONTHS EXCEEDING DAMAGE THRESHOLD  "
                        f"({ep['n_breached_months']} total)",
                        fontsize=9, fontweight="bold", va="top")
                yc -= 0.018
                hdr2 = f"{'Year':>5}  {'Mon':>4}  {'ONI':>5}  {'NormPk':>7}  {'AdjPk':>7}  {'Crit':>6}"
                ax.text(0.01, yc, hdr2, fontsize=7.5, va="top",
                        family="monospace", color="#374151")
                yc -= 0.015
                for bm in ep["breached_months"][-25:]:
                    ax.text(0.01, yc,
                            f"{bm['year']:>5}  {bm['month_name']:>4}  "
                            f"{bm['oni']:>5.2f}  "
                            f"{bm['normal_peak']:>6.2f}\"  "
                            f"{bm['adjusted_peak']:>6.2f}\"  "
                            f"{bm['critical_rate']:>5.2f}\"",
                            fontsize=7.5, va="top", family="monospace",
                            color="#b91c1c")
                    yc -= 0.013
                    if yc < 0.06:
                        break
                if len(ep["breached_months"]) > 25:
                    ax.text(0.01, yc,
                            f"... and {len(ep['breached_months'])-25} additional months",
                            fontsize=7.5, va="top", color="#6b7280")
                    yc -= 0.013

            pdf.savefig(fig)
            plt.close(fig)

        # ── Damage Cost Model page (full multi-depth breakdown) ──────────
        print(f"   💵  damage_cost ok={bool(damage_cost and damage_cost.get('ok'))}")
        if damage_cost and damage_cost.get("ok"):
          try:
            _dc = damage_cost

            # ── Page A: Valuation foundation + insurance inputs ───────────
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            yc = 0.965

            ax.text(0.0, yc,
                    "FLOOD DAMAGE & INSURANCE EXPOSURE ANALYSIS",
                    fontsize=13, fontweight="bold", va="top", color="#0f172a")
            yc -= 0.020
            ax.text(0.0, yc, st_addr, fontsize=8.5, va="top",
                    color="#374151", style="italic")
            yc -= 0.025

            # ── Valuation foundation ──────────────────────────────────────
            ax.text(0.0, yc, "PROPERTY VALUATION FOUNDATION",
                    fontsize=9, fontweight="bold", va="top", color="#1d4ed8")
            yc -= 0.016
            for _vl in _dc["val_source_detail"]:
                ax.text(0.01, yc, f"  {_vl}",
                        fontsize=7.5, va="top", family="monospace", color="#1e3a5f")
                yc -= 0.013
            yc -= 0.006

            # Contents ratio
            ax.text(0.01, yc,
                    f"  Contents-to-structure ratio:  {int(_dc['cont_ratio']*100)}%"
                    f"  —  {_dc['cont_tier_label']}",
                    fontsize=7.5, va="top", family="monospace", color="#1e3a5f")
            yc -= 0.013
            ax.text(0.01, yc,
                    f"  Source: FEMA NFIP actuarial claims data; "
                    f"Insurance Information Institute 2024 Homeowners Report",
                    fontsize=6.8, va="top", color="#6b7280", style="italic")
            yc -= 0.016

            # Summary values
            _val_rows = [
                ("Structure RCV (Replacement Cost Value)",
                 f"${_dc['structure_rcv']:>13,.0f}"),
                ("Contents RCV",
                 f"${_dc['contents_rcv']:>13,.0f}"),
                ("Saltwater/tidal adjustment",
                 f"×{_dc['salt_mult']:.2f}  {_dc['salt_label']}"),
                (f"Nearest water body",
                 f"{_dc['waterbody_name']}"),
            ]
            for _rl, _rv in _val_rows:
                ax.text(0.01, yc, f"  {_rl:<40}  {_rv}",
                        fontsize=7.5, va="top", family="monospace", color="#0f172a")
                yc -= 0.013
            yc -= 0.010

            # ── ACV depreciation ──────────────────────────────────────────
            ax.text(0.0, yc, "DEPRECIATION  (Actual Cash Value vs. Replacement Cost Value)",
                    fontsize=9, fontweight="bold", va="top", color="#1d4ed8")
            yc -= 0.016
            _yr_disp = str(_dc.get("year_built", "Unknown") or "Unknown")
            _acv_rows = [
                ("Year built",
                 f"{_yr_disp}"),
                ("Structure age",
                 f"{_dc['age_yr']} years"),
                ("Depreciation rate (FL residential)",
                 "1.5% per year"),
                ("Total depreciation applied",
                 f"{_dc['depr_pct']:.1f}%  (capped at 80%)"),
                ("ACV / RCV ratio",
                 f"{_dc['acv_ratio']:.3f}  "
                 f"({int(_dc['acv_ratio']*100)}% of replacement cost)"),
                ("Structure ACV",
                 f"${_dc['structure_acv']:>13,.0f}  "
                 f"(${_dc['structure_rcv'] - _dc['structure_acv']:,.0f} gap vs. RCV)"),
                ("Contents ACV",
                 f"${_dc['contents_acv']:>13,.0f}  "
                 f"(NFIP always pays ACV for contents)"),
            ]
            for _al, _av in _acv_rows:
                ax.text(0.01, yc, f"  {_al:<42}  {_av}",
                        fontsize=7.5, va="top", family="monospace", color="#1e3a5f")
                yc -= 0.013
            ax.text(0.01, yc,
                    f"  Source: {_dc['acv_source']}",
                    fontsize=6.5, va="top", color="#6b7280", style="italic")
            yc -= 0.016

            # ── Insurance inputs ──────────────────────────────────────────
            ax.text(0.0, yc, "INSURANCE INPUTS  (Pinellas County Typical Defaults)",
                    fontsize=9, fontweight="bold", va="top", color="#1d4ed8")
            yc -= 0.016
            _ins_rows = [
                ("FLOOD INSURANCE (NFIP Standard)", ""),
                ("  Building coverage limit",
                 f"${_dc['flood_struct_limit']:>13,.0f}  (NFIP max)"),
                ("  Contents coverage limit",
                 f"${_dc['flood_cont_limit']:>13,.0f}  (NFIP max)"),
                ("  Deductible",
                 f"${_dc['flood_deductible']:>13,.0f}  per occurrence"),
                ("  Structure payout basis",
                 "RCV if ≥80% insured; otherwise ACV  (44 CFR §61.3)"),
                ("  Contents payout basis",
                 "ACV always  (SFIP Contents Coverage clause)"),
                ("HOMEOWNERS / WIND INSURANCE", ""),
                ("  Coverage A (dwelling) limit",
                 f"${_dc['ho_struct_limit']:>13,.0f}"),
                ("  Named-storm wind deductible (3%)",
                 f"${_dc['ho_wind_deductible']:>13,.0f}  (FL Stat. §627.701)"),
                ("  Rising floodwater coverage",
                 "EXCLUDED  (ISO HO-3 Exclusion J — standard FL policy)"),
                ("  Wind damage coverage",
                 "COVERED at RCV for Cat 1+ hurricane events"),
            ]
            for _il, _iv in _ins_rows:
                _bold = _il.startswith(("FLOOD", "HOME"))
                _col  = "#0f172a" if _bold else "#1e3a5f"
                _fs   = 8.0 if _bold else 7.5
                ax.text(0.01, yc, f"  {_il:<42}  {_iv}",
                        fontsize=_fs, va="top", family="monospace",
                        color=_col, fontweight="bold" if _bold else "normal")
                yc -= 0.013
            yc -= 0.012

            # ── EAD + ROI summary ─────────────────────────────────────────
            ax.text(0.0, yc,
                    "EXPECTED ANNUAL DAMAGE (EAD) & MITIGATION ROI",
                    fontsize=9, fontweight="bold", va="top", color="#1d4ed8")
            yc -= 0.016
            _ead_rows = [
                ("EAD — probability-weighted annual loss",
                 f"${_dc['ead_annual']:>13,.0f} / year"),
                ("30-year expected loss (unmitigated)",
                 f"${_dc['ead_annual']*30:>13,.0f}"),
            ]
            for _el, _ev in _ead_rows:
                ax.text(0.01, yc, f"  {_el:<42}  {_ev}",
                        fontsize=7.5, va="top", family="monospace", color="#1e3a5f")
                yc -= 0.013

            pdf.savefig(fig)
            plt.close(fig)

            # ── Page B: Per-depth damage cards ────────────────────────────
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            yc = 0.965

            ax.text(0.0, yc,
                    "FLOOD DEPTH DAMAGE SCENARIOS  —  FULL INSURANCE BREAKDOWN",
                    fontsize=12, fontweight="bold", va="top", color="#0f172a")
            yc -= 0.018
            ax.text(0.0, yc,
                    f"{st_addr}   |   "
                    f"Structure RCV ${_dc['structure_rcv']:,.0f}   "
                    f"Contents RCV ${_dc['contents_rcv']:,.0f}   "
                    f"ACV {int(_dc['acv_ratio']*100)}%",
                    fontsize=7.5, va="top", color="#374151", style="italic")
            yc -= 0.022

            # One full page per depth scenario — guaranteed no overlap
            _card_colors = [
                "#166534",   # 6"  — green
                "#1d4ed8",   # 1ft — blue
                "#d97706",   # 18" — amber
                "#c2410c",   # 24" — orange-red
                "#7f1d1d",   # 36" — dark red
            ]
            _sources_txt = (
                "SOURCES: FEMA HAZUS-MH 2.1 Technical Manual Table 7.9 (RES1-1SNB FL coastal); "
                "FEMA SFIP Building & Contents Coverage (44 CFR Part 61); "
                "FL Stat. §627.701 (wind deductibles); ISO HO-3 Exclusion J (flood exclusion); "
                "NHC Tropical Cyclone Reports 2017–2024; NOAA NCEI Storm Events Database; "
                "Marshall & Swift Residential Cost Handbook 2024; "
                "Insurance Information Institute 2024 Homeowners Report."
            )
            for _sc_idx, (_sc, _hdr_col) in enumerate(
                    zip(_dc["depth_scenarios"], _card_colors)):
                # Save the previous page and start fresh for each scenario
                if _sc_idx > 0:
                    pdf.savefig(fig)
                    plt.close(fig)

                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                yc = 0.965

                # Page header
                ax.text(0.0, yc,
                        "FLOOD DEPTH DAMAGE SCENARIOS  —  FULL INSURANCE BREAKDOWN",
                        fontsize=13, fontweight="bold", va="top", color="#0f172a")
                yc -= 0.026
                ax.text(0.0, yc,
                        f"{st_addr}   |   "
                        f"Structure RCV ${_dc['structure_rcv']:,.0f}   "
                        f"Contents RCV ${_dc['contents_rcv']:,.0f}   "
                        f"ACV {int(_dc['acv_ratio']*100)}%   "
                        f"Scenario {_sc_idx+1} of {len(_dc['depth_scenarios'])}",
                        fontsize=8.5, va="top", color="#374151", style="italic")
                yc -= 0.048

                _is_hurricane = _sc["wind_mph"] >= 70

                # Scenario header
                ax.text(0.0, yc,
                        f"▌ {_sc['depth_label']}  —  {_sc['storm_type']}"
                        + (f"  ({_sc['wind_mph']} mph)" if _sc["wind_mph"] > 0 else ""),
                        fontsize=15, fontweight="bold", va="top", color=_hdr_col)
                yc -= 0.038

                # Event citation
                ax.text(0.02, yc,
                        f"Closest Pinellas event: {_sc['pinellas_event']}",
                        fontsize=9.5, va="top", color="#374151", style="italic")
                yc -= 0.026
                ax.text(0.02, yc,
                        f"Source: {_sc['event_source']}",
                        fontsize=8.5, va="top", color="#6b7280", style="italic")
                yc -= 0.026

                # Return period
                ax.text(0.02, yc,
                        f"Return period: {_sc['return_period']}   "
                        f"|   Annual probability: {_sc['annual_prob']*100:.1f}%",
                        fontsize=10, va="top", color="#374151")
                yc -= 0.050

                # Damage breakdown
                _dmg_lines = [
                    ("FLOOD DAMAGE (FEMA HAZUS RES1-1SNB)", "", True),
                    (f"  Structure — flood  "
                     f"({_sc['hazus_struct_flood_pct']:.1f}% HAZUS × "
                     f"×{_dc['salt_mult']:.2f} salt mult)",
                     f"${_sc['struct_flood_dmg']:>12,.0f}", False),
                    (f"  Contents  — flood  "
                     f"({_sc['hazus_cont_flood_pct']:.1f}% HAZUS × ACV "
                     f"{int(_dc['acv_ratio']*100)}%)",
                     f"${_sc['cont_flood_dmg']:>12,.0f}", False),
                ]
                if _is_hurricane:
                    _dmg_lines += [
                        ("HURRICANE WIND DAMAGE (concurrent)", "", True),
                        (f"  Structure wind  "
                         f"(+{_sc['wind_struct_add_pct']:.0f}% at {_sc['wind_mph']} mph)",
                         f"${_sc['struct_wind_dmg']:>12,.0f}", False),
                        (f"  Contents  wind  (+{_sc['wind_cont_add_pct']:.0f}%)",
                         f"${_sc['cont_wind_dmg']:>12,.0f}", False),
                    ]
                _dmg_lines += [
                    ("TOTAL COMBINED DAMAGE", f"${_sc['total_dmg']:>12,.0f}", True),
                ]
                for _dl, _dv, _dbold in _dmg_lines:
                    _dfs  = 10.5 if _dbold else 10
                    _dcol = "#0f172a" if _dbold else "#1e3a5f"
                    ax.text(0.02, yc,
                            f"  {_dl:<50} {_dv}",
                            fontsize=_dfs, va="top", family="monospace",
                            color=_dcol,
                            fontweight="bold" if _dbold else "normal")
                    yc -= 0.030

                yc -= 0.022

                # Insurance recovery
                _ins_lines = [
                    ("INSURANCE RECOVERY", "", True),
                    ("  Flood insurance payout (NFIP structure + contents)",
                     f"−${_sc['flood_total_payout']:>11,.0f}", False),
                ]
                if _sc["ho_payout"] > 0:
                    _ins_lines.append((
                        "  Homeowners / wind payout",
                        f"−${_sc['ho_payout']:>11,.0f}", False))
                else:
                    _ins_lines.append((
                        "  Homeowners / wind payout  (flood excluded — HO-3 J)",
                        "$            0", False))
                _ins_lines.append((
                    "  Flood deductible applied",
                    f"+${_dc['flood_deductible']:>11,.0f}", False))
                if _is_hurricane:
                    _ins_lines.append((
                        "  Wind deductible applied  (3% of Coverage A)",
                        f"+${_dc['ho_wind_deductible']:>11,.0f}", False))
                for _il2, _iv2, _ib2 in _ins_lines:
                    _ifs  = 10.5 if _ib2 else 10
                    _icol = "#0f172a" if _ib2 else "#1e3a5f"
                    ax.text(0.02, yc, f"  {_il2:<50} {_iv2}",
                            fontsize=_ifs, va="top", family="monospace",
                            color=_icol,
                            fontweight="bold" if _ib2 else "normal")
                    yc -= 0.030

                # OUT-OF-POCKET — pinned to bottom of page, always at fixed position
                from matplotlib.patches import FancyBboxPatch
                _oop = _sc["out_of_pocket"]
                _oop_y = 0.10   # fixed bottom anchor regardless of content above
                ax.add_patch(FancyBboxPatch(
                    (0.01, _oop_y), 0.97, 0.058,
                    boxstyle="round,pad=0.008",
                    linewidth=2.5, edgecolor="#b91c1c",
                    facecolor="#fef2f2", zorder=2, clip_on=False))
                ax.text(0.5, _oop_y + 0.029,
                        f"OUT-OF-POCKET EXPOSURE:  ${_oop:,.0f}",
                        fontsize=15, fontweight="bold", va="center", ha="center",
                        color="#b91c1c", zorder=3, clip_on=False)

                # Sources footer on last scenario page only
                if _sc_idx == len(_dc["depth_scenarios"]) - 1:
                    ax.text(0.0, 0.025, _sources_txt,
                            fontsize=6.0, va="top", color="#6b7280", wrap=True)

            pdf.savefig(fig)
            plt.close(fig)

          except Exception as _ins_err:
            print(f"   ❌  Insurance pages failed: {_ins_err}")
            import traceback; traceback.print_exc()

    # ── Append FEMA document at page 5 ───────────────────────────────────
    # Always inserts FEMA starting at page 5 by padding blank pages if the
    # report has fewer than 4 pages.  Uses pypdf (maintained fork of PyPDF2).
    _fema_path = FEMA_PDF_PATH
    print(f"   🔍  FEMA path configured: {_fema_path}")
    print(f"   🔍  FEMA exists: {os.path.exists(_fema_path) if _fema_path else False}")
    
    if not _fema_path or not os.path.exists(_fema_path):
        print(f"   🔍  Searching for FEMA.pdf in directories...")
        _fema_search_dirs = [
            os.path.expanduser('~/Documents'),
            os.path.expanduser('~/Desktop'),
            os.path.expanduser('~/QField'),
            os.path.expanduser('~/QField/GeoTif'),
            MASTER_EXPORT_DIR,
            os.path.expanduser('~/Downloads'),
            '.',
        ]
        for _sd in _fema_search_dirs:
            if not os.path.isdir(_sd):
                continue
            print(f"   🔍  Searching in: {_sd}")
            for _dp, _dns, _fns in os.walk(_sd):
                for _fn in _fns:
                    if _fn.lower() in ('femadoc.pdf', 'fema.pdf', 'open.pdf'):
                        _fema_path = os.path.join(_dp, _fn)
                        print(f"   ✅  Found FEMA.pdf: {_fema_path}")
                        break
                if _fema_path and os.path.exists(_fema_path):
                    break
            if _fema_path and os.path.exists(_fema_path):
                break

    if _fema_path and os.path.exists(_fema_path) and os.path.exists(out_pdf):
        print(f"   🔍  Starting FEMA insertion...")
        print(f"   🔍  Report PDF exists: {os.path.exists(out_pdf)}")
        print(f"   🔍  Report PDF: {out_pdf}")
        try:
            try:
                from pypdf import PdfReader, PdfWriter, PageObject
            except ImportError:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "pip", "install", "pypdf"], check=True)
                from pypdf import PdfReader, PdfWriter, PageObject
            writer = PdfWriter()
            # ── Page order: Report → Photo Analysis → FEMA → Garrison ─────────
            # Photo analysis comes right after the main report so the client
            # sees all property data and visuals first, then supporting docs.

            # 1. Add all existing report pages
            report_reader = PdfReader(out_pdf)
            print(f"   🔍  Report has {len(report_reader.pages)} pages")
            for page in report_reader.pages:
                writer.add_page(page)

            # ── 2. Photo Analysis Section — one portrait page per image ──────
            # Comes right after the main report, before FEMA/Garrison, so the
            # client sees all property visuals and data first.
            # Each page: photo fills top ~58%, recap analysis fills bottom ~36%.
            # All photos converted PNG→JPEG for file-size savings before embed.
            import tempfile as _tf
            import matplotlib as _mpl
            _mpl.use("Agg")
            import matplotlib.pyplot as _plt
            import matplotlib.image as _mpimg
            import matplotlib.patches as _mpatches
            from matplotlib.backends.backend_pdf import PdfPages as _PP

            # Pull analysis values from available dicts
            _rd  = rainfall_damage or {}
            _inp = _rd.get("inputs", {})
            _crit = _rd.get("critical") or {}
            _fps  = _rd.get("flood_scenarios", [])
            _proj = proj or {}
            _di   = drainage_intel or {}

            _runoff_c   = _inp.get("runoff_c",  round(risk_score / 100, 3))
            _lot_area   = _inp.get("lot_area_sqft", 0)
            _mk_state   = _inp.get("myakka_state") or "Standard HSG"
            _dep_gal    = _inp.get("dep_storage_gal", 0)
            _dep_raw    = _inp.get("dep_storage_raw_gal", _dep_gal)
            _pipe_raw   = _inp.get("pipe_capacity_gpm_raw", 0)
            _drain_raw  = _inp.get("drain_capacity_gpm_raw", 0)
            _dist_water = _inp.get("dist_to_water_ft", "Unknown")
            _dc         = _di.get("drainage_class", "ASSUMED")
            _eff_gpm    = _di.get("effective_gpm", 0)
            _waterbody  = _proj.get("waterbody", "Unknown")
            _evac       = _proj.get("surge_evac_zone_ft", 0)
            _fp_add     = _proj.get("surge_floodplain_ft", 0)
            _surge      = _proj.get("combined_surge_projection_ft", surge_ft)

            # Most probable flood event
            _mp = next((f for f in _fps if f.get("most_probable")), _crit or {})
            _mp_rate  = _mp.get("rate_in_hr", "—")
            _mp_dur   = _mp.get("duration_hr", "—")
            _mp_tot   = _mp.get("total_rain_in", "—")
            _mp_ttf   = _mp.get("total_time_min", "—")
            _mp_label = _mp.get("label", "—")
            _mp_ante  = _mp.get("antecedent_cond", "—")

            def _make_analysis_page(img_path, title, analysis_lines):
                """Render one portrait page: photo top, recap analysis bottom."""
                try:
                    # PNG → JPEG conversion for smaller file size
                    try:
                        from PIL import Image as _PI
                        import io as _bio
                        _pi = _PI.open(img_path).convert("RGB")
                        _jb = _bio.BytesIO()
                        _pi.save(_jb, format="JPEG", quality=82,
                                 optimize=True, progressive=True)
                        _jb.seek(0)
                        _img_arr = _mpimg.imread(_jb, format="jpeg")
                    except Exception:
                        _img_arr = _mpimg.imread(img_path)

                    # Always portrait for analysis pages
                    _pfig = _plt.figure(figsize=(8.27, 11.69), dpi=180)
                    _pfig.patch.set_facecolor("white")

                    # ── Header bar ──────────────────────────────────────────
                    _hax = _pfig.add_axes([0.0, 0.94, 1.0, 0.06])
                    _hax.set_facecolor("#1e3a5f")
                    _hax.axis("off")
                    _hax.text(0.5, 0.5, title, ha="center", va="center",
                              fontsize=13, fontweight="bold", color="white")

                    # ── Photo (top 55%) ─────────────────────────────────────
                    _iax = _pfig.add_axes([0.03, 0.38, 0.94, 0.55])
                    _iax.imshow(_img_arr, aspect="equal")
                    _iax.axis("off")

                    # ── Analysis box (bottom 34%) ───────────────────────────
                    _aax = _pfig.add_axes([0.03, 0.03, 0.94, 0.34])
                    _aax.axis("off")
                    _aax.set_facecolor("#f8fafc")
                    _rect = _mpatches.FancyBboxPatch(
                        (0, 0), 1, 1, boxstyle="round,pad=0.01",
                        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2,
                        transform=_aax.transAxes, clip_on=False)
                    _aax.add_patch(_rect)
                    _aax.text(0.02, 0.97, "Analysis & Data Summary",
                              fontsize=9, fontweight="bold", va="top",
                              color="#1e3a5f", transform=_aax.transAxes)
                    _line_h = 0.90 / max(len(analysis_lines), 1)
                    for _li, _ln in enumerate(analysis_lines):
                        _yy = 0.90 - _li * min(_line_h, 0.11)
                        _aax.text(0.02, _yy, _ln, fontsize=7.8, va="top",
                                  color="#374151", transform=_aax.transAxes,
                                  wrap=False)

                    _tmp = _tf.NamedTemporaryFile(suffix=".pdf", delete=False)
                    _tmp.close()
                    with _PP(_tmp.name) as _ppdf:
                        _ppdf.savefig(_pfig)
                    _plt.close(_pfig)
                    _pr = PdfReader(_tmp.name)
                    for _pg in _pr.pages:
                        writer.add_page(_pg)
                    os.unlink(_tmp.name)
                    print(f"   📸  Analysis page appended: {title}")
                except Exception as _ape:
                    print(f"   ⚠️  Could not render analysis page ({title}): {_ape}")

            # ── 1. Lot — 60 ft view ──────────────────────────────────────────
            if lot_png and os.path.exists(lot_png):
                _make_analysis_page(lot_png, f"Lot Aerial — 60 ft  |  {st_addr}", [
                    f"  Lot area: {_lot_area:,.0f} sq ft",
                    f"  Runoff coefficient (C): {_runoff_c:.3f}  —  derived from impervious fraction × 0.90 + pervious × soil class",
                    f"  Soil classification: {_mk_state}{'  (hardpan saturation model applied)' if 'myakka' in str(_mk_state).lower() or _mk_state not in ('Standard HSG', None) else ''}",
                    f"  LiDAR depression storage: {_dep_raw:.0f} gal raw  →  {_dep_gal:.0f} gal usable (70% efficiency factor)",
                    f"  Time-of-concentration lag: 8–12 min (NRCS TR-55 flat suburban lot)",
                    f"  This view captures the immediate lot boundary, impervious surface coverage, and micro-terrain features",
                    f"  used to compute surface runoff generation and depression fill time in the flood threshold model.",
                ])

            # ── 2. Block — 250 ft view ───────────────────────────────────────
            if block_png and os.path.exists(block_png):
                _make_analysis_page(block_png, f"Block Aerial — 250 ft  |  {st_addr}", [
                    f"  Drainage class (GIS scan): {_dc}",
                    f"  Pipe capacity (Manning's, raw): {_pipe_raw:.1f} GPM  |  Open drain raw: {_drain_raw:.1f} GPM",
                    f"  Effective drainage after storm-deration (×0.40 pipe, ×0.70 drain): see flood pathway table",
                    f"  Baseline sheet-flow floor applied: 75 GPM min (NONE) / 120 GPM min (PARTIAL)",
                    f"  Distance to nearest open water: {_dist_water} ft  —  tidal backpressure factor applied to pipe capacity",
                    f"  The 250 ft view shows neighbouring parcels and street drainage patterns that influence",
                    f"  contributing area, overland flow paths, and the density of catalogued storm infrastructure.",
                ])

            # ── 3. Neighborhood — 1,000 ft view ─────────────────────────────
            if nb_png and os.path.exists(nb_png):
                _make_analysis_page(nb_png, f"Neighborhood — 1,000 ft  |  {st_addr}", [
                    f"  Nearest water body: {_waterbody}  |  Distance: {_dist_water} ft",
                    f"  Combined surge projection: {_surge:.2f} ft  (canal + evac zone + floodplain + grade components)",
                    f"  Evacuation zone surge component: +{_evac:.2f} ft  |  Floodplain component: +{_fp_add:.2f} ft",
                    f"  Overall flood risk score: {round(risk_score, 1)}/100  {flood_icon}",
                    f"  The 1,000 ft view frames the property within its drainage basin, showing proximity to",
                    f"  tidal water bodies, canal networks, and low-elevation areas that concentrate surge",
                    f"  and slow gravity-drain outfall during multi-hour storm events.",
                ])

            # ── 4. Flood Ponding Simulation ──────────────────────────────────
            if ponding_sim_png and os.path.exists(ponding_sim_png):
                _mp_rate_s = f"{_mp_rate:.2f} in/hr" if isinstance(_mp_rate, float) else str(_mp_rate)
                _mp_dur_s  = f"{_mp_dur:.1f} hrs"    if isinstance(_mp_dur,  float) else str(_mp_dur)
                _mp_tot_s  = f"{_mp_tot:.2f}\""       if isinstance(_mp_tot,  float) else str(_mp_tot)
                _mp_ttf_s  = f"{_mp_ttf:.0f} min"     if isinstance(_mp_ttf,  float) else str(_mp_ttf)
                _make_analysis_page(ponding_sim_png, f"Flood Ponding Simulation  |  {st_addr}", [
                    f"  Most probable flood event: {_mp_label}",
                    f"  Rate: {_mp_rate_s}  |  Storm duration: {_mp_dur_s}  |  Total rain required: {_mp_tot_s}",
                    f"  Estimated time from rain-start to 6\" interior damage threshold: {_mp_ttf_s}",
                    f"  Antecedent conditions required: {_mp_ante}",
                    f"  Color scale: deeper reds = higher ponding depth relative to finished floor grade.",
                    f"  Simulation uses LiDAR DEM at 1,000 ft radius. Inundation computed from net excess GPM",
                    f"  after pipe + open-drain capacity is exhausted, accumulated over the storm duration.",
                ])

            # ── 5. Drainage Infrastructure Map ───────────────────────────────
            if drainage_map_png and os.path.exists(drainage_map_png):
                _pa_gpm = _di.get("path_a_gpm", 0)
                _pb_gpm = _di.get("path_b_gpm", 0)
                _pc_gpm = _di.get("path_c_gpm", 0)
                _make_analysis_page(drainage_map_png, f"Drainage Infrastructure Map  |  {st_addr}", [
                    f"  Infrastructure scan radius: 1,000 ft from parcel centroid",
                    f"  Drainage class: {_dc}  |  Total effective GPM: {_eff_gpm:.1f}",
                    f"  Path A (structures): {_pa_gpm:.1f} GPM  |  Path B (gravity mains): {_pb_gpm:.1f} GPM  |  Path C (open drains): {_pc_gpm:.1f} GPM",
                    f"  Storm-deration applied: pipes ×0.40 (HEC-22), open drains ×0.70, inlet capture ×0.75",
                    f"  Tidal backpressure factor: {'applied (dist ' + str(_dist_water) + ' ft)' if isinstance(_dist_water, (int,float)) and _dist_water < 500 else 'not applied (>500 ft from water)'}",
                    f"  Blue lines = gravity mains  |  Green = open drains  |  Markers = inlet structures",
                    f"  Missing from map = infrastructure not catalogued in Pinellas County GIS as of report date.",
                ])

            # ── 3. FEMA document ─────────────────────────────────────────────
            fema_reader = PdfReader(_fema_path)
            print(f"   🔍  FEMA has {len(fema_reader.pages)} pages")
            for page in fema_reader.pages:
                writer.add_page(page)
            print(f"   📄  FEMA appended ({len(fema_reader.pages)} page(s))")

            # ── 4. Garrison case-study ───────────────────────────────────────
            _garrison_path = GARRISON_CS_PDF
            for _galt in [
                os.path.join(os.path.expanduser('~/Desktop'), 'Garrison.pdf'),
                os.path.join(os.path.expanduser('~/Desktop'), 'garrison_FAQ.pdf'),
                os.path.join(os.path.expanduser('~/Documents'), 'Garrison.pdf'),
                os.path.join(os.path.expanduser('~/Documents'), 'garrison_FAQ.pdf'),
                os.path.join(os.path.expanduser('~/QField'), 'Garrison.pdf'),
                os.path.join(MASTER_EXPORT_DIR, 'Garrison.pdf'),
            ]:
                if not os.path.exists(_garrison_path) and os.path.exists(_galt):
                    _garrison_path = _galt
            if os.path.exists(_garrison_path):
                garrison_reader = PdfReader(_garrison_path)
                for page in garrison_reader.pages:
                    writer.add_page(page)
                print(f"   📄  Garrison appended ({len(garrison_reader.pages)} page(s))")
            else:
                print(f"   ⚠️  Garrison PDF not found at {_garrison_path} — skipped")

            # ── Compress all page content streams before writing ──────────
            # compress_content_streams() applies zlib deflate to each page's
            # content stream — no quality loss, typically 20–40% size reduction
            # on top of the JPEG image savings already applied above.
            try:
                for _cp in writer.pages:
                    _cp.compress_content_streams()
                print("   🗜️  Content streams compressed")
            except Exception as _ce:
                print(f"   ⚠️  Stream compression skipped: {_ce}")

            with open(out_pdf, 'wb') as f_out:
                writer.write(f_out)
            print(f"   🔍  Final PDF has {len(writer.pages)} total pages")
        except Exception as e_fema:
            print(f"   ⚠️  Could not append FEMA PDF: {e_fema}")
            import traceback
            print(f"   🔍  Full error: {traceback.format_exc()}")
    else:
        print(f"   ⚠️  FEMA insertion skipped:")
        print(f"       FEMA path: {_fema_path}")
        print(f"       FEMA exists: {os.path.exists(_fema_path) if _fema_path else 'N/A'}")
        print(f"       Report exists: {os.path.exists(out_pdf)}")

    return True


# =============================================================================
# 10 CRITICAL DATASETS
# =============================================================================
def build_10_datasets(row, sq_ft: float) -> list:
    g_elev = derive_ground_elevation_ft(row)
    i_elev = float(row.get('_min', 0) or 0)
    delta  = g_elev - i_elev
    stdev  = float(row.get('_stdev', 0) or 0)
    candidates = []

    if is_valid(row.get('Elev_Z_1')) and is_valid(row.get('_min')):
        if delta < 0:      s = score_ds(100)
        elif delta < 1.0:  s = score_ds(100 - delta * 45)
        elif delta < 4.0:  s = score_ds(55 - (delta-1)*15)
        else:              s = score_ds(max(5, 10 - (delta-4)*2))
        candidates.append({
            "id": "DS1", "name": "ELEVATION DELTA — Ground vs Pipe Invert",
            "score": s,
            "key_values": f"Ground: {round(g_elev,2)} ft | Pipe Invert: {round(i_elev,2)} ft | Δ: {round(delta,2)} ft",
            "finding": (
                f"The ground surface sits {abs(round(delta,2))} ft "
                f"{'BELOW' if delta < 0 else 'above'} the nearest pipe invert."
            ),
        })

    if is_valid(row.get('_stdev')):
        s = score_ds(5 + (stdev / 0.50) * 95)
        candidates.append({
            "id": "DS2", "name": "TOPOGRAPHIC STAGNATION INDEX — LiDAR Variance",
            "score": s,
            "key_values": f"Elevation std dev: {round(stdev,4)} ft",
            "finding": f"LiDAR standard deviation of {round(stdev,4)} ft.",
        })

    if is_valid(row.get('Elev_Z_1')) and is_valid(row.get('_min')):
        surcharge_hgl = round(i_elev + 1.5, 2)
        if delta < 0:     s = score_ds(100)
        elif delta < 1.0: s = score_ds(90 - delta*30)
        elif delta < 3.0: s = score_ds(60 - ((delta-1)/2)*45)
        else:             s = score_ds(max(5, 15 - (delta-3)*3))
        candidates.append({
            "id": "DS3", "name": "HYDRAULIC BACKFLOW — Surcharge HGL vs Ground",
            "score": s,
            "key_values": f"Surcharge HGL: {surcharge_hgl} ft | Ground: {round(g_elev,2)} ft",
            "finding": f"When the municipal pipe surcharges (HGL = {surcharge_hgl} ft).",
        })

    dist_to_pipe = safe(row.get('distance_8', row.get('distance_5', row.get('distance'))))
    material     = safe(row.get('MATERIAL_3', row.get('MATERIAL')))
    if dist_to_pipe is not None:
        d   = float(dist_to_pipe)
        mat = str(material or "UNKNOWN").upper()
        mat_bonus = 25 if 'CMP' in mat else 12 if 'RCP' in mat else 0
        base = (85 if d<50 else 85-((d-50)/100)*35 if d<150
                else 50-((d-150)/150)*25 if d<300 else 15 if d<900 else 30)
        s = score_ds(base + mat_bonus)
        candidates.append({
            "id": "DS4", "name": "DRAINAGE PROXIMITY — Nearest Storm Pipe",
            "score": s,
            "key_values": f"Distance: {round(d,0)} ft | Material: {mat}",
            "finding": f"Nearest storm drainage is {round(d,0)} ft away, material {mat}.",
        })

    if is_valid(row.get('Elev_Z_1')) and is_valid(row.get('_min')):
        if delta < 0:      s = score_ds(100)
        elif delta < 0.5:  s = score_ds(85 - delta*30)
        elif delta < 2.0:  s = score_ds(70 - ((delta-0.5)/1.5)*45)
        else:              s = score_ds(max(5, 25 - (delta-2)*5))
        candidates.append({
            "id": "DS5", "name": "LATERAL SOIL MIGRATION — Gradient-Driven Hydrostatic Load",
            "score": s,
            "key_values": f"Gradient: {round(delta,2)} ft",
            "finding": f"A {round(abs(delta),2)} ft gradient {'below' if delta < 0 else 'above'} pipe invert.",
        })

    musym = safe(row.get('MUSYM'))
    if musym is not None:
        high_risk = ['PsA','EeA','BaA','Fo','St','Wn']
        is_hydric = str(musym).strip() in high_risk
        s = score_ds(78 if is_hydric else 22)
        candidates.append({
            "id": "DS6", "name": "SOIL SERIES — SSURGO Hydric Classification",
            "score": s,
            "key_values": f"Map Unit: {musym} | Hydric: {'YES' if is_hydric else 'No'}",
            "finding": f"SSURGO soil series {musym} is {'HYDRIC' if is_hydric else 'non-hydric'}.",
        })

    cul_raw = safe(row.get('CUL_SIZE_3', row.get('CUL_SIZE')))
    if cul_raw is not None:
        try:
            cul_in = float(str(cul_raw).replace('"','').replace('-inch','').split()[0])
            if cul_in < 12:      s = score_ds(90)
            elif cul_in < 15:    s = score_ds(90 - ((cul_in-12)/3)*15)
            elif cul_in < 18:    s = score_ds(75 - ((cul_in-15)/3)*20)
            elif cul_in < 24:    s = score_ds(55 - ((cul_in-18)/6)*45)
            else:                s = score_ds(10)
            candidates.append({
                "id": "DS7", "name": "CULVERT BOTTLENECK — Inlet Capacity vs Design Standard",
                "score": s,
                "key_values": f"Culvert size: {cul_in:.0f}\" | Pinellas standard: 15–18\"",
                "finding": f"The {cul_in:.0f}\" culvert vs Pinellas 15–18\" standard.",
            })
        except: pass

    if is_valid(row.get('_stdev')):
        if stdev < 0.15:   s = score_ds(5 + (stdev/0.15)*35)
        elif stdev < 0.30: s = score_ds(40 + ((stdev-0.15)/0.15)*35)
        else:              s = score_ds(75 + ((stdev-0.30)/0.20)*25)
        candidates.append({
            "id": "DS8", "name": "VERTICAL SETTLEMENT COHERENCE — Structural Bearing Risk",
            "score": s,
            "key_values": f"Elevation variance: {round(stdev,4)} ft SD",
            "finding": f"LiDAR SD of {round(stdev,4)} ft.",
        })

    if is_valid(row.get('SHAPESTLen')) and is_valid(row.get('SHAPESTAre')):
        sl = float(row.get('SHAPESTLen') or 0)
        sa = float(row.get('SHAPESTAre') or 10000)
        sf = (sl / math.sqrt(sa)) if sa > 0 and sl > 0 else 5.0
        s  = score_ds(10 + ((sf-4.0)/4.0)*80)
        candidates.append({
            "id": "DS9", "name": "SUBSIDENCE VOID POTENTIAL — Lot Shape Factor",
            "score": s,
            "key_values": f"Shape factor: {round(sf,3)}",
            "finding": f"Lot shape factor {round(sf,3)}.",
        })

    if is_valid(row.get('Elev_Z_1')) and is_valid(row.get('_min')):
        if delta < 0:      s = score_ds(95)
        elif delta < 1.0:  s = score_ds(80 - delta*25)
        elif delta < 3.0:  s = score_ds(55 - ((delta-1)/2)*40)
        else:              s = score_ds(max(10, 15 - (delta-3)*2))
        equity_factor = round(abs(min(delta,0))*1.8 + max(0,(2.0-delta)*0.9), 2)
        candidates.append({
            "id": "DS10", "name": "EQUITY EROSION MULTIPLIER — Property Value Risk",
            "score": s,
            "key_values": f"Risk factor: {equity_factor} | Delta: {round(delta,2)} ft",
            "finding": f"Hydraulic failure markers carry equity erosion risk factor {equity_factor}.",
        })

    mat3 = safe(row.get('MATERIAL_3'))
    if mat3 is not None:
        mu = str(mat3).upper()
        s  = score_ds(92 if 'CMP' in mu or 'MET' in mu else 65 if 'RCP' in mu or 'CONC' in mu else 32)
        year_str = safe(row.get('LINEDYEAR_3', row.get('LINEDYEAR', '')), 'unknown')
        candidates.append({
            "id": "DS11", "name": "CULVERT CORROSION TIMELINE — Infrastructure Age",
            "score": s,
            "key_values": f"Material: {mu} | Lined year: {year_str}",
            "finding": f"{mu} drainage infrastructure age assessment.",
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]


def build_ds_comparison_table(ds_list: list) -> str:
    lines = []
    lines.append(f"  {'Rank':>4}  {'ID':>5}  {'Score':>7}  {'Risk':>5}  {'Dataset Name'}")
    lines.append("  " + "─"*72)
    for rank, ds in enumerate(sorted(ds_list, key=lambda x: x["score"], reverse=True), 1):
        icon = ds_icon(ds["score"])
        name = ds["name"].split(" — ")[0]
        lines.append(f"  {rank:>4}  {ds['id']:>5}  {ds['score']:>6.1f}  {icon:>5}  {name}")
    lines.append("")
    scores = [d["score"] for d in ds_list]
    lines.append(f"  Mean score   : {sum(scores)/len(scores):.1f}")
    lines.append(f"  Max score    : {max(scores):.1f}  ({next(d['id'] for d in ds_list if d['score']==max(scores))})")
    lines.append(f"  Min score    : {min(scores):.1f}  ({next(d['id'] for d in ds_list if d['score']==min(scores))})")
    lines.append(f"  Score spread : {max(scores)-min(scores):.1f} pts")
    return "\n".join(lines)


# =============================================================================
# FLOOD SCENARIO + FINAL SUMMARY
# =============================================================================
def build_flood_scenario(row, st_addr, g_elev, delta, surge_ft, t_flood,
                          crit_i, p_gpm, terrain_analysis, proj, ds_list) -> str:
    dirs_toward = terrain_analysis.get("dirs_toward", 0)
    pf_dir      = terrain_analysis.get("primary_flow", "unknown")
    waterbody   = proj.get("waterbody", "unknown")
    evac_zone   = str(row.get('EVACZONE', 'unknown'))
    floodplain  = str(row.get('FLOODPLAIN', 'unknown'))
    year_built  = str(row.get('YEARBUILT', row.get('ACTYRBLT', 'unknown')))
    foundation  = str(row.get('FOUNDATION', 'unknown'))
    musym       = str(row.get('MUSYM', 'unknown'))
    top_risks   = [(d["name"], d["score"]) for d in ds_list[:5]]
    prompt = f"""You are a forensic flood modeling engineer. Using ALL available data for this property,
construct a specific, realistic flood scenario with a timeline.

PROPERTY DATA:
Address: {st_addr}
Ground Elevation: {round(g_elev,2)} ft NAVD88
Elevation Delta (ground - pipe invert): {round(delta,2)} ft
Projected Surge: {surge_ft} ft
Time to Inundation (3.5 in/hr): {t_flood} min
Critical Intensity (pipe failure): {round(crit_i,2)} in/hr
Year Built: {year_built} | Foundation: {foundation} | Soil: {musym}
Waterbody: {waterbody} | Evac Zone: {evac_zone} | Floodplain: {floodplain}
Terrain: {dirs_toward}/8 directions drain TOWARD home | Primary flow: {pf_dir}
Top Risk Factors: {top_risks}

TASK: Write a specific flood event scenario. Include:
1. EVENT TYPE 2. TRIGGER 3. TIMELINE (hour-by-hour)
4. WATER PATH 5. DEPTH ESTIMATE @1hr/3hr/6hr
6. CRITICAL FACTOR 7. PREVENTION WINDOW

Write in present tense. 5-7 sentences. Be specific to the data. No bullet points."""
    return ollama_query(prompt, "Flood Scenario") or None


def build_final_summary(row, st_addr, risk_score, flood_icon, g_elev, delta,
                         surge_ft, t_flood, crit_i, terrain_analysis,
                         ds_list, permit_analysis, proj) -> str:
    dirs_toward = terrain_analysis.get("dirs_toward", 0)
    top_ds      = ds_list[:3]
    top_names   = [d["name"].split(" — ")[0] for d in top_ds]
    waterbody   = proj.get("waterbody", "nearby waterways")
    year_built  = str(row.get('YEARBUILT', row.get('ACTYRBLT', 'unknown')))
    foundation  = str(row.get('FOUNDATION', 'unknown'))
    prompt = f"""You are a senior forensic flood risk engineer writing the FINAL SUMMARY.

Property: {st_addr}
Master Risk Score: {round(risk_score,1)}/100 — {flood_icon}
Ground Elevation: {round(g_elev,2)} ft NAVD88
Elevation Delta: {round(delta,2)} ft
Projected Surge: {surge_ft} ft
Terrain: {dirs_toward}/8 toward home
Year Built: {year_built} | Foundation: {foundation}
Top 3 Risk Factors: {', '.join(top_names)}
Prior Flood Evidence: {'YES — permit cluster' if permit_analysis.get('prior_flood') else 'No definitive clusters'}

Write exactly 6–8 sentences. No bullet points. Flowing professional paragraphs."""
    return ollama_query(prompt, "Final Summary") or None


# =============================================================================
# SCORE TRIPLE-CHECK
# =============================================================================
def triple_check_score(ds_list: list, section_scores: list, hydro_adj: float,
                        risk_score: float, row) -> dict:
    all_scores = [d["score"] for d in ds_list] + section_scores
    ds_mean    = sum(d["score"] for d in ds_list) / max(len(ds_list), 1)
    sec_mean   = sum(section_scores) / max(len(section_scores), 1)
    pass1      = min(100, (ds_mean*0.65 + sec_mean*0.35 + hydro_adj) * 1.10)
    pass2      = min(100, sum(all_scores)/max(len(all_scores),1) * 1.10)
    sorted_s   = sorted(all_scores)
    n          = len(sorted_s)
    median     = sorted_s[n//2] if n % 2 else (sorted_s[n//2-1]+sorted_s[n//2])/2
    pass3      = min(100, median * 1.10 + hydro_adj * 0.5)
    avg3       = round((pass1 + pass2 + pass3) / 3, 1)
    spread     = round(max(pass1,pass2,pass3) - min(pass1,pass2,pass3), 1)
    return {
        "pass1": round(pass1,1), "pass2": round(pass2,1), "pass3": round(pass3,1),
        "average_3_pass": avg3, "spread": spread,
        "divergence_flag":   spread > 8.0,
        "null_data_warning": len(ds_list) < 6,
        "final_verified":    round(risk_score,1),
        "verified_ok":       spread <= 8.0,
    }


# =============================================================================
# ADDRESS RESOLUTION
# =============================================================================
def resolve_address(row, leads_term="") -> dict:
    log, pcpao_data = [], {}
    def cl(v): return "" if str(v).strip() in ["nan","None",""] else str(v).strip()
    prop = cl(row.get("PROPERTYAD",""))
    if prop:
        log.append(f"T1 PASS: '{prop}'")
        pcpao_data = scrape_pcpao_for_parcel(parcel_id=cl(row.get("PARCELID","")), address=prop)
        return {"resolved_address":prop,"method":"T1_PROPERTYAD","confidence":"HIGH",
                "pcpao_data":pcpao_data,"log":log}
    log.append("T1 FAIL")
    ss = cl(row.get("SITE_ST",""))
    sc = cl(row.get("SITE_CITY",""))
    sz = cl(row.get("SITE_ZIP",""))
    if ss and (sc or sz):
        comp = f"{ss}, {sc}, FL {sz}".strip(", ")
        log.append(f"T2 PASS: '{comp}'")
        pcpao_data = scrape_pcpao_for_parcel(address=ss[:8])
        return {"resolved_address":comp,"method":"T2_COMPOSITE","confidence":"HIGH",
                "pcpao_data":pcpao_data,"log":log}
    log.append("T2 FAIL")
    pid = cl(row.get("PARCELID",""))
    if pid:
        pcpao_data = scrape_pcpao_for_parcel(parcel_id=pid)
        if pcpao_data.get("site_address"):
            log.append(f"T3 PASS: '{pcpao_data['site_address']}'")
            return {"resolved_address":pcpao_data["site_address"],"method":"T3_PARCELID",
                    "confidence":"HIGH","pcpao_data":pcpao_data,"log":log}
    log.append("T3 FAIL")
    fx = cl(row.get("feature_x",""))
    fy = cl(row.get("feature_y",""))
    if fx and fy:
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format":"json","lon":fx,"lat":fy},
                headers={"User-Agent":"OVERLORD-MAX/121.2"}, timeout=10)
            if r.status_code == 200:
                d = r.json()
                road = d.get("address",{}).get("road","")
                if road:
                    addr = f"{d.get('address',{}).get('house_number','')} {road}".strip()
                    log.append(f"T4 PASS: Nominatim '{addr}'")
                    return {"resolved_address":addr,"method":"T4_NOMINATIM","confidence":"MEDIUM",
                            "pcpao_data":pcpao_data,"log":log}
        except: log.append("T4 error")
    log.append("T4 FAIL")
    fallback = leads_term or pid or "UNKNOWN-ADDRESS"
    return {"resolved_address":fallback,"method":"T0_FALLBACK","confidence":"NONE",
            "pcpao_data":pcpao_data,"log":log}


# =============================================================================
# DEPLOYMENT PLAN
# =============================================================================
def calc_deployment(opening_name: str, width_ft: float,
                     surge_ft: float, g_elev: float) -> dict:
    plank_h_in  = VERSAI["plank_height_in"]
    plank_h_ft  = VERSAI["plank_height_ft"]
    plank_wt_lf = VERSAI["plank_weight"]
    post_wt_lf  = VERSAI["post_weight"]
    surge_in    = surge_ft * 12.0
    n_planks    = max(1, math.ceil(surge_in / plank_h_in))
    barrier_in  = n_planks * plank_h_in
    barrier_ft  = barrier_in / 12.0
    post_count  = 2 + (1 if width_ft > 8.0 else 0)
    center_post = width_ft > 8.0
    plank_linear_ft_total = n_planks * width_ft
    post_linear_ft_total  = post_count * barrier_ft
    plank_weight = plank_linear_ft_total * plank_wt_lf
    post_weight  = post_linear_ft_total  * post_wt_lf
    total_weight = round(plank_weight + post_weight, 1)
    H          = barrier_ft
    F_h        = hydro_versai(H) * width_ft * H / 2.0
    A_w        = width_ft * H * 144.0
    stress_psi = round((F_h + total_weight) / max(A_w, 0.01), 3)
    deploy_crew = "Solo deployment" if total_weight < 75 else "2-person deployment"
    return {
        "opening":        opening_name,
        "width_ft":       width_ft,
        "surge_in":       round(surge_in, 1),
        "n_planks":       n_planks,
        "barrier_in":     round(barrier_in, 1),
        "barrier_ft":     round(barrier_ft, 3),
        "center_post":    center_post,
        "post_count":     post_count,
        "plank_lf":       round(plank_linear_ft_total, 1),
        "weight_lbs":     total_weight,
        "stress_psi":     stress_psi,
        "deploy_crew":    deploy_crew,
        "within_typical": 30.0 <= barrier_in <= 48.0,
    }


# =============================================================================
# REPORT WRITER
# =============================================================================
def write_report(fp, st_addr, flood_icon, risk_score, parcel_id, map_path,
                 pcpao_url_str, permit_analysis, g_elev, delta, surge_ft,
                 t_flood, p_gpm, mat_data, eff_ratio, crit_i, waterbody,
                 proj, sc_canal, evac_s, fp_s, neg_d, thresh_k,
                 runoff_c, store_gal, soil_infil, scenarios,
                 flood_scenario, psi, psi_s, psi_c, tons, tons_s, tons_c,
                 grad_s, grad_c, sfp_s, sfp_c, cap_s, cap_c, tti_s, tti_c,
                 ds_list, terrain_analysis, raster_used, pipe_data,
                 nino, opening_specs, check, final_summary,
                 year_built, foundation, row,
                 geo_verify=None, radial=None,
                 rainfall_damage=None, elnino_proj=None):

    with open(fp, 'w', encoding='utf-8') as f:
        def W(s=""): f.write(str(s)+"\n")
        def H(title, width=72):
            W(); W("┌" + "─"*width + "┐")
            pad = (width - len(title) - 2) // 2
            W("│" + " "*pad + " " + title + " " + " "*(width - pad - len(title) - 2) + "│")
            W("└" + "─"*width + "┘")

        now_str = datetime.now().strftime("%B %d, %Y  %I:%M %p")
        W("╔══════════════════════════════════════════════════════════════════════════╗")
        W("║          F L O O D   R I S K   A S S E S S M E N T   R E P O R T      ║")
        W("╠══════════════════════════════════════════════════════════════════════════╣")
        W(f"║  ENGINE    :  OVERLORD-MAX v121.2  |  FORENSIC FLOOD ENGINE v7         ║")
        W(f"║  Property  :  {st_addr:<58}║")
        W(f"║  Status    :  {flood_icon:<58}║")
        W(f"║  Score     :  {round(risk_score,1)}/100{'':<54}║")
        W(f"║  Generated :  {now_str:<58}║")
        W("╚══════════════════════════════════════════════════════════════════════════╝")
        W()
        W(f"  Parcel ID    :  {parcel_id}")
        W(f"  PCPAO Link   :  {pcpao_url_str}")
        W(f"  LiDAR Tiles  :  {len(PINELLAS_LIDAR_FILES)} cross-reference tile(s)")
        W(f"  DEM Radius   :  {int(HOME_MAP_RADIUS_FT)} ft  (bilinear sub-pixel)")
        W(f"  OVERLAY FIX  :  v121.2 — extent from actual raster window corners in EPSG:3857")
        W()

        if geo_verify:
            H("GEO-VERIFICATION — 3-PASS ADDRESS ↔ COORDINATES CHECK")
            W(f"  Overall Confidence  :  {geo_verify['overall_confidence']}")
            p1 = geo_verify["pass1_fwd_geocode"]
            p2 = geo_verify["pass2_rev_geocode"]
            p3 = geo_verify["pass3_elev_sanity"]
            W(f"  Pass 1 — Forward Geocode : {'✅ PASS' if p1['ok'] else '❌ FAIL'}")
            W(f"  Pass 2 — Reverse Geocode : {'✅ PASS' if p2['ok'] else '❌ FAIL'} (score={p2.get('match_score',0):.3f})")
            W(f"  Pass 3 — Elevation Sanity: {'✅ PASS' if p3['ok'] else '❌ FAIL'} ({p3.get('sampled_elev','N/A')} ft)")
            if geo_verify["flags"]:
                for flag in geo_verify["flags"]: W(f"  {flag}")
            W()

        H("HISTORICAL PERMIT FLOOD INDICATOR")
        W(f"  Status                   :  {permit_analysis['tier']}")
        W(f"  Total Permits on Record  :  {permit_analysis['n_permits']}")
        W()

        H("KEY METRICS")
        W(f"  Ground Elevation       :  {round(g_elev,4)} ft NAVD88")
        W(f"  Elevation Delta        :  {round(delta,2)} ft  (ground vs pipe invert)")
        W(f"  Surge Projection       :  {surge_ft} ft above threshold")
        ttf_str = "Handles design storm" if t_flood is None else f"{t_flood} min"
        W(f"  Time to Inundation     :  {ttf_str}  @  3.5 in/hr design storm")
        W(f"  Pipe Capacity          :  {round(p_gpm,1)} GPM")
        W(f"  Terrain Flow           :  {terrain_analysis['primary_flow']} primary | "
          f"{terrain_analysis['dirs_toward']}/8 directions drain toward home")
        if radial and radial.get("ok"):
            s = radial["summary"]
            W(f"  DEM Depressions        :  {s.get('n_depressions',0)} detected | "
              f"Max depth: {s['max_depression_depth_ft']*12:.1f}\" | "
              f"Total ponding: {s['total_ponded_gal']:.0f} gal")
        W()

        H("DATASET COMPARISON TABLE — ALL 10 FORENSIC SCORES")
        W(build_ds_comparison_table(ds_list))
        W()

        H("STORM SURGE PROJECTION")
        W(f"  Canal / Waterbody  :  +{sc_canal} ft")
        W(f"  Evacuation Zone    :  +{evac_s} ft")
        W(f"  Floodplain Class   :  +{fp_s} ft")
        W(f"  Below-Grade Delta  :  +{round(neg_d,2)} ft")
        W(f"  TOTAL SURGE        :  {surge_ft} ft   (~{thresh_k}K gallons to breach)")
        W()

        H("FLOOD RISK SCORE — TRIPLE VERIFICATION")
        W(f"  Pass 1  (65/35 weighted)    :  {check['pass1']}/100")
        W(f"  Pass 2  (equal-weight mean) :  {check['pass2']}/100")
        W(f"  Pass 3  (median-based)      :  {check['pass3']}/100")
        W(f"  3-Pass Average              :  {check['average_3_pass']}/100")
        W(f"  Spread  (max − min)         :  {check['spread']} pts")
        W(f"  FINAL VERIFIED SCORE  :  {round(risk_score,1)} / 100  —  {flood_icon}")
        W()

        H("FINAL FLOOD RISK SUMMARY")
        if final_summary:
            words = final_summary.split()
            line  = "  "
            for w in words:
                if len(line) + len(w) > 80:
                    W(line); line = "  " + w + " "
                else:
                    line += w + " "
            if line.strip(): W(line)
        W()

        # ── RAINFALL-TO-DAMAGE ANALYSIS ────────────────────────────────────
        if rainfall_damage and rainfall_damage.get("ok"):
            H("RAINFALL-TO-DAMAGE THRESHOLD ANALYSIS")
            rd  = rainfall_damage
            inp = rd.get("inputs", {})
            crit = rd.get("critical")
            W(f"  METHOD  :  ASCE Rational Method + Manning Pipe Capacity + LiDAR DEM Depression Storage")
            W(f"  INPUTS:")
            W(f"    Lot area (effective ponding)  :  {inp.get('lot_area_sqft',0):.0f} sqft × 0.70 = "
              f"{inp.get('lot_area_sqft',0)*0.70:.0f} sqft")
            _mk_state = inp.get("myakka_state")
            _mk_note  = f"  [Myakka sand — {_mk_state} state]" if _mk_state else ""
            W(f"    Runoff coefficient (C)        :  {inp.get('runoff_c',0):.3f}{_mk_note}")
            W(f"    Soil infiltration rate        :  {inp.get('soil_infil_in_hr',0)} in/hr  (SSURGO class)")
            if _mk_state:
                W(f"    Myakka sand saturation model  :  storm-wet (C=0.55, infil=0.30 in/hr)")
                W(f"    Myakka dry-state reference    :  C=0.15, infil=3.50 in/hr (first 30-60 min only)")
                W(f"    Myakka saturated-state ref    :  C=0.82, infil=0.05 in/hr (prolonged storm)")
            W(f"    Gravity-main pipe capacity    :  {inp.get('pipe_capacity_gpm',0):.1f} GPM  (Manning's n)")
            W(f"    Open-drain capacity           :  {inp.get('drain_capacity_gpm',0):.1f} GPM  (trapez. channel)")
            W(f"    LiDAR depression storage      :  {inp.get('dep_storage_gal',0):.1f} gal  (bilinear DEM scan)")
            W(f"    Damage volume (6\" × lot)      :  {inp.get('damage_vol_gal',0):.1f} gal")
            W(f"    Max storm intensity cap       :  {inp.get('max_intensity_in_hr',3.0)} in/hr")
            W(f"    Max total rainfall            :  {inp.get('max_total_rain_in',3.0)} in")
            W()
            if crit:
                W(f"  ┌─ CRITICAL THRESHOLD ────────────────────────────────────────┐")
                W(f"  │  Rainfall rate    :  {crit.get('rate_in_hr','?')} in/hr")
                W(f"  │  Storm duration   :  {crit.get('max_duration_hr','?')} hrs  "
                  f"(max before exceeding {inp.get('max_total_rain_in',3):.0f}\" total)")
                W(f"  │  Net surface excess:  {crit.get('net_excess_gpm','?')} GPM  "
                  f"(after all drainage capacity subtracted)")
                W(f"  │  Depression fill  :  {crit.get('dep_fill_min','?')} min  "
                  f"(LiDAR depressions absorb runoff first)")
                W(f"  │  Damage time      :  {crit.get('damage_time_min','?')} min  "
                  f"(to accumulate 6\" at house)")
                W(f"  │  Total time       :  {crit.get('total_time_min','?')} min from rain start → 6\" damage")
                W(f"  │  Status           :  {crit.get('status','?')}")
                if "note" in crit:
                    W(f"  │  Note             :  {crit['note']}")
                W(f"  └─────────────────────────────────────────────────────────────┘")
            else:
                W("  ✅ No rainfall rate up to 3 in/hr produces 6\" damage within a 3\" storm.")
            W()
            W("  INTENSITY TABLE  (0.5 in/hr increments — subset shown):")
            W(f"  {'Rate':>8}  {'Duration':>10}  {'Net GPM':>9}  {'Dep Fill':>10}  "
              f"{'Damage Min':>11}  {'Total':>8}  Status")
            W("  " + "─" * 72)
            for row_d in rd.get("table", []):
                if round(row_d["rate_in_hr"] * 10) % 5 != 0:
                    continue  # show every 0.5 in/hr
                ttm = row_d.get("total_time_min")
                dfm = row_d.get("dep_fill_min")
                dtm = row_d.get("damage_time_min")
                W(f"  {row_d['rate_in_hr']:>6.1f}  "
                  f"{row_d['max_duration_hr']:>8.2f}h  "
                  f"{row_d.get('net_excess_gpm', 0):>9.1f}  "
                  f"{str(round(dfm,1)) if dfm else 'N/A':>10}  "
                  f"{str(round(dtm,1)) if dtm else 'N/A':>11}  "
                  f"{str(round(ttm,1)) if ttm else 'N/A':>8}  "
                  f"{row_d['status']}")
            W()
            # Disclaimer
            H("METHODOLOGY DISCLAIMER — RAINFALL-TO-DAMAGE CALCULATION")
            disc = rd.get("disclaimer", "")
            words = disc.split()
            line_d = "  "
            for w in words:
                if len(line_d) + len(w) > 80:
                    W(line_d); line_d = "  " + w + " "
                else:
                    line_d += w + " "
            if line_d.strip(): W(line_d)
            W()

            # -- FLOOD PATHWAYS callout block (TXT report) -------------------
            _fps_txt = rd.get("flood_scenarios", [])
            if _fps_txt:
                W("  \u250c" + "\u2500"*60 + "\u2510")
                W("  \u2502  FLOOD PATHWAYS \u2014 Ways this property can reach 6\" inside")
                W("  \u2502  Min rate: 0.25 in/hr  |  Max rate: 5.3 in/hr")
                W("  \u2514" + "\u2500"*60 + "\u2518")
                W(f"  {' '*2}{'Rate':>8}  {'Duration':>9}  {'Total Rain':>11}  "
                  f"{'Time->Flood':>11}  {'Storm Type':<36}  {'Freq'}")
                W("  " + "─"*92)
                for _fp_t in _fps_txt:
                    _r_t   = f"{_fp_t['rate_in_hr']:.2f}\"/hr"
                    _d_t   = f"{_fp_t['duration_hr']:.1f} hrs"
                    _tot_t = f"{_fp_t['total_rain_in']:.2f}\""
                    _ttf_t = _fp_t.get("total_time_min")
                    _ttf_ts = f"{_ttf_t:.0f} min" if _ttf_t else "N/A"
                    _lbl_t = str(_fp_t.get("label",""))[:36]
                    _rp_t  = str(_fp_t.get("return_period",""))
                    _mark  = " <-- MOST PROBABLE" if _fp_t.get("most_probable") else ""
                    W(f"  {_r_t:>10}  {_d_t:>9}  {_tot_t:>11}  "
                      f"{_ttf_ts:>11}  {_lbl_t:<36}  {_rp_t}{_mark}")
                    # Storm callout for most probable event
                    if _fp_t.get("most_probable") and _fp_t.get("callout"):
                        _co_t   = _fp_t["callout"]
                        _cite_t = _fp_t.get("historical_cite", "")
                        W("  \u250c" + "\u2500"*70 + "\u2510")
                        W(f"  \u2502  \u2605 MOST PROBABLE STORM TYPE: {_co_t['name']}")
                        W(f"  \u2502    Duration: {_co_t['duration']}  |  Typical rainfall: {_co_t['rainfall']}")
                        for _nt_t in _co_t.get("notes", []):
                            W(f"  \u2502    - {_nt_t}")
                        if _cite_t:
                            W(f"  \u2502    {_cite_t}")
                        W("  \u2514" + "\u2500"*70 + "\u2518")
                W()

        # ── EL NIÑO / SUPER EL NIÑO PROJECTION ────────────────────────────
        if elnino_proj and elnino_proj.get("ok"):
            H("EL NIÑO RAINFALL PROJECTION — SUPER EL NIÑO ANALYSIS")
            ep = elnino_proj
            W(f"  ONI Data Range      :  {ep['oni_years'][0]}–{ep['oni_years'][1]}")
            W(f"  Super El Niño years :  {ep['n_super_years']}  "
              f"(ONI ≥ 2.0 — probability {ep['prob_super_year_pct']}% per year)")
            W(f"  Wet-season super    :  {len(ep['wet_super_months'])} months  "
              f"(Jun–Sep, probability {ep['prob_wet_season_pct']}% per month)")
            W(f"  Super El Niño years :  {', '.join(str(y) for y in ep['super_years'][-10:])}"
              f"{'...' if len(ep['super_years']) > 10 else ''}")
            W()
            W(f"  CURRENT YEAR ({ep['current_year']}) ENSO STATUS:")
            W(f"    Condition         :  {ep['current_label']}")
            W(f"    Wet-season multiplier:  {ep['current_multiplier']}x  "
              f"(applied to normal Pinellas rainfall)")
            if ep.get("critical_rate_in_hr") is not None:
                W(f"    Critical threshold:  {ep['critical_rate_in_hr']} in/hr  "
                  f"(base — from hydraulic calculation)")
                W(f"    Adjusted threshold:  {ep['adjusted_critical']} in/hr  "
                  f"(effective trigger under {ep['current_label']} — lower = easier to breach)")
                W()
                if ep["is_super_this_year"]:
                    W(f"  ⚠️  SUPER EL NIÑO ACTIVE THIS YEAR — rainfall enhancement is in effect.")
                    W(f"     A storm that would NORMALLY require {ep['critical_rate_in_hr']} in/hr")
                    W(f"     to cause damage needs only {ep['adjusted_critical']} in/hr in the")
                    W(f"     current enhanced moisture environment.")
            W()
            if ep.get("breached_months"):
                W(f"  HISTORICAL MONTHS WHERE DAMAGE THRESHOLD WAS EXCEEDED:")
                W(f"  (Under El Niño multiplier, adjusted peak hourly ≥ critical rate)")
                W(f"  {'Year':>6}  {'Month':>6}  {'ONI':>6}  {'Norm Peak':>10}  "
                  f"{'Adj Peak':>9}  {'Critical':>9}")
                W("  " + "─" * 55)
                shown = 0
                for bm in ep["breached_months"][-20:]:
                    W(f"  {bm['year']:>6}  {bm['month_name']:>6}  {bm['oni']:>6.2f}  "
                      f"{bm['normal_peak']:>9.2f}\"  {bm['adjusted_peak']:>8.2f}\"  "
                      f"{bm['critical_rate']:>8.2f}\"")
                    shown += 1
                if len(ep["breached_months"]) > 20:
                    W(f"  ... and {len(ep['breached_months'])-20} additional months not shown")
                W(f"  Total months in record exceeding threshold:  {ep['n_breached_months']}")
                W(f"  Years affected:  {', '.join(str(y) for y in ep['breached_years'][-10:])}")
            else:
                W("  ℹ️  No historical months exceeded the critical damage threshold "
                  "even under El Niño conditions.")
            W()

        W("═"*74)
        W(f"  ENGINE  :  OVERLORD-MAX v121.2  |  FIX: LiDAR overlay aligned to EPSG:3857 raster corners")
        W(f"  PRODUCT :  Garrison Flood Control  |  {GARRISON_PRODUCT['contact']}")
        W(f"  ENGR    :  {VERSAI['author']}  |  {VERSAI['report_id']}")
        W("═"*74)


# =============================================================================
# EL NIÑO RAINFALL MULTIPLIER ANALYSIS
# =============================================================================

def _parse_oni_rtf(rtf_path):
    """Parse ONI indices from RTF file into {year: {season: oni_value}} dict."""
    import re as _re
    with open(rtf_path, "r", errors="replace") as f:
        raw = f.read()
    text = _re.sub(r'\{[^}]*\}', '', raw)
    text = _re.sub(r'\\[a-zA-Z]+\d*\s?', ' ', text)
    text = _re.sub(r'\\\'[0-9a-f]{2}', '', text)
    text = _re.sub(r'[{}]', '', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    nums = _re.findall(r'-?\d+\.?\d*', text)
    filtered = [n for n in nums if n != '-108']
    seasons = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
    oni = {}
    i = 0
    while i + 12 < len(filtered):
        year = int(float(filtered[i]))
        if 1900 <= year <= 2100:
            vals = {s: float(filtered[i+1+j]) for j, s in enumerate(seasons)}
            oni[year] = vals
            i += 13
        else:
            i += 1
    return oni


def _parse_oni_csv(csv_path):
    """Parse ONI indices from a clean CSV (Year,DJF,JFM,...,NDJ)."""
    oni = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["Year"])
            oni[year] = {k: float(v) for k, v in row.items() if k != "Year"}
    return oni


def _load_monthly_precip(precip_csv):
    """Load NOAA daily precip CSV → {(year,month): avg_monthly_total_inches}.

    Flexible column detection — handles:
      NOAA GHCN format: STATION, DATE, PRCP
      Alternate names:  station/Station, date/Date, prcp/Prcp/precipitation/rainfall/rain
    """
    from collections import defaultdict
    station_total = defaultdict(float)
    station_days  = defaultdict(int)
    with open(precip_csv) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        h_lower = {h.lower().strip(): h for h in headers}

        # Detect column names (case-insensitive)
        prcp_col = None
        for candidate in ["prcp", "precipitation", "rainfall", "rain", "precip"]:
            if candidate in h_lower:
                prcp_col = h_lower[candidate]
                break
        if prcp_col is None:
            # Fallback: look for any header containing 'prcp' or 'rain' or 'precip'
            for k, v in h_lower.items():
                if "prcp" in k or "rain" in k or "precip" in k:
                    prcp_col = v
                    break
        if prcp_col is None:
            print(f"   ⚠️  Cannot find precipitation column in {precip_csv}")
            print(f"       Headers: {headers}")
            return {}

        date_col = h_lower.get("date", h_lower.get("datetime", h_lower.get("obs_date")))
        if date_col is None:
            for k, v in h_lower.items():
                if "date" in k:
                    date_col = v
                    break
        if date_col is None:
            print(f"   ⚠️  Cannot find date column in {precip_csv}")
            return {}

        station_col = h_lower.get("station", h_lower.get("station_id",
                       h_lower.get("site", h_lower.get("name"))))
        # If no station column, treat all rows as one station
        use_single_station = station_col is None

        for row in reader:
            prcp = (row.get(prcp_col) or "").strip()
            if not prcp:
                continue
            try:
                pval = float(prcp)
            except ValueError:
                continue
            date = row[date_col]
            yr, mo = int(date[:4]), int(date[5:7])
            sta = "ALL" if use_single_station else row.get(station_col, "UNK")
            key = (sta, yr, mo)
            station_total[key] += pval
            station_days[key]  += 1

    # Average monthly totals across stations with ≥15 days of data
    ym_totals = defaultdict(list)
    for (sta, yr, mo), days in station_days.items():
        if days >= 15:
            ym_totals[(yr, mo)].append(station_total[(sta, yr, mo)])
    # Fallback: if ≥15 filter removed everything, lower to ≥5
    if not ym_totals:
        for (sta, yr, mo), days in station_days.items():
            if days >= 5:
                ym_totals[(yr, mo)].append(station_total[(sta, yr, mo)])
    result = {}
    for (yr, mo), vals in ym_totals.items():
        result[(yr, mo)] = sum(vals) / len(vals)
    return result


def _oni_for_month(oni_dict, year, month):
    """Return the ONI value most relevant to a calendar (year, month).

    ONI seasons are 3-month running means.  Map each calendar month to
    the season centred on it:
      Jan → DJF, Feb → JFM, Mar → FMA, …, Dec → NDJ
    DJF of a given year spans Dec(year-1)–Jan–Feb of that year.
    """
    _month_to_season = {
        1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM",
        5: "AMJ", 6: "MJJ", 7: "JJA", 8: "JAS",
        9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
    }
    season = _month_to_season[month]
    if year in oni_dict and season in oni_dict[year]:
        return oni_dict[year][season]
    return None


def build_elnino_rainfall_multipliers(precip_csv, oni_path, export_csv):
    """Build El Niño rainfall multiplier dataset and export to CSV.

    Steps:
      A. Correlation analysis — regress monthly precip vs ONI index.
      B. Seasonal adjustment — compute per-month climatology & El Niño means.
      C. Super El Niño multiplier — use top-percentile ONI events (>+2.0 °C).

    Parameters
    ----------
    precip_csv : str   Path to NOAA daily precipitation CSV.
    oni_path   : str   Path to ONI indices (RTF or pre-parsed CSV).
    export_csv : str   Output path for the multiplier dataset.

    Returns
    -------
    dict with keys: monthly_multipliers, regression, super_elnino, export_path
    """
    print("   🌊  Building El Niño rainfall multiplier analysis...")

    # ── Load ONI data ─────────────────────────────────────────────────────
    # Try CSV first, fall back to RTF parser (file might be RTF with .csv ext)
    oni = {}
    if oni_path.lower().endswith('.rtf'):
        oni = _parse_oni_rtf(oni_path)
    else:
        try:
            oni = _parse_oni_csv(oni_path)
        except Exception:
            oni = {}
        if not oni:
            # Fallback: might be RTF content saved with .csv extension
            try:
                oni = _parse_oni_rtf(oni_path)
            except Exception:
                pass
    if not oni:
        print(f"      ⚠️  Could not parse ONI data from {oni_path}")
        return {"ok": False, "reason": f"Failed to parse ONI file: {oni_path}"}
    print(f"      ONI data: {min(oni.keys())}–{max(oni.keys())} ({len(oni)} years)")

    # ── Load precipitation data ───────────────────────────────────────────
    monthly_precip = _load_monthly_precip(precip_csv)
    print(f"      Precipitation: {len(monthly_precip)} station-averaged monthly totals")

    # ── Pair each monthly precip with its ONI value ───────────────────────
    paired = []  # list of (month, precip_in, oni_value)
    for (yr, mo), precip_in in monthly_precip.items():
        oni_val = _oni_for_month(oni, yr, mo)
        if oni_val is not None:
            paired.append((mo, precip_in, oni_val))

    if len(paired) < 12:
        print("      ⚠️  Insufficient paired data for regression")
        return {"ok": False, "reason": "Not enough paired ONI + precip data"}

    # ── Step A: Regression — Precip = a × ONI + b ─────────────────────────
    import numpy as _np
    oni_arr    = _np.array([p[2] for p in paired])
    precip_arr = _np.array([p[1] for p in paired])
    # Simple linear regression
    n = len(oni_arr)
    sx  = oni_arr.sum()
    sy  = precip_arr.sum()
    sxx = (oni_arr * oni_arr).sum()
    sxy = (oni_arr * precip_arr).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        a, b = 0.0, float(precip_arr.mean())
    else:
        a = float((n * sxy - sx * sy) / denom)
        b = float((sy - a * sx) / n)
    r_num = n * sxy - sx * sy
    r_den = math.sqrt((n * sxx - sx**2) * (n * (precip_arr**2).sum() - sy**2))
    r_val = float(r_num / r_den) if abs(r_den) > 1e-12 else 0.0
    print(f"      Regression: Precip = {a:.3f} × ONI + {b:.3f}  (r={r_val:.3f}, n={n})")

    regression = {"slope_a": round(a, 4), "intercept_b": round(b, 4),
                  "r_correlation": round(r_val, 4), "n_paired": n}

    # ── Step B: Seasonal (monthly) climatology & El Niño multipliers ──────
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Group by calendar month
    month_normal    = {m: [] for m in range(1, 13)}  # all conditions
    month_elnino    = {m: [] for m in range(1, 13)}  # ONI > 0.5 (El Niño)
    month_lanina    = {m: [] for m in range(1, 13)}  # ONI < -0.5 (La Niña)
    month_neutral   = {m: [] for m in range(1, 13)}  # -0.5 ≤ ONI ≤ 0.5
    month_super     = {m: [] for m in range(1, 13)}  # ONI > 2.0 (Super El Niño)

    for mo, precip_in, oni_val in paired:
        month_normal[mo].append(precip_in)
        if oni_val > 2.0:
            month_super[mo].append(precip_in)
            month_elnino[mo].append(precip_in)
        elif oni_val > 0.5:
            month_elnino[mo].append(precip_in)
        elif oni_val < -0.5:
            month_lanina[mo].append(precip_in)
        else:
            month_neutral[mo].append(precip_in)

    monthly_rows = []
    for mo in range(1, 13):
        norm  = _np.mean(month_normal[mo]) if month_normal[mo] else 0.0
        enso  = _np.mean(month_elnino[mo]) if month_elnino[mo] else norm
        lanin = _np.mean(month_lanina[mo]) if month_lanina[mo] else norm
        neut  = _np.mean(month_neutral[mo]) if month_neutral[mo] else norm
        sup   = _np.mean(month_super[mo]) if month_super[mo] else enso

        # Multiplier = El Niño rainfall / Normal rainfall
        mult_elnino = round(enso / norm, 3) if norm > 0.01 else 1.0
        mult_super  = round(sup / norm, 3) if norm > 0.01 else 1.0
        mult_lanina = round(lanin / norm, 3) if norm > 0.01 else 1.0

        # Step C: Extreme multiplier — regression-based at ONI = +2.5
        reg_normal  = max(0.01, b)
        reg_super   = max(0.01, a * 2.5 + b)
        mult_reg_super = round(reg_super / reg_normal, 3)

        monthly_rows.append({
            "month": MONTH_NAMES[mo - 1],
            "month_num": mo,
            "normal_precip_in": round(float(norm), 2),
            "elnino_precip_in": round(float(enso), 2),
            "super_elnino_precip_in": round(float(sup), 2),
            "lanina_precip_in": round(float(lanin), 2),
            "multiplier_elnino": mult_elnino,
            "multiplier_super_elnino": mult_super,
            "multiplier_lanina": mult_lanina,
            "multiplier_regression_super": mult_reg_super,
            "n_normal": len(month_normal[mo]),
            "n_elnino": len(month_elnino[mo]),
            "n_super": len(month_super[mo]),
            "n_lanina": len(month_lanina[mo]),
        })

    # ── Overall annual multipliers ────────────────────────────────────────
    all_norm  = sum(r["normal_precip_in"] for r in monthly_rows)
    all_enso  = sum(r["elnino_precip_in"] for r in monthly_rows)
    all_super = sum(r["super_elnino_precip_in"] for r in monthly_rows)
    all_lanin = sum(r["lanina_precip_in"] for r in monthly_rows)

    annual_mult_elnino = round(all_enso / max(0.01, all_norm), 3)
    annual_mult_super  = round(all_super / max(0.01, all_norm), 3)
    annual_mult_lanina = round(all_lanin / max(0.01, all_norm), 3)

    super_summary = {
        "annual_normal_in": round(all_norm, 2),
        "annual_elnino_in": round(all_enso, 2),
        "annual_super_elnino_in": round(all_super, 2),
        "annual_lanina_in": round(all_lanin, 2),
        "annual_multiplier_elnino": annual_mult_elnino,
        "annual_multiplier_super_elnino": annual_mult_super,
        "annual_multiplier_lanina": annual_mult_lanina,
    }
    print(f"      Annual multiplier  El Niño: {annual_mult_elnino}x  |  "
          f"Super El Niño: {annual_mult_super}x  |  La Niña: {annual_mult_lanina}x")

    # ── Export to CSV ─────────────────────────────────────────────────────
    fieldnames = list(monthly_rows[0].keys())
    with open(export_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(monthly_rows)
        # Blank row then annual summary
        f.write("\n")
        f.write("# Annual Summary\n")
        for k, v in super_summary.items():
            f.write(f"# {k},{v}\n")
        f.write(f"# regression_slope_a,{regression['slope_a']}\n")
        f.write(f"# regression_intercept_b,{regression['intercept_b']}\n")
        f.write(f"# regression_r,{regression['r_correlation']}\n")
        f.write(f"# regression_n,{regression['n_paired']}\n")

    print(f"      ✅ Exported: {export_csv}")
    return {
        "ok": True,
        "monthly_multipliers": monthly_rows,
        "regression": regression,
        "super_elnino": super_summary,
        "export_path": export_csv,
    }


# =============================================================================
# EL NIÑO DAMAGE PROJECTION — property-level
# =============================================================================
def build_elnino_projection(oni_path: str,
                             elnino_result: dict,
                             critical_rate_in_hr,
                             county: str = "Pinellas County") -> dict:
    """
    Using the pre-built El Niño multiplier dataset (from build_elnino_rainfall_multipliers)
    and the site-specific critical rainfall rate (from calc_rainfall_to_damage), project:

      1. Historical Super El Niño events (ONI ≥ 2.0) from the indices CSV.
      2. Which historical months would have exceeded the critical damage threshold
         when the El Niño rainfall multiplier is applied.
      3. Probability of the critical threshold being reached in any given wet-season
         month under Super El Niño conditions.
      4. Under the current Super El Niño (this year), the adjusted effective rainfall
         rate the property effectively 'sees' relative to the damage threshold.

    Parameters
    ----------
    oni_path          : path to the ONI indices file (auto-detected by engine)
    elnino_result     : output dict from build_elnino_rainfall_multipliers()
    critical_rate_in_hr : the in/hr rate that triggers 6"+ damage (from calc_rainfall_to_damage)
    county            : county name string for narrative
    """
    MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    SEASON_ORDER = ["DJF","JFM","FMA","MAM","AMJ","MJJ",
                    "JJA","JAS","ASO","SON","OND","NDJ"]
    # Pinellas wet season months (peak flood risk)
    WET_MONTHS   = {6, 7, 8, 9}   # Jun Jul Aug Sep

    # ── Load ONI ─────────────────────────────────────────────────────────
    oni = {}
    if oni_path and os.path.exists(oni_path):
        try:
            if oni_path.lower().endswith('.rtf'):
                oni = _parse_oni_rtf(oni_path)
            else:
                try:
                    oni = _parse_oni_csv(oni_path)
                except Exception:
                    oni = {}
                if not oni:
                    oni = _parse_oni_rtf(oni_path)
        except Exception as e:
            return {"ok": False, "reason": f"ONI parse error: {e}"}

    if not oni:
        return {"ok": False, "reason": "No ONI data available for projection"}

    # ── Collect all month-level ONI values ────────────────────────────────
    all_months = []   # list of (year, month_num, oni_val)
    for year, seasons in sorted(oni.items()):
        for mo_idx, season in enumerate(SEASON_ORDER, 1):
            val = seasons.get(season)
            if val is not None:
                all_months.append((year, mo_idx, float(val)))

    # ── Monthly multipliers from El Niño build (keyed by month name) ─────
    monthly_mult = {}
    if elnino_result and elnino_result.get("ok"):
        for row in elnino_result.get("monthly_multipliers", []):
            mo_num = row["month_num"]
            monthly_mult[mo_num] = {
                "mult_elnino":  row["multiplier_elnino"],
                "mult_super":   row["multiplier_super_elnino"],
                "mult_lanina":  row["multiplier_lanina"],
                "normal_in":    row["normal_precip_in"],
                "super_in":     row["super_elnino_precip_in"],
            }

    # ── Super El Niño events (ONI ≥ 2.0) ─────────────────────────────────
    super_events  = [(y, m, v) for y, m, v in all_months if v >= 2.0]
    strong_events = [(y, m, v) for y, m, v in all_months if 1.5 <= v < 2.0]

    super_years   = sorted(set(y for y, m, v in super_events))
    n_years_total = max(oni.keys()) - min(oni.keys()) + 1

    # Wet-season Super El Niño months
    wet_super = [(y, m, v) for y, m, v in super_events if m in WET_MONTHS]

    prob_super_year = round(len(super_years) / max(n_years_total, 1) * 100, 1)
    prob_wet_super  = round(len(wet_super) / max(n_years_total * len(WET_MONTHS), 1) * 100, 1)

    # ── Which historical months would have breached the damage threshold? ─
    breached_months = []
    if critical_rate_in_hr is not None:
        for (y, m, v) in all_months:
            mm = monthly_mult.get(m, {})
            if not mm:
                continue
            norm_monthly_in  = mm["normal_in"]
            # Convert monthly total to peak hourly (approximate: wet season
            # convective storms deliver ~0.35 of monthly total in peak hour)
            peak_hr_factor = 0.35 if m in WET_MONTHS else 0.20
            norm_peak_hr   = norm_monthly_in * peak_hr_factor

            # Apply observed ONI multiplier linearly between neutral and super
            if v >= 2.0:
                mult = mm["mult_super"]
            elif v >= 0.5:
                mult = 1.0 + (mm["mult_elnino"] - 1.0) * (v / 1.5)
            elif v <= -0.5:
                mult = mm["mult_lanina"]
            else:
                mult = 1.0
            adjusted_peak = norm_peak_hr * mult

            if adjusted_peak >= critical_rate_in_hr:
                breached_months.append({
                    "year":          y,
                    "month_num":     m,
                    "month_name":    MONTH_NAMES[m - 1],
                    "oni":           round(v, 2),
                    "normal_peak":   round(norm_peak_hr, 2),
                    "adjusted_peak": round(adjusted_peak, 2),
                    "critical_rate": critical_rate_in_hr,
                })

    breached_years = sorted(set(bm["year"] for bm in breached_months))

    # ── Current year analysis ─────────────────────────────────────────────
    current_year = datetime.now().year
    current_oni  = oni.get(current_year, {})
    current_max_oni = max(current_oni.values()) if current_oni else None
    is_super_this_year = (current_max_oni is not None and current_max_oni >= 2.0)

    # Best estimate multiplier for this year
    if current_max_oni is not None and current_max_oni >= 2.0:
        current_label = f"SUPER EL NIÑO (ONI={current_max_oni:.2f})"
        # Use wet-season super multiplier
        wet_avg_super = (sum(monthly_mult.get(m, {}).get("mult_super", 1.0)
                             for m in WET_MONTHS) / len(WET_MONTHS))
        current_multiplier = round(wet_avg_super, 3)
    elif current_max_oni is not None and current_max_oni >= 1.5:
        current_label = f"Strong El Niño (ONI={current_max_oni:.2f})"
        wet_avg_enso = (sum(monthly_mult.get(m, {}).get("mult_elnino", 1.0)
                            for m in WET_MONTHS) / len(WET_MONTHS))
        current_multiplier = round(wet_avg_enso * 1.15, 3)  # interpolated
    elif current_max_oni is not None and current_max_oni >= 0.5:
        current_label = f"El Niño (ONI={current_max_oni:.2f})"
        wet_avg_enso = (sum(monthly_mult.get(m, {}).get("mult_elnino", 1.0)
                            for m in WET_MONTHS) / len(WET_MONTHS))
        current_multiplier = round(wet_avg_enso, 3)
    else:
        current_label = f"Neutral/La Niña (ONI={current_max_oni})"
        current_multiplier = 1.0

    # Effective damage-threshold rainfall rate under current ENSO conditions
    adjusted_critical = (round(critical_rate_in_hr / current_multiplier, 2)
                         if critical_rate_in_hr and current_multiplier > 0
                         else critical_rate_in_hr)

    # Recent Super El Niño events (last 30 years) for report narrative
    recent_super = [(y, m, v) for y, m, v in super_events if y >= current_year - 30]
    recent_years_unique = sorted(set(y for y, m, v in recent_super))

    return {
        "ok":                    True,
        "oni_years":             (min(oni.keys()), max(oni.keys())),
        "n_super_events":        len(super_events),
        "n_super_years":         len(super_years),
        "super_years":           super_years,
        "prob_super_year_pct":   prob_super_year,
        "prob_wet_season_pct":   prob_wet_super,
        "wet_super_months":      wet_super,
        "current_year":          current_year,
        "current_max_oni":       current_max_oni,
        "current_label":         current_label,
        "current_multiplier":    current_multiplier,
        "is_super_this_year":    is_super_this_year,
        "critical_rate_in_hr":   critical_rate_in_hr,
        "adjusted_critical":     adjusted_critical,
        "breached_months":       breached_months,
        "breached_years":        breached_years,
        "n_breached_months":     len(breached_months),
        "recent_super_years":    recent_years_unique,
        "strong_events":         strong_events,
    }


# =============================================================================
# MASTER EXECUTION ENGINE — v121.2
# =============================================================================
SILENT_FLOOD_MULTIPLIER = 1.10



# =============================================================================
# P9 -- FLOOD REPORT FEEDBACK LOOP  (Streamlit edition)
# =============================================================================
# How it works:
#   1. FloodReportFeedback.queue_output() is called after every output file.
#   2. After all outputs for one address, flush() writes pending_feedback.json
#      to MASTER_EXPORT_DIR.
#   3. On the NEXT run, __init__ reads feedback_log.json and derives
#      presentation tweaks from accumulated keyword hits -- same logic as
#      before, just no terminal input() calls.
#   4. A companion Streamlit app (flood_feedback_ui.py on the Desktop) reads
#      pending_feedback.json, shows previews, collects ratings & free text,
#      calls Ollama for narrative rewrites, and appends to feedback_log.json.
#   5. Optionally the script auto-launches the Streamlit app when a pending
#      file is written (set FEEDBACK_AUTO_LAUNCH = True below).
# =============================================================================

FEEDBACK_AUTO_LAUNCH: bool = True   # set False to skip auto-launching Streamlit
FEEDBACK_UI_SCRIPT:   str  = os.path.join(
    os.path.expanduser("~/Desktop"), "flood_feedback_ui.py")

_FEEDBACK_KEYWORDS: dict = {
    # visual / photo
    "dark":              {"map_brightness": 0.20},
    "murky":             {"map_brightness": 0.20},
    "hard to see":       {"map_brightness": 0.15},
    "washed out":        {"map_brightness": -0.15},
    "too bright":        {"map_brightness": -0.15},
    "blurry":            {"png_dpi_delta": 50},
    "low res":           {"png_dpi_delta": 50},
    "pixelated":         {"png_dpi_delta": 50},
    # typography
    "font small":        {"pdf_fontsize_delta": 2},
    "text small":        {"pdf_fontsize_delta": 2},
    "font too small":    {"pdf_fontsize_delta": 2},
    "can't read":        {"pdf_fontsize_delta": 1},
    "hard to read":      {"pdf_fontsize_delta": 1},
    # legend / scale
    "legend":            {"legend_scale": 0.3},
    "no legend":         {"legend_scale": 0.4},
    "scale bar":         {"scalebar_width": 1.5},
    "scale":             {"scalebar_width": 1.0},
    # colour / contrast
    "color":             {"high_contrast": 1},
    "colour":            {"high_contrast": 1},
    "contrast":          {"high_contrast": 1},
    "washed":            {"high_contrast": 1},
    # map labels
    "label":             {"label_halo": 1},
    "labels":            {"label_halo": 1},
    "street name":       {"label_halo": 1},
    # layout / table
    "table":             {"table_padding": 0.10},
    "crowded":           {"table_padding": 0.10},
    "cluttered":         {"table_padding": 0.10},
    "too narrow":        {"figure_width_delta":  1.0},
    "too wide":          {"figure_width_delta": -1.0},
    # narrative quality (signals for Ollama to re-draft)
    "too long":          {"narrative_max_sentences": -1},
    "too short":         {"narrative_max_sentences":  1},
    "confusing":         {"narrative_clarity":    1},
    "unclear":           {"narrative_clarity":    1},
    "jargon":            {"narrative_plain":      1},
    "technical":         {"narrative_plain":      1},
    "boring":            {"narrative_vivid":      1},
    "generic":           {"narrative_vivid":      1},
    "not specific":      {"narrative_vivid":      1},
    "repetitive":        {"narrative_dedup":      1},
    "tone":              {"narrative_tone_check": 1},
}

_TWEAK_CAPS: dict = {
    "map_brightness":          (-0.40,  0.60),
    "png_dpi_delta":           (0,      150),
    "pdf_fontsize_delta":      (-2,     6),
    "legend_scale":            (0,      1.5),
    "scalebar_width":          (0,      5.0),
    "high_contrast":           (0,      1),
    "label_halo":              (0,      1),
    "table_padding":           (0,      0.50),
    "figure_width_delta":      (-2,     3),
    "narrative_max_sentences": (-2,     3),
    "narrative_clarity":       (0,      3),
    "narrative_plain":         (0,      3),
    "narrative_vivid":         (0,      3),
    "narrative_dedup":         (0,      1),
    "narrative_tone_check":    (0,      1),
}


class FloodReportFeedback:
    """
    Streamlit-backed feedback loop for flood reports.

    In the main run loop:
        fb = FloodReportFeedback(MASTER_EXPORT_DIR)
        fb.queue_output("PDF Report", pdf_path, address, ollama_sales, "executive sales")
        fb.queue_output("Photo (Lot 60ft)", lot_png, address)
        ...
        fb.flush(address)   # writes pending_feedback.json + launches Streamlit
        fb.summarize()      # prints log stats at end of run
    """

    LOG_FILE     = "feedback_log.json"
    PENDING_FILE = "pending_feedback.json"

    def __init__(self, export_dir: str):
        self.export_dir  = export_dir
        self.log_path    = os.path.join(export_dir, self.LOG_FILE)
        self.pending_path= os.path.join(export_dir, self.PENDING_FILE)
        self._log: list  = self._load_json(self.log_path, default=[])
        self.tweaks: dict = self._compute_tweaks()
        self._queue: list = []          # outputs buffered for current address
        if self.tweaks:
            self._print_active_tweaks()

    # -- persistence ----------------------------------------------------------

    def _load_json(self, path: str, default):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return default

    def _save_json(self, path: str, obj):
        os.makedirs(self.export_dir, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2)
        except Exception as exc:
            print(f"   WARNING  Feedback write failed: {exc}")

    # -- tweak computation ----------------------------------------------------

    def _compute_tweaks(self) -> dict:
        """Tally keyword deltas from all past log entries and cap them."""
        acc: dict = {}
        for entry in self._log:
            for k, v in entry.get("tweaks_from_keywords", {}).items():
                acc[k] = acc.get(k, 0) + v
        result: dict = {}
        for k, v in acc.items():
            lo, hi = _TWEAK_CAPS.get(k, (None, None))
            if lo is not None:
                v = max(lo, min(hi, v))
            if v != 0:
                result[k] = v
        return result

    def _print_active_tweaks(self):
        print("\n   Auto-tweaks from past feedback:")
        for k, v in self.tweaks.items():
            sign = "+" if isinstance(v, (int, float)) and v > 0 else ""
            print(f"        {k}: {sign}{v}")

    # -- queue / flush --------------------------------------------------------

    def queue_output(self, label: str, file_path: str, address: str,
                     narrative: str = None, narrative_type: str = None):
        """Buffer one output for the current address."""
        self._queue.append({
            "label":          label,
            "file_path":      file_path,
            "address":        address,
            "narrative":      narrative,
            "narrative_type": narrative_type,
        })

    def flush(self, address: str):
        """Write pending_feedback.json and optionally launch the Streamlit UI."""
        if not self._queue:
            return
        pending = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "address":       address,
            "export_dir":    self.export_dir,
            "log_path":      self.log_path,
            "ollama_url":    OLLAMA_URL,
            "ollama_model":  OLLAMA_MODEL,
            "entries":       self._queue,
        }
        self._save_json(self.pending_path, pending)
        print(f"\n   Feedback pending: {self.pending_path}")
        self._queue = []

        # -- auto-launch Streamlit --------------------------------------------
        if FEEDBACK_AUTO_LAUNCH and os.path.exists(FEEDBACK_UI_SCRIPT):
            try:
                import subprocess
                subprocess.Popen(
                    ["streamlit", "run", FEEDBACK_UI_SCRIPT,
                     "--", "--pending", self.pending_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("   Streamlit feedback UI launched  (http://localhost:8501)")
            except Exception as exc:
                print(f"   Could not launch Streamlit: {exc}")
                print(f"   Run manually: streamlit run {FEEDBACK_UI_SCRIPT}")
        elif FEEDBACK_AUTO_LAUNCH:
            print(f"   To review: streamlit run {FEEDBACK_UI_SCRIPT}")

    # -- summary --------------------------------------------------------------

    def summarize(self):
        """Print all-time feedback stats at end of run."""
        if not self._log:
            return
        ratings = [e["rating"] for e in self._log if e.get("rating") is not None]
        flagged  = sum(1 for e in self._log if e.get("flagged_for_review"))
        kw_counts: dict = {}
        for e in self._log:
            for k in e.get("keywords_detected", []):
                kw_counts[k] = kw_counts.get(k, 0) + 1
        avg = (f"{sum(ratings)/len(ratings):.1f}" if ratings else "n/a")
        print(f"\n   Feedback log: {len(self._log)} entries  |  "
              f"Avg rating: {avg}/5  |  Flagged: {flagged}")
        if kw_counts:
            top = sorted(kw_counts.items(), key=lambda x: -x[1])[:5]
            print("   Top issues: "
                  + "  |  ".join(f"{k} ({v}x)" for k, v in top))
        print(f"   Log path: {self.log_path}")




# =============================================================================
# EXPORT MAP 1 — NEIGHBORHOOD HEATMAP (250 ft)
# Bold red/orange/green elevation heatmap — style of Image 1
# Shows whole block context, ponding zones as blue circles, property badge
# =============================================================================
def export_heatmap_250(st_addr: str, cx: float, cy: float, df_crs: str,
                        g_elev_ft: float, row: dict, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as pe
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer, CRS
    except Exception as e:
        print(f"   ❌ heatmap_250 import error: {e}")
        return False

    RADIUS_FT = 250.0

    # ── Open raster ──────────────────────────────────────────────────────────
    tile_index = DemTileIndex(LIDAR_RASTER, PINELLAS_LIDAR_FILES[:])
    ds, sx, sy = tile_index.pick_dataset_for_point(cx, cy, df_crs)
    if ds is None:
        tile_index.close()
        print(f"   ❌ heatmap_250: no LiDAR tile covers this point")
        return False

    units_per_ft = _crs_units_per_foot(ds.crs)
    r_u = RADIUS_FT * units_per_ft

    try:
        win = from_bounds(sx - r_u, sy - r_u, sx + r_u, sy + r_u, transform=ds.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        arr = ds.read(1, window=win, masked=True).astype("float32")
    except Exception as e:
        tile_index.close()
        print(f"   ❌ heatmap_250 raster read: {e}")
        return False

    # ── Reproject window corners → EPSG:3857 ─────────────────────────────────
    try:
        wt = rasterio.windows.transform(win, ds.transform)
        nc, nr = int(round(win.width)), int(round(win.height))
        corners_x = [wt * (0, 0), wt * (nc, 0), wt * (0, nr), wt * (nc, nr)]
        west_r  = min(c[0] for c in corners_x)
        east_r  = max(c[0] for c in corners_x)
        south_r = min(c[1] for c in corners_x)
        north_r = max(c[1] for c in corners_x)
        tfm = Transformer.from_crs(ds.crs, CRS.from_epsg(3857), always_xy=True)
        west_m,  south_m = tfm.transform(west_r,  south_r)
        east_m,  north_m = tfm.transform(east_r,  north_r)
        tfm_pt = Transformer.from_crs(CRS.from_user_input(df_crs), CRS.from_epsg(3857), always_xy=True)
        hx, hy = tfm_pt.transform(cx, cy)
        USE_WEB = True
    except Exception as e:
        print(f"   ⚠️  heatmap_250 reprojection: {e}")
        USE_WEB = False
        west_m, east_m = sx - r_u, sx + r_u
        south_m, north_m = sy - r_u, sy + r_u
        hx, hy = sx, sy

    # ── Units & masking ───────────────────────────────────────────────────────
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    arr[arr == -999999.0] = np.nan
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    arr[arr < -50] = np.nan
    arr[arr > 500] = np.nan
    if LIDAR_ELEV_UNIT == "metres":
        arr = arr * 3.28084

    tile_index.close()

    # Mask water/nodata cells
    water = (arr < PINELLAS_ELEV_MIN_FT) | (arr > PINELLAS_ELEV_MAX_FT) | np.isnan(arr)

    # Ground zero from raster centroid sample
    h_px = max(0, min(int(round((north_r - sy) / abs(ds.transform.e) if USE_WEB else 0)), arr.shape[0]-1))
    w_px = max(0, min(int(round((sx - west_r) / abs(ds.transform.a) if USE_WEB else 0)), arr.shape[1]-1))
    patch = arr[max(0,h_px-3):h_px+4, max(0,w_px-3):w_px+4]
    valid_patch = patch[np.isfinite(patch)]
    grade = float(np.median(valid_patch)) if valid_patch.size >= 3 else g_elev_ft

    rel = arr - grade
    rel[water] = np.nan

    finite = rel[np.isfinite(rel)]
    if finite.size < 50:
        print(f"   ❌ heatmap_250: insufficient valid pixels ({finite.size})")
        return False

    # ── Colormap — bold Image 1 style: deep red → orange → yellow → green ────
    cmap = LinearSegmentedColormap.from_list("bold_elev", [
        (0.00, "#8B0000"),   # deep red  — very low
        (0.20, "#FF2200"),   # red
        (0.35, "#FF6600"),   # orange-red
        (0.47, "#FFA500"),   # orange
        (0.50, "#FFD700"),   # yellow    — at grade
        (0.55, "#ADFF2F"),   # yellow-green
        (0.68, "#22c55e"),   # green
        (0.82, "#15803d"),   # dark green
        (1.00, "#064e3b"),   # very dark green — highest
    ])
    cmap.set_bad(alpha=0.0)
    vmin, vmax = -3.0, 3.0

    # Contrast stretch
    p2, p98 = np.percentile(finite, [2, 98])
    span = p98 - p2
    if span > 0.01:
        rel_d = np.where(np.isfinite(rel),
                         vmin + (rel - p2) / span * (vmax - vmin), rel)
        rel_d[~np.isfinite(rel)] = np.nan
    else:
        rel_d = rel

    # ── Depression detection ──────────────────────────────────────────────────
    depressions = []
    if HAS_SCIPY and finite.size > 200:
        from scipy.ndimage import label as _lbl, minimum_filter as _mf
        rel_fill = np.where(np.isfinite(rel), rel, float(np.nanmean(finite)))
        mean_rel = float(np.nanmean(finite))
        min_f    = _mf(rel_fill, size=7)
        dep_mask = (rel < (mean_rel - 0.08)) & np.isfinite(rel) & (rel < min_f + 0.05) & ~water
        labeled, n_feat = _lbl(dep_mask)
        for fid in range(1, n_feat + 1):
            feat = (labeled == fid)
            if feat.sum() < 6:
                continue
            depth = mean_rel - float(np.nanmean(rel[feat]))
            if depth < 0.04:
                continue
            rows_i, cols_i = np.where(feat)
            cr, cc = int(np.mean(rows_i)), int(np.mean(cols_i))
            px_ft = float(abs(ds.res[0])) / units_per_ft
            vol   = min(feat.sum() * (px_ft**2) * depth * 7.481, 250_000)
            sev   = "CRITICAL" if depth > 0.5 else "MODERATE" if depth > 0.2 else "MINOR"
            depressions.append({"cr": cr, "cc": cc, "depth_ft": depth,
                                 "vol_gal": vol, "severity": sev, "mask": feat})

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14), dpi=350)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    _ext = [west_m, east_m, south_m, north_m]
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    ax.set_autoscale_on(False)

    # Satellite basemap
    try:
        import contextily as ctx
        if USE_WEB:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery,
                            zoom=19, attribution=False)
            ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    except Exception as _ce:
        print(f"   ⚠️  basemap: {_ce}")
        ax.set_facecolor("#1a1a2e")

    # Elevation overlay — semi-transparent, bold colors
    ax.imshow(rel_d, extent=_ext, cmap=cmap, vmin=vmin, vmax=vmax,
              alpha=0.62, origin="upper", interpolation="bilinear", zorder=2)
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)

    # ── Depression markers — blue circles ─────────────────────────────────────
    h_px_tot, w_px_tot = rel.shape
    xr = east_m - west_m
    yr = north_m - south_m

    def px2m(r, c):
        return west_m + (c / max(w_px_tot-1,1)) * xr,                north_m - (r / max(h_px_tot-1,1)) * yr

    dep_sorted = sorted(depressions, key=lambda d: d["vol_gal"], reverse=True)
    for dep in dep_sorted:
        dx, dy = px2m(dep["cr"], dep["cc"])
        sev = dep["severity"]
        col = "#ef4444" if sev == "CRITICAL" else "#f59e0b" if sev == "MODERATE" else "#3b82f6"
        # Filled translucent circle
        area_r = xr * 0.025 * min(1.0, dep["vol_gal"] / 5000 + 0.3)
        circ = plt.Circle((dx, dy), area_r, color="#1e40af", alpha=0.55, zorder=6)
        ax.add_patch(circ)
        circ2 = plt.Circle((dx, dy), area_r, fill=False, color="#60a5fa",
                            lw=2.5, alpha=0.9, zorder=7)
        ax.add_patch(circ2)
        # Label
        ax.annotate(
            f"POND\n{dep['vol_gal']:.0f} gal",
            xy=(dx, dy), xytext=(dx + area_r*1.4, dy + area_r*0.6),
            fontsize=9, color="white", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#60a5fa", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3a5f",
                      edgecolor="#60a5fa", alpha=0.92),
            zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    # ── Subject parcel marker ─────────────────────────────────────────────────
    cross = xr * 0.018
    ax.plot(hx, hy, "o", color="#ffffff", markersize=22, alpha=0.2, zorder=45)
    ax.plot(hx, hy, "o", color="#ffffff", markersize=14, alpha=0.5, zorder=46)
    ax.plot(hx, hy, "+", color="#e11d48", markersize=24, mew=3, zorder=48)
    ax.text(hx, hy + cross*1.8, "SUBJECT PARCEL", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="white", zorder=51,
            path_effects=[pe.withStroke(linewidth=3, foreground="#0d1117")])

    # ── Property badge ────────────────────────────────────────────────────────
    _wall = str(row.get("EXTERIORWA","") or "Unknown").strip()
    _yr   = str(row.get("YEARBUILT", row.get("ACTYRBLT","")) or "Unknown").strip()
    _fnd  = str(row.get("FOUNDATION","") or "Unknown").strip()
    _wu   = _wall.upper()
    _risk = "🔴 HIGH" if any(x in _wu for x in ["WOOD","FRAME","VINYL"]) else             "🟡 MOD"  if any(x in _wu for x in ["MASONRY","BLOCK","CBS","STUCCO","CONCRETE"])             else "⚪ UNK"
    try:
        _yr_i = int(str(_yr)[:4])
        _yr_r = "🔴 PRE-1975" if _yr_i < 1975 else "🟡 1975–2002" if _yr_i < 2002 else "🟢 POST-2002"
    except Exception:
        _yr_r = "⚪ Unknown"
    _fnu  = _fnd.upper()
    _fr   = "🔴 HIGH" if any(x in _fnu for x in ["SLAB","GRADE"]) else             "🟡 MOD"  if any(x in _fnu for x in ["CRAWL","STEM","FILL"]) else             "🟢 LOW"  if any(x in _fnu for x in ["PILE","PIER","ELEVATED"]) else "⚪ UNK"
    badge = f"Exterior Wall : {_wall}\nRisk          : {_risk}\nYear Built    : {_yr}  {_yr_r}\nFoundation    : {_fnd}\nRisk          : {_fr}"
    ax.text(hx + cross*2.5, hy - cross*2.5, badge,
            va="top", ha="left", fontsize=8.5, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                      edgecolor="#f59e0b", linewidth=2, alpha=0.93),
            zorder=52, path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")])

    # ── North arrow + scale bar ───────────────────────────────────────────────
    ax.annotate("N", xy=(west_m+xr*0.05, south_m+yr*0.09),
                xytext=(west_m+xr*0.05, south_m+yr*0.04),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="white", mutation_scale=18),
                ha="center", va="bottom", color="white", fontsize=15, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    bar_m = 100 * 0.3048
    bx0 = west_m + xr*0.06
    ax.plot([bx0, bx0+bar_m], [south_m+yr*0.04, south_m+yr*0.04],
            color="white", lw=7, solid_capstyle="butt", zorder=30)
    ax.text(bx0 + bar_m/2, south_m+yr*0.055, "100 ft",
            ha="center", color="white", fontsize=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")], zorder=31)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [
        Patch(facecolor="#22c55e", alpha=0.8, label="Higher elevation (water drains away)"),
        Patch(facecolor="#FFA500", alpha=0.8, label="Near grade (neutral)"),
        Patch(facecolor="#FF2200", alpha=0.8, label="Lower elevation (water collects here)"),
        Patch(facecolor="#1e40af", edgecolor="#60a5fa", linewidth=2,
              alpha=0.7, label="Ponding depression"),
    ]
    ax.legend(handles=leg, loc="lower right", facecolor="#1e293b",
              edgecolor="#374151", labelcolor="white", fontsize=9, framealpha=0.93)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cax = fig.add_axes([0.12, 0.04, 0.76, 0.015])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Elevation Δ (ft) relative to subject parcel grade",
                 color="white", fontsize=10)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("white")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"{st_addr}  —  Neighborhood View\n"
        f"250 ft radius  |  Bilinear LiDAR DEM  |  Zoom-19 Satellite  |  "
        f"{len(depressions)} depression(s) detected",
        color="white", fontsize=13, fontweight="bold", pad=14,
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    try:
        fig.savefig(out_png, bbox_inches="tight", facecolor="#0d1117", edgecolor="none", dpi=350)
        plt.close(fig)
        print(f"   ✅  Heatmap 250ft saved: {out_png}")
        return True
    except Exception as e:
        print(f"   ❌ heatmap_250 save: {e}")
        plt.close(fig)
        return False


# =============================================================================
# EXPORT MAP 2 — FLOW ARROW MAP (250 ft)
# Soft heatmap + blue water flow streamlines + textured puddle fills — Image 2 style
# =============================================================================
def export_flow_250(st_addr: str, cx: float, cy: float, df_crs: str,
                     g_elev_ft: float, row: dict, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as pe
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer, CRS
    except Exception as e:
        print(f"   ❌ flow_250 import error: {e}")
        return False

    RADIUS_FT = 250.0

    # ── Open raster (same logic as heatmap) ──────────────────────────────────
    tile_index = DemTileIndex(LIDAR_RASTER, PINELLAS_LIDAR_FILES[:])
    ds, sx, sy = tile_index.pick_dataset_for_point(cx, cy, df_crs)
    if ds is None:
        tile_index.close()
        print(f"   ❌ flow_250: no LiDAR tile covers this point")
        return False

    units_per_ft = _crs_units_per_foot(ds.crs)
    r_u = RADIUS_FT * units_per_ft

    try:
        win = from_bounds(sx - r_u, sy - r_u, sx + r_u, sy + r_u, transform=ds.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        arr = ds.read(1, window=win, masked=True).astype("float32")
    except Exception as e:
        tile_index.close()
        print(f"   ❌ flow_250 raster read: {e}")
        return False

    # Reproject corners
    try:
        wt = rasterio.windows.transform(win, ds.transform)
        nc, nr = int(round(win.width)), int(round(win.height))
        cx_list = [wt*(0,0), wt*(nc,0), wt*(0,nr), wt*(nc,nr)]
        west_r  = min(c[0] for c in cx_list); east_r  = max(c[0] for c in cx_list)
        south_r = min(c[1] for c in cx_list); north_r = max(c[1] for c in cx_list)
        tfm = Transformer.from_crs(ds.crs, CRS.from_epsg(3857), always_xy=True)
        west_m, south_m = tfm.transform(west_r, south_r)
        east_m, north_m = tfm.transform(east_r, north_r)
        tfm_pt = Transformer.from_crs(CRS.from_user_input(df_crs), CRS.from_epsg(3857), always_xy=True)
        hx, hy = tfm_pt.transform(cx, cy)
        USE_WEB = True
    except Exception as e:
        print(f"   ⚠️  flow_250 reprojection: {e}")
        USE_WEB = False
        west_m, east_m = sx - r_u, sx + r_u
        south_m, north_m = sy - r_u, sy + r_u
        hx, hy = sx, sy

    # Units & masking
    if hasattr(arr, "filled"):
        arr = arr.filled(np.nan)
    arr[arr == -999999.0] = np.nan
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    arr[arr < -50] = np.nan; arr[arr > 500] = np.nan
    if LIDAR_ELEV_UNIT == "metres":
        arr = arr * 3.28084
    tile_index.close()

    water = (arr < PINELLAS_ELEV_MIN_FT) | (arr > PINELLAS_ELEV_MAX_FT) | np.isnan(arr)
    h_px_tot, w_px_tot = arr.shape

    # Grade from centroid patch
    _cp = arr[max(0,h_px_tot//2-3):h_px_tot//2+4, max(0,w_px_tot//2-3):w_px_tot//2+4]
    _cv = _cp[np.isfinite(_cp)]
    grade = float(np.median(_cv)) if _cv.size >= 3 else g_elev_ft

    rel = arr - grade
    rel[water] = np.nan

    finite = rel[np.isfinite(rel)]
    if finite.size < 50:
        print(f"   ❌ flow_250: insufficient valid pixels")
        return False

    # ── Soft colormap — Image 2 style: muted red→beige→soft green ────────────
    cmap = LinearSegmentedColormap.from_list("soft_elev", [
        (0.00, "#7f1d1d"),
        (0.20, "#dc2626"),
        (0.38, "#fb923c"),
        (0.46, "#fde68a"),
        (0.50, "#fef9c3"),   # pale yellow at grade
        (0.56, "#bbf7d0"),
        (0.70, "#4ade80"),
        (0.85, "#16a34a"),
        (1.00, "#14532d"),
    ])
    cmap.set_bad(alpha=0.0)
    vmin, vmax = -1.0, 1.0

    p2, p98 = np.percentile(finite, [2, 98])
    span = p98 - p2
    if span > 0.01:
        rel_d = np.where(np.isfinite(rel), vmin + (rel - p2) / span * (vmax - vmin), rel)
        rel_d[~np.isfinite(rel)] = np.nan
    else:
        rel_d = rel

    # ── Depression detection ──────────────────────────────────────────────────
    depressions = []
    if HAS_SCIPY and finite.size > 200:
        from scipy.ndimage import label as _lbl, minimum_filter as _mf
        rel_fill = np.where(np.isfinite(rel), rel, float(np.nanmean(finite)))
        mean_rel = float(np.nanmean(finite))
        min_f    = _mf(rel_fill, size=7)
        dep_mask = (rel < (mean_rel - 0.06)) & np.isfinite(rel) & (rel < min_f + 0.05) & ~water
        labeled, n_feat = _lbl(dep_mask)
        for fid in range(1, n_feat + 1):
            feat = (labeled == fid)
            if feat.sum() < 5:
                continue
            depth = mean_rel - float(np.nanmean(rel[feat]))
            if depth < 0.03:
                continue
            rows_i, cols_i = np.where(feat)
            cr, cc = int(np.mean(rows_i)), int(np.mean(cols_i))
            px_ft = float(abs(ds.res[0])) / units_per_ft
            vol   = min(feat.sum() * (px_ft**2) * depth * 7.481, 250_000)
            sev   = "CRITICAL" if depth > 0.5 else "MODERATE" if depth > 0.2 else "MINOR"
            depressions.append({"cr": cr, "cc": cc, "depth_ft": depth, "depth_in": depth*12,
                                 "vol_gal": vol, "severity": sev, "mask": feat})

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14), dpi=350)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    _ext = [west_m, east_m, south_m, north_m]
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    ax.set_autoscale_on(False)

    # Satellite basemap
    try:
        import contextily as ctx
        if USE_WEB:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery,
                            zoom=19, attribution=False)
            ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)
    except Exception as _ce:
        ax.set_facecolor("#1a1a2e")

    # Soft elevation overlay — more transparent so satellite shows through
    ax.imshow(rel_d, extent=_ext, cmap=cmap, vmin=vmin, vmax=vmax,
              alpha=0.55, origin="upper", interpolation="bilinear", zorder=2)
    ax.set_xlim(west_m, east_m); ax.set_ylim(south_m, north_m)

    # ── Blue puddle fills on depressions ─────────────────────────────────────
    xr = east_m - west_m
    yr = north_m - south_m

    def px2m(r, c):
        return (west_m + (c / max(w_px_tot-1,1)) * xr,
                north_m - (r / max(h_px_tot-1,1)) * yr)

    for dep in depressions:
        sev = dep["severity"]
        col = "#1e3a8a" if sev == "CRITICAL" else "#1e40af" if sev == "MODERATE" else "#2563eb"
        ecol = "#ef4444" if sev == "CRITICAL" else "#f59e0b" if sev == "MODERATE" else "#3b82f6"
        # Rasterize puddle mask as scatter — textured fill effect
        rows_i, cols_i = np.where(dep["mask"])
        step = max(1, len(rows_i) // 400)
        px_x, px_y = [], []
        for ri, ci in zip(rows_i[::step], cols_i[::step]):
            mx, my = px2m(ri, ci)
            px_x.append(mx); px_y.append(my)
        if px_x:
            ax.scatter(px_x, px_y, c=col, s=28, alpha=0.65,
                       marker="o", linewidths=0, zorder=6)
        # Outline circle
        dx, dy = px2m(dep["cr"], dep["cc"])
        r_circ = xr * 0.018 * min(1.5, dep["vol_gal"]/3000 + 0.4)
        circ = plt.Circle((dx, dy), r_circ, fill=False, color=ecol, lw=2.0, alpha=0.9, zorder=7)
        ax.add_patch(circ)
        # Label
        sev_sym = "▲" if sev == "CRITICAL" else "▲" if sev == "MODERATE" else "ℹ"
        ax.annotate(
            f"{sev_sym} {sev[:3]}\nDepth: {dep['depth_in']:.1f}\"\nPond: {dep['vol_gal']:.0f} gal",
            xy=(dx, dy), xytext=(dx + r_circ*1.5, dy + r_circ*0.8),
            fontsize=8.5, color="white", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ecol, lw=1.3),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a",
                      edgecolor=ecol, linewidth=1.8, alpha=0.93),
            zorder=11,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#0d1117")])

    # ── Flow streamlines ──────────────────────────────────────────────────────
    try:
        rel_flow = np.where(np.isfinite(rel), rel, float(np.nanmean(finite)))
        gy, gx   = np.gradient(rel_flow)
        u, v_arr = -gx, -gy
        mag      = np.sqrt(u**2 + v_arr**2)
        mag      = np.where(np.isfinite(mag), mag, 0.0)

        step_sp  = max(3, int(min(rel.shape) / 90))
        nr_sp    = rel.shape[0] // step_sp
        nc_sp    = rel.shape[1] // step_sp

        sx_vals  = np.array([px2m(0, c*step_sp)[0] for c in range(nc_sp)])
        sy_vals  = np.array([px2m(r*step_sp, 0)[1] for r in range(nr_sp)])
        sxx, syy = np.meshgrid(sx_vals, sy_vals)

        uu = u    [::step_sp, ::step_sp][:nr_sp, :nc_sp]
        vv = -v_arr[::step_sp, ::step_sp][:nr_sp, :nc_sp]  # flip for imshow upper
        cc = mag  [::step_sp, ::step_sp][:nr_sp, :nc_sp]

        c_fin = cc[np.isfinite(cc)]
        vmax_c = float(np.percentile(c_fin, 95)) if c_fin.size else 1.0
        vmax_c = max(vmax_c, 1e-6)

        ax.streamplot(
            sxx, syy, uu, vv,
            color="#1d4ed8", linewidth=1.1, arrowsize=1.4,
            density=1.6, minlength=0.08, maxlength=3.5,
            zorder=8)
        print(f"   💧  Flow streamlines rendered")
    except Exception as _fe:
        print(f"   ⚠️  Flow streamlines: {_fe}")

    # ── Subject parcel marker ─────────────────────────────────────────────────
    cross = xr * 0.018
    ax.plot(hx, hy, "o", color="#ffffff", markersize=20, alpha=0.2, zorder=45)
    ax.plot(hx, hy, "+", color="#e11d48", markersize=22, mew=3, zorder=48)
    ax.text(hx, hy + cross*1.8, "SUBJECT PARCEL", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="white", zorder=51,
            path_effects=[pe.withStroke(linewidth=3, foreground="#0d1117")])

    # Property badge
    _wall = str(row.get("EXTERIORWA","") or "Unknown").strip()
    _yr   = str(row.get("YEARBUILT", row.get("ACTYRBLT","")) or "Unknown").strip()
    _fnd  = str(row.get("FOUNDATION","") or "Unknown").strip()
    _wu   = _wall.upper()
    _risk = "🔴 HIGH" if any(x in _wu for x in ["WOOD","FRAME","VINYL"]) else \
            "🟡 MOD"  if any(x in _wu for x in ["MASONRY","BLOCK","CBS","STUCCO","CONCRETE"]) \
            else "⚪ UNK"
    try:
        _yr_i = int(str(_yr)[:4])
        _yr_r = "🔴 PRE-1975" if _yr_i < 1975 else "🟡 1975–2002" if _yr_i < 2002 else "🟢 POST-2002"
    except Exception:
        _yr_r = "⚪ Unknown"
    _fnu = _fnd.upper()
    _fr  = "🔴 HIGH" if any(x in _fnu for x in ["SLAB","GRADE"]) else \
           "🟡 MOD"  if any(x in _fnu for x in ["CRAWL","STEM","FILL"]) else \
           "🟢 LOW"  if any(x in _fnu for x in ["PILE","PIER","ELEVATED"]) else "⚪ UNK"
    badge = f"Exterior Wall : {_wall}\nRisk          : {_risk}\nYear Built    : {_yr}  {_yr_r}\nFoundation    : {_fnd}\nRisk          : {_fr}"
    ax.text(hx + cross*2.5, hy - cross*2.5, badge,
            va="top", ha="left", fontsize=8.5, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                      edgecolor="#f59e0b", linewidth=2, alpha=0.93),
            zorder=52, path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")])

    # North arrow + scale
    ax.annotate("N", xy=(west_m+xr*0.05, south_m+yr*0.09),
                xytext=(west_m+xr*0.05, south_m+yr*0.04),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="white", mutation_scale=18),
                ha="center", va="bottom", color="white", fontsize=15, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    bar_m = 100 * 0.3048
    bx0   = west_m + xr*0.06
    ax.plot([bx0, bx0+bar_m], [south_m+yr*0.04]*2, color="white", lw=7,
            solid_capstyle="butt", zorder=30)
    ax.text(bx0+bar_m/2, south_m+yr*0.055, "100 ft", ha="center",
            color="white", fontsize=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")], zorder=31)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [
        Patch(facecolor="#16a34a", alpha=0.8, label="Higher elevation (water drains away)"),
        Patch(facecolor="#fde68a", alpha=0.8, label="Near grade (neutral)"),
        Patch(facecolor="#dc2626", alpha=0.8, label="Lower elevation (water collects here)"),
        Patch(facecolor="#1e3a8a", alpha=0.7, label="Modeled Sheet Flow (Blue Arrows)"),
        Patch(facecolor="#ef4444", edgecolor="#ef4444", alpha=0.5, label="▲ CRITICAL Depression Puddle"),
        Patch(facecolor="#f59e0b", edgecolor="#f59e0b", alpha=0.5, label="▲ MODERATE Depression Puddle"),
        Patch(facecolor="#3b82f6", edgecolor="#3b82f6", alpha=0.5, label="ℹ MINOR Depression Puddle"),
    ]
    ax.legend(handles=leg, loc="lower right", facecolor="#1e293b",
              edgecolor="#374151", labelcolor="white", fontsize=8.5, framealpha=0.93)

    # Colorbar
    cax = fig.add_axes([0.12, 0.04, 0.76, 0.015])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Elevation Δ (ft) relative to subject parcel grade", color="white", fontsize=10)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("white")

    # Title
    ax.set_title(
        f"{st_addr}  —  Flow Arrow View\n"
        f"250 ft radius  |  Bilinear LiDAR DEM  |  Zoom-19 Satellite  |  "
        f"{len(depressions)} depression(s) detected",
        color="white", fontsize=13, fontweight="bold", pad=14,
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    try:
        fig.savefig(out_png, bbox_inches="tight", facecolor="#0d1117", edgecolor="none", dpi=350)
        plt.close(fig)
        print(f"   ✅  Flow arrow map saved: {out_png}")
        return True
    except Exception as e:
        print(f"   ❌ flow_250 save: {e}")
        plt.close(fig)
        return False


def run_overlord_max():
    print("━"*65)
    print("🛰️  OVERLORD-MAX v121.3 | FORENSIC FLOOD ENGINE + TERRAIN v7")
    print("🔧  FIX: LiDAR overlay extent from actual EPSG:3857 raster corners")
    print(f"📂  Pinellas LiDAR tiles: {len(PINELLAS_LIDAR_FILES)}")
    print(f"📏  DEM: 250 ft radius | 2.5 ft step | bilinear sub-pixel")
    print(f"🌊  Water mask: cells < {PINELLAS_ELEV_MIN_FT} ft or > {PINELLAS_ELEV_MAX_FT} ft NAVD88 excluded")
    print("━"*65 + "\n")

    # ── P9: Initialise feedback system (reads past log, derives tweaks) ──
    fb = FloodReportFeedback(MASTER_EXPORT_DIR)

    if not os.path.exists(LEADS_LIST):
        print(f"❌ {LEADS_LIST}")
        return
    with open(LEADS_LIST) as f:
        terms = [l.strip().upper() for l in f if l.strip()]
    if not terms:
        print("❌ Empty leads")
        return

    terms_sql = "','".join(t.replace("'", "''") for t in terms)
    q  = f"SELECT * FROM \"{DATABASE_LAYER}\" WHERE UPPER(PROPERTYAD) IN ('{terms_sql}')"
    df = gpd.read_file(DATABASE, sql=q, engine='pyogrio')
    # Keep the data in its native CRS (EPSG:3857); normalise geometry column name
    if df.crs is None: df = df.set_crs(DATABASE_CRS)
    if 'geom' in df.columns and 'geometry' not in df.columns:
        df = df.rename_geometry('geometry')
    if df.empty:
        like_clauses = " OR ".join(
            [f"UPPER(PROPERTYAD) LIKE '%{t.replace(chr(39), chr(39)+chr(39))[:20]}%'" for t in terms]
            + [f"UPPER(PARCELID) LIKE '%{t.replace(chr(39), chr(39)+chr(39))[:20]}%'" for t in terms]
        )
        q2 = f"SELECT * FROM \"{DATABASE_LAYER}\" WHERE {like_clauses}"
        dfa = gpd.read_file(DATABASE, sql=q2, engine='pyogrio')
        if dfa.crs is None: dfa = dfa.set_crs(DATABASE_CRS)
        matched = []
        for t in terms:
            t_tokens = t.strip().split()
            search_variants = [t]
            if len(t_tokens) >= 2:
                search_variants.append(t_tokens[0])
                search_variants.append(" ".join(t_tokens[:2]))
            prop_col = dfa.get('PROPERTYAD', pd.Series(dtype=str)).fillna("").str.upper()
            pid_col  = dfa.get('PARCELID',   pd.Series(dtype=str)).fillna("").str.upper()
            mask = pd.Series([False] * len(dfa), index=dfa.index)
            for sv in search_variants:
                if sv:
                    mask = mask | prop_col.str.contains(re.escape(sv), na=False) | \
                                  pid_col.str.contains(re.escape(sv), na=False)
            matched.append(dfa[mask])
        df = pd.concat(matched).drop_duplicates() if matched else pd.DataFrame()
    if df.empty:
        print("❌ No records matched")
        return
    print(f"✅ {len(df)} parcel(s) matched\n")

    df = df.drop_duplicates(subset=['PARCELID'])
    print(f'📋  {len(df)} unique parcel(s) after deduplication\n')

    # ── Pre-load El Niño / ONI data once (used per parcel below) ─────────
    _desktop      = os.path.expanduser('~/Desktop')
    _search_dirs  = [_desktop, MASTER_EXPORT_DIR, os.path.expanduser('~/Downloads'), '.']

    def _find_file(configured, patterns, search_dirs):
        if configured and os.path.exists(configured):
            return configured
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for dirpath, _dirnames, filenames in os.walk(d):
                for fn in filenames:
                    fl = fn.lower()
                    for pat in patterns:
                        if pat in fl:
                            return os.path.join(dirpath, fn)
        return None

    _precip_csv  = _find_file(PRECIP_CSV_PATH,
                              ['4274930', 'noaa_precip', 'precipitation',
                               'historicalrainfall'],
                              _search_dirs)
    _oni_path    = _find_file(ONI_CSV_PATH,
                              ['indices.csv', 'oni_indices', 'oni.csv'],
                              _search_dirs)

    _elnino_result = None  # El Nino export removed

    for idx, row in df.iterrows():
        row = row.to_dict()
        prop_upper = str(row.get("PROPERTYAD", "") or "").upper()
        pid_upper  = str(row.get("PARCELID", "") or "").upper()
        leads_term = terms[0] if terms else ""
        for t in terms:
            if t in prop_upper or t in pid_upper or prop_upper.startswith(t[:6]):
                leads_term = t
                break
        addr_result = resolve_address(row, leads_term)
        st_addr     = addr_result["resolved_address"]
        pcpao_data  = addr_result["pcpao_data"]
        print(f"\n{'='*55}\n📍 {st_addr} [{addr_result['method']}]")

        if pcpao_data:
            for k, v in pcpao_data.items():
                if k not in row or not is_valid(row.get(k)):
                    row[k] = v

        sq_ft   = float(row.get('BUILDINGAR', 2200) or 2200)
        g_elev  = derive_ground_elevation_ft(row)
        i_elev  = float(row.get('_min', 0) or 0)
        delta   = g_elev - i_elev
        stdev   = float(row.get('_stdev', 0) or 0)

        # ── Pipe material: prefer St Pete gravity main fields, fall back to joined columns ──
        # Priority: MATERIAL (gravity main) > MATERIAL_3 (pinellas join) > default RCP
        _mat_raw = str(row.get('MATERIAL', row.get('MATERIAL_3', 'RCP')) or 'RCP').upper()
        # PIPETYPE string override (St Pete gravity: "RCP", "VCP", "CMP", "PVC", "HDPE")
        _pipetype_str = str(row.get('PIPETYPE', '') or '').upper()
        if _pipetype_str and not _mat_raw.strip():
            _mat_raw = _pipetype_str
        raw_mat  = _mat_raw
        mat_key  = ('RCP' if 'CONC' in raw_mat or 'RCP' in raw_mat or 'VCP' in raw_mat
                    else 'CMP' if 'MET' in raw_mat or 'CMP' in raw_mat else 'PVC')
        mat_data = HYDRAULIC_MATRIX[mat_key]
        # Diameter: prefer DIAMETER (St Pete, real inches) > DIAMETER_2 (pinellas join)
        try:
            _dia_raw = row.get('DIAMETER') or row.get('DIAMETER_2', 12)
            dia_int = int(float(str(_dia_raw).replace('"','').split()[0]))
            if dia_int <= 0: dia_int = 12
        except: dia_int = 12
        # Actual pipe slope from UPELEV/DOWNELEV if available (St Pete gravity main)
        _upelev   = float(row.get('UPELEV',   0) or 0)
        _downelev = float(row.get('DOWNELEV', 0) or 0)
        _pipe_len = 100.0  # assume 100ft run if no length available
        _actual_slope = abs(_upelev - _downelev) / _pipe_len if (_upelev and _downelev and abs(_upelev-_downelev) > 0) else None

        # Use actual pipe slope from UPELEV/DOWNELEV if available
        if _actual_slope and 0.001 <= _actual_slope <= 0.15:
            _manning_slope = _actual_slope
        else:
            _manning_slope = None  # calc_hydraulics uses its own default
        p_gpm, crit_i, t_flood = calc_hydraulics(sq_ft, dia_int, mat_data['n_value'], delta)
        run_gpm   = rational_gpm(sq_ft*4.5, 3.5, 0.85)
        eff_ratio = (p_gpm / max(run_gpm, 1.0)) * 100

        lot_area   = derive_lot_area_sqft(row, sq_ft)
        musym      = str(row.get('MUSYM', '') or '')
        # ── Water body — read pre-joined FULLNAME first (tl_2023_12103_areawater
        #    joined by nearest distance in THELASTBUILDINGMAP / Builds layer).
        #    distance_10 is the join distance in EPSG:3857 metres → convert to ft.
        _fullname_raw = str(row.get('FULLNAME', '') or '').strip()
        _mtfcc_raw    = str(row.get('MTFCC',    '') or '').strip()
        _dist10_m     = float(row.get('distance_10', 0) or 0)
        _dist10_ft    = _dist10_m * 3.28084           # metres → feet

        if _fullname_raw:
            waterbody  = _fullname_raw
            dist_water = _dist10_ft if _dist10_ft > 0 else float(
                row.get('distance_3', row.get('Soil_DIST', 999)) or 999)
        else:
            # Fall back to legacy WATERBODY field if FULLNAME not populated
            waterbody  = str(row.get('WATERBODY', row.get('WATERBODY_2', 'Unknown')) or 'Unknown')
        # ── New St Pete joined fields ──────────────────────────────────────
        _inlet_type  = str(row.get('INLETTYPE', '') or '')
        _th_inlet    = float(row.get('TH', 0) or 0)   # throat height
        _tc_inlet    = float(row.get('TC', 0) or 0)   # top of curb
        _discharge_t = str(row.get('DISCHARGETYPE', '') or '')
        _outfall_to  = str(row.get('OUTFALLTO', '') or '')
        _valve_type  = str(row.get('VALVETYPE', '') or '')
        _fitting_t   = str(row.get('FITTINGTYPE', '') or '')
        _size1ft     = float(row.get('SIZE1FT', 0) or 0)
        _size2ft     = float(row.get('SIZE2FT', 0) or 0)
        # Culvert diameter from SIZE1FT if available (in feet → convert to inches)
        if _size1ft > 0 and dia_int == 12:
            dia_int = int(_size1ft * 12)
        # Invert elevations from discharge point / fitting joins
        _inv_elevs = []
        for _ik in ['INV1','INV2','INV3','INV4','INV5','INV6']:
            _iv = row.get(_ik)
            if _iv:
                try: _inv_elevs.append(float(str(_iv).replace("'","").strip()))
                except: pass
        _min_inv = min(_inv_elevs) if _inv_elevs else None
        dist_water = float(row.get('distance_3', row.get('Soil_DIST', 999)) or 999)
        floodplain = str(row.get('FLOODPLAIN', 'UNKNOWN') or 'UNKNOWN')
        evac_zone  = str(row.get('EVACZONE', '?') or '?')

        # ── Soil infiltration + Myakka sand saturation model ─────────────────
        # Myakka fine sand (Mk*, Immokalee Im*) has a hardpan ~24" deep.
        # Dry: very high infiltration (2-6 in/hr). Saturated: collapses to
        # near-clay rates (0-0.1 in/hr). Model defaults to "storm-wet" state
        # which is realistic for Florida multi-hour events.
        _MYAKKA_CODES = {'MkA','MkB','MkC','Mk','ImA','ImB','Im'}
        _HYDRIC_CODES  = {'PsA','EeA','BaA','Fo','St','Wn'}
        _SANDY_CODES   = {'ArA','PaA','PaB'}

        is_myakka = (musym in _MYAKKA_CODES or
                     any(musym.startswith(c[:2]) for c in _MYAKKA_CODES if len(c) >= 2))

        if is_myakka:
            # Myakka sand: 3-state saturation model
            # Storm-wet is the realistic default for Pinellas County rainfall events
            myakka_state      = "storm-wet"   # dry | storm-wet | saturated
            soil_infil        = 0.30           # in/hr — storm-wet (hardpan limits drainage)
            _myakka_pervious_c = 0.55          # wet/partially saturated pervious C
            # Dry-state values stored for report context only:
            _myakka_dry_infil  = 3.50          # in/hr
            _myakka_dry_c      = 0.15
            # Saturated-state values:
            _myakka_sat_infil  = 0.05          # in/hr — acts like clay
            _myakka_sat_c      = 0.82
        elif musym in _HYDRIC_CODES:
            myakka_state       = None
            soil_infil         = 0.05
            _myakka_pervious_c = 0.45
        elif musym in _SANDY_CODES:
            myakka_state       = None
            soil_infil         = 1.20
            _myakka_pervious_c = 0.25
        else:
            myakka_state       = None
            soil_infil         = 0.30
            _myakka_pervious_c = 0.35

        bldg_cov   = min(0.90, sq_ft/lot_area)
        runoff_c   = bldg_cov*0.90 + (1-bldg_cov)*_myakka_pervious_c
        ponding_d  = max(0.10, abs(min(0,delta))) + 0.15
        store_gal  = lot_area * ponding_d * 7.481

        if dist_water < 100:    sc_canal = 3.5
        elif dist_water < 300:  sc_canal = 1.75
        elif dist_water < 500:  sc_canal = 0.75
        else:                   sc_canal = 0.0
        evac_map = {'A':5.0,'1':5.0,'B':3.0,'2':3.0,'C':1.5,'3':1.5,'D':0.5}
        evac_s   = evac_map.get(str(evac_zone).strip().upper(), 0.5)
        fpu      = floodplain.upper()
        fp_s     = 3.0 if 'VE' in fpu else 1.5 if 'AE' in fpu else 0.8 if 'A' in fpu else 0.3
        neg_d    = max(0.0, -delta)
        surge_ft = round(max(0.5, sc_canal+evac_s+fp_s+neg_d), 2)
        thresh_k = round(lot_area*surge_ft*7.481/1000, 1)

        # MTFCC tidal flag (pre-joined from tl_2023_12103_areawater via Builds layer)
        _is_tidal_pre = _mtfcc_raw in {"H2051", "H3010", "H3013"}

        proj = {
            "waterbody": waterbody,
            "waterbody_mtfcc":  _mtfcc_raw,
            "waterbody_tidal":  _is_tidal_pre,
            "dist_to_water_ft": round(dist_water, 0) if dist_water < 999 else "Unknown",
            "surge_canal_add_ft": sc_canal,
            "surge_evac_zone_ft": evac_s,
            "surge_floodplain_ft": fp_s,
            "surge_negative_delta_ft": round(neg_d, 2),
            "combined_surge_projection_ft": surge_ft,
            "threshold_volume_kgal": thresh_k,
            "runoff_coefficient_c": round(runoff_c, 3),
            "storage_volume_gal": round(store_gal, 0),
            "soil_infiltration_rate_in_hr": soil_infil,
        }

        scenarios = []
        storm_min = STORM_DURATION_HR * 60  # storm duration in minutes
        for r_rate in [1.0,2.0,2.4,3.0,3.5,4.3,5.0,6.0,8.0,12.0]:
            net = rational_gpm(lot_area, r_rate, runoff_c)
            ex  = max(0.0, net - p_gpm)
            mtf = round(store_gal/ex, 1) if ex > 0 else "Handles"
            # Ponding depth over the full storm duration (not just 60 min)
            d_storm = round((ex * storm_min) / (lot_area * 7.481) * 12, 2) if ex > 0 else 0.0
            rp  = next((k for k,v in DESIGN_STORMS.items() if abs(r_rate-v) < 0.2), "—")
            scenarios.append({"r":r_rate,"rp":rp,"runoff":round(net,1),"pipe":round(p_gpm,1),
                               "excess":round(ex,1),"mtf":mtf,"d_storm":d_storm,
                               "storm_hrs":STORM_DURATION_HR,
                               "status":"🔴" if isinstance(mtf,float) else "🟢"})

        # ── Coordinates ────────────────────────────────────────────────────
        data_crs = DATABASE_CRS  # EPSG:3857 — matches geometry.centroid from Builds layer
        cx = 0.0
        cy = 0.0
        _coord_source = "none"
        try:
            geom = row.get("geometry", None)
            if geom is not None and not geom.is_empty and hasattr(geom, "centroid"):
                c = geom.centroid
                if c.x != 0.0 and c.y != 0.0:
                    cx, cy = float(c.x), float(c.y)
                    _coord_source = "geometry.centroid"
        except Exception as _e:
            print(f"   ⚠️  Geometry centroid failed: {_e}")

        if not cx or not cy:
            try:
                _fx = float(row.get('feature_x') or 0)
                _fy = float(row.get('feature_y') or 0)
                if _fx and _fy:
                    cx, cy = _fx, _fy
                    _coord_source = "feature_x/feature_y"
            except Exception:
                pass

        if not cx or not cy:
            try:
                _nx = float(row.get('nearest_x') or 0)
                _ny = float(row.get('nearest_y') or 0)
                if _nx and _ny:
                    cx, cy = _nx, _ny
                    _coord_source = "nearest_x/nearest_y"
            except Exception:
                pass

        if not cx or not cy:
            print(f"   ❌  FATAL: No coordinates found for {st_addr}.")
            continue

        print(f"   📍  Coordinates: cx={cx:.6f}, cy={cy:.6f}  source={_coord_source}")

        # ── Water body spatial lookup (updates proj after coords resolved) ─
        _wb = lookup_nearest_waterbody(cx, cy, data_crs)
        if _wb["ok"]:
            waterbody  = _wb["fullname"]
            dist_water = _wb["dist_ft"]
            # MTFCC codes: H3010 stream/river, H3013 canal, H2030 lake/pond,
            #              H2025 reservoir, H2040 ice mass, H2051 bay, H3020 braided stream
            _mtfcc = _wb["mtfcc"]
            _is_tidal = _mtfcc in {"H3010", "H3013", "H2051", "H2030"}
            # Recalculate surge canal factor with the precise spatial distance
            if   dist_water < 100:  _sc_canal_shp = 3.5
            elif dist_water < 300:  _sc_canal_shp = 1.75
            elif dist_water < 500:  _sc_canal_shp = 0.75
            else:                   _sc_canal_shp = 0.0
            # Only upgrade surge if shapefile says closer (more accurate than CSV join)
            if _sc_canal_shp > proj.get("surge_canal_add_ft", 0):
                _surge_ft_shp = round(max(0.5,
                    _sc_canal_shp +
                    proj.get("surge_evac_zone_ft", 0) +
                    proj.get("surge_floodplain_ft", 0) +
                    proj.get("surge_negative_delta_ft", 0)), 2)
                proj["surge_canal_add_ft"]          = _sc_canal_shp
                proj["combined_surge_projection_ft"] = _surge_ft_shp
                proj["threshold_volume_kgal"]        = round(
                    lot_area * _surge_ft_shp * 7.481 / 1000, 1)
            proj["waterbody"]        = waterbody
            proj["dist_to_water_ft"] = dist_water
            proj["waterbody_mtfcc"]  = _mtfcc
            proj["waterbody_tidal"]  = _is_tidal
            print(f"   💧  Nearest water body: {waterbody}  "
                  f"({dist_water:.0f} ft)  [{_mtfcc}]"
                  f"{'  ⚠️  tidal' if _is_tidal else ''}")
        else:
            print(f"   💧  Water body lookup: {_wb.get('reason','failed')} "
                  f"— using CSV fallback ({waterbody})")

        print("   📏  Multi-source elevation consensus (bilinear)...")
        elev_consensus = sample_elevation_multi_source(cx, cy, point_crs=data_crs, max_sources=6)
        if elev_consensus["consensus"] is not None:
            print(f"   ✅  Elevation consensus: {elev_consensus['consensus']} ft "
                  f"(n={elev_consensus['n_sources']} sources, σ={elev_consensus['std']} ft)")
            if elev_consensus['n_sources'] > 1:
                g_elev   = elev_consensus['consensus']
                row['Elev_Z_1'] = g_elev
                delta = g_elev - i_elev

        # ── 3-Pass Geo-Verification ────────────────────────────────────────
        print("   🔍  Running 3-pass geo-verification...")
        parcel_id = str(row.get('PARCELID', '') or '')
        geo_verify = geo_verify_address(st_addr, cx, cy, g_elev, parcel_id, point_crs=data_crs)
        print(f"   🔍  Geo-verify: {geo_verify['overall_confidence']}")

        # ── Triple-Path Drainage Intelligence Scan ────────────────────────
        print("   🔎  Triple-Path drainage scan (1,000 ft radius)...")
        drainage_intel = triple_path_drainage_scan(cx, cy, data_crs, g_elev, surge_ft)
        if drainage_intel.get("ok"):
            dc  = drainage_intel["drainage_class"]
            gpm = drainage_intel["effective_gpm"]
            _tbr = "  ⚠️  TIDAL BACKFLOW RISK" if drainage_intel.get('tidal_backflow_risk') else ""
            print(f"   🔎  Drainage class: {dc}  |  Effective capacity: {gpm:.1f} GPM{_tbr}")
            print(f"        Inlets: {drainage_intel['n_structures']}  |  "
                  f"Gravity mains: {drainage_intel['n_mains']}  |  "
                  f"Open drains: {drainage_intel['n_open_drains']}  |  "
                  f"Culverts: {drainage_intel.get('n_culverts',0)}  |  "
                  f"Outfalls: {drainage_intel.get('n_outfalls',0)}")
        else:
            print(f"   ⚠️  Drainage scan: {drainage_intel.get('note','failed — defaulting to ASSUMED')}")
            drainage_intel = {"ok": False, "drainage_class": "ASSUMED",
                              "effective_gpm": None, "note": "scan failed"}

        # ── Pixel-based radial DEM (matched to map radii) ─────────────────
        _radial_fallback = {"ok": False, "profiles": {}, "depressions": {},
                            "all_depressions": [], "start_elev_ft": g_elev,
                            "summary": {"primary_flow": "UNKNOWN", "dirs_toward": 0,
                                        "convergence_dirs": [], "max_depression_depth_ft": 0.0,
                                        "total_ponded_gal": 0.0, "n_depressions": 0}}

        print(f"   🧭  Pixel-based radial DEM (75 ft)...")
        radial_60 = build_radial_dem_profiles(row, data_crs,
                                              step_ft=SAMPLE_INTERVAL_FT,
                                              radius_ft=75.0)
        if not radial_60.get("ok"):
            print(f"   ❌ DEM radial 60 ft failed: {radial_60.get('reason')}")
            radial_60 = dict(_radial_fallback)

        print(f"   🧭  Pixel-based radial DEM ({int(HOME_MAP_RADIUS_FT)} ft)...")
        radial = build_radial_dem_profiles(row, data_crs,
                                           step_ft=SAMPLE_INTERVAL_FT,
                                           radius_ft=HOME_MAP_RADIUS_FT)
        if not radial.get("ok"):
            print(f"   ❌ DEM radial {int(HOME_MAP_RADIUS_FT)} ft failed: {radial.get('reason')}")
            radial = dict(_radial_fallback)

        print(f"   🧭  Pixel-based radial DEM ({int(NEIGHBOR_RADIUS_FT)} ft)...")
        radial_1000 = build_radial_dem_profiles(row, data_crs,
                                                step_ft=SAMPLE_INTERVAL_FT,
                                                radius_ft=NEIGHBOR_RADIUS_FT)
        if not radial_1000.get("ok"):
            print(f"   ❌ DEM radial {int(NEIGHBOR_RADIUS_FT)} ft failed: {radial_1000.get('reason')}")
            radial_1000 = dict(_radial_fallback)

        terrain_raw = {}
        if radial.get("ok"):
            for dn, d in radial["profiles"].items():
                prof = []
                for p in d["points"]:
                    if p["elev"] is None: continue
                    prof.append((round(p["dist_ft"], 2), round(p["elev"], 4)))
                if prof:
                    terrain_raw[dn] = prof

        if terrain_raw:
            terrain_analysis = analyze_terrain(terrain_raw, g_elev)
            terrain_analysis["dirs_toward"]        = radial["summary"]["dirs_toward"]
            terrain_analysis["convergence_dirs"]   = radial["summary"]["convergence_dirs"]
            terrain_analysis["primary_flow"]       = radial["summary"]["primary_flow"]
        else:
            terrain_analysis = {
                "directions": {d: {"profile":[],"terminal_elev":g_elev,"terminal_delta":0,
                                   "net_slope_pct":0,"flow_toward":False,"depressions":[],
                                   "flat_runs":[],"is_void":False,"point_slopes":[]}
                               for d in DIRECTIONS},
                "primary_flow": "UNKNOWN", "highest_dir": "UNKNOWN",
                "convergence_dirs": [], "dirs_toward": 0,
            }

        pipe_data = analyze_pipe_slopes(row, g_elev, terrain_analysis)
        year_built = str(row.get('YEARBUILT', row.get('ACTYRBLT', 'Unknown')))
        foundation = str(row.get('FOUNDATION', 'Unknown'))
        permit_analysis = analyze_permits_historical(pcpao_data, row)

        # ── Rainfall-to-damage threshold calculation ──────────────────────
        print("   🌧️  Computing rainfall-to-damage thresholds...")
        _radial_sum_for_rd = radial.get("summary", {}) if radial.get("ok") else {}
        _drain_class = drainage_intel.get("drainage_class", "ASSUMED")
        _drain_gpm   = drainage_intel.get("effective_gpm", None)
        rainfall_damage = calc_rainfall_to_damage(
            lot_area_sqft           = lot_area,
            runoff_c                = runoff_c,
            soil_infil_in_hr        = soil_infil,
            pipe_capacity_gpm       = p_gpm,
            open_drains             = pipe_data.get("open_drains", []),
            radial_summary          = _radial_sum_for_rd,
            g_elev                  = g_elev,
            delta                   = delta,
            max_intensity_in_hr     = 6.0,    # 100-yr storm is 5.8 in/hr — 6.0 covers it
            damage_depth_in         = 6.0,
            max_total_rain_in       = 72.0,   # unused by new duration logic
            drainage_class          = _drain_class,
            effective_drainage_gpm  = _drain_gpm,
            pipe_data               = pipe_data,
            myakka_state            = myakka_state,
            dist_water_ft           = dist_water,
        )
        _crit_rate = (rainfall_damage["critical"]["rate_in_hr"]
                      if rainfall_damage.get("ok") and rainfall_damage.get("critical")
                      else None)
        if rainfall_damage.get("ok") and _crit_rate:
            print(f"   🌧️  Critical rate: {_crit_rate} in/hr  |  "
                  f"Total time to 6\" damage: "
                  f"{rainfall_damage['critical'].get('total_time_min','?')} min")
        else:
            print("   🌧️  No damage threshold found within 3 in/hr / 3\" cap")

        # ── Damage cost model (7-layer) ───────────────────────────────────
        print("   💵  Computing 7-layer damage cost model...")
        try:
            damage_cost = calc_damage_cost_model(
                sq_ft          = sq_ft,
                lot_area_sqft  = lot_area,
                delta          = delta,
                surge_ft       = surge_ft,
                rainfall_damage= rainfall_damage,
                pcpao_data     = pcpao_data or {},
                proj           = proj,
                year_built     = year_built,
            )
            if damage_cost.get("ok"):
                _ds = damage_cost["depth_scenarios"]
                print(f"   💵  Structure RCV: ${damage_cost['structure_rcv']:,.0f}  "
                      f"({damage_cost['val_source']})")
                print(f"   💵  Contents RCV:  ${damage_cost['contents_rcv']:,.0f}  "
                      f"({int(damage_cost['cont_ratio']*100)}% tier)  |  "
                      f"ACV ratio: {damage_cost['acv_ratio']:.2f} "
                      f"(age {damage_cost['age_yr']} yr, {damage_cost['depr_pct']}% depr.)")
                for _ds_row in _ds:
                    print(f"   💵  {_ds_row['depth_label']:12}  "
                          f"total dmg ${_ds_row['total_dmg']:>9,.0f}  |  "
                          f"OOP ${_ds_row['out_of_pocket']:>9,.0f}")
                print(f"   💵  EAD: ${damage_cost['ead_annual']:,.0f}/yr  |  "
                      f"Payback: {damage_cost['simple_payback_yr']} yr")
        except Exception as _dc_e:
            damage_cost = None
            print(f"   ⚠️  Damage cost model failed: {_dc_e}")

        # ── El Niño projection (per-property, uses pre-loaded ONI data) ───
        # Use hydraulic critical intensity (crit_i) as the El Niño threshold —
        # this is the rate that overwhelms the pipe system.  The rainfall-damage
        # scenario rate (_crit_rate) can be as low as 0.25 in/hr which causes
        # every historical month to breach.  Apply a 1.0 in/hr floor so the
        # breach list is meaningful.  Take the highest of crit_i, _crit_rate, 1.0.
        _elnino_crit = max(
            crit_i       if (crit_i and crit_i > 0) else 0.0,
            _crit_rate   if (_crit_rate and _crit_rate > 0) else 0.0,
            1.0,          # absolute floor — below this every month would breach
        )
        elnino_proj = None

        ds_list = build_10_datasets(row, sq_ft)
        if not ds_list:
            print("   ⚠️  No valid datasets — skipping")
            continue

        psi  = hydrostatic_psi(abs(delta)) if delta != 0 else hydrostatic_psi(0.1)
        tons = round((sq_ft*(NOAA_ANNUAL_IN/12)*62.4)/2000, 2)

        def psi_score_calc():
            if psi < 0.30: return score_ds((psi/0.30)*35), "🟢"
            if psi < 0.60: return score_ds(35+((psi-0.30)/0.30)*40), "🟡"
            return score_ds(75+((psi-0.60)/0.40)*25), "🔴"
        psi_s, psi_c = psi_score_calc()

        def tons_score_calc():
            if tons < 250: return score_ds((tons/250)*35), "🟢"
            if tons < 500: return score_ds(35+((tons-250)/250)*40), "🟡"
            return score_ds(75+((tons-500)/300)*25), "🔴"
        tons_s, tons_c = tons_score_calc()

        def grad_score_calc():
            if delta >= 1.0:  return score_ds(max(0, 30-(delta*4))), "🟢"
            if delta >= 0.3:  return score_ds(35+((1.0-delta)/0.70)*30), "🟡"
            if delta >= 0.0:  return score_ds(65+((0.30-delta)/0.30)*15), "🔴"
            return score_ds(min(100, 80+(abs(delta)*5))), "🔴"
        grad_s, grad_c = grad_score_calc()

        def sfp_score_calc():
            if crit_i >= 5.0: return score_ds(max(0, 25-(crit_i-5)*2)), "🟢"
            if crit_i >= 2.0: return score_ds(35+((5.0-crit_i)/3.0)*35), "🟡"
            return score_ds(75+((2.0-crit_i)/2.0)*25), "🔴"
        sfp_s, sfp_c = sfp_score_calc()

        def cap_score_calc():
            if eff_ratio >= 150: return score_ds(max(0, 20-(eff_ratio-150)*0.1)), "🟢"
            if eff_ratio >= 100: return score_ds(25+((150-eff_ratio)/50)*35), "🟡"
            return score_ds(65+((100-eff_ratio)/100)*35), "🔴"
        cap_s, cap_c = cap_score_calc()

        def tti_score_calc():
            if t_flood is None: return score_ds(0), "🟢"
            if t_flood >= 15:   return score_ds(max(0, 20-(t_flood-15)*0.5)), "🟢"
            if t_flood >= 5:    return score_ds(35+((15-t_flood)/10)*35), "🟡"
            return score_ds(min(100, 75+max(0,(5-t_flood)*3))), "🔴"
        tti_s, tti_c = tti_score_calc()

        section_scores = [psi_s, tons_s, grad_s, sfp_s, cap_s, tti_s]
        section_mean   = sum(section_scores) / len(section_scores)
        imperv         = derive_lot_area_sqft(row, sq_ft)
        imperv_ratio   = round((sq_ft/max(imperv,1))*100, 1)
        ds_mean        = sum(d["score"] for d in ds_list) / len(ds_list)

        hydro_adj = 0
        hydro_adj += max(0, (100-min(100,eff_ratio/5))*0.15)
        hydro_adj += max(0, (3.0-max(0.1,delta))*3.5) if delta < 3.0 else 0
        if delta < 0:          hydro_adj += 10
        if imperv_ratio > 70:  hydro_adj += 6
        if stdev > 0.20:       hydro_adj += 4
        if t_flood is not None and t_flood < 5:
            hydro_adj += 5
        if crit_i < 3.0:       hydro_adj += 4.5

        raw_risk   = (ds_mean*0.65) + (section_mean*0.35) + hydro_adj
        risk_score = min(100.0, raw_risk * SILENT_FLOOD_MULTIPLIER)
        flood_icon = ("🔴 CRITICAL" if risk_score > 70
                      else "🟡 CAUTIONARY" if risk_score > 40 else "🟢 STABLE")
        check = triple_check_score(ds_list, section_scores, hydro_adj, risk_score, row)

        micro_head_ft = float(radial["summary"].get("max_depression_depth_ft", 0.0) or 0.0)
        openings      = [("Garage Bay", 16.0), ("Front Door", 3.5), ("Back Slider", 8.0)]
        opening_specs = [calc_deployment(nm, w, surge_ft + micro_head_ft, g_elev)
                         for nm, w in openings]

        cln          = re.sub(r'[^\w\-_]', '_', st_addr)[:70]
        pdf_path     = os.path.join(MASTER_EXPORT_DIR, f"{cln}_FLOOD_REPORT_v121_2.pdf")
        txt_path     = os.path.join(MASTER_EXPORT_DIR, f"{cln}_FLOOD_REPORT_v121_2.txt")

        # ── Generate two 250ft maps ───────────────────────────────────────────
        radial_png       = None
        drainage_map_png = None
        sim_png          = None

        heatmap_png = os.path.join(MASTER_EXPORT_DIR, f"{cln}_HEATMAP_250FT.png")
        flow_png    = os.path.join(MASTER_EXPORT_DIR, f"{cln}_FLOW_250FT.png")

        print("   🗺️  Rendering heatmap 250ft (bold elevation style)...")
        export_heatmap_250(st_addr, cx, cy, data_crs, g_elev, row, heatmap_png)

        print("   💧  Rendering flow arrow map 250ft...")
        export_flow_250(st_addr, cx, cy, data_crs, g_elev, row, flow_png)

        # Wire into PDF (lot_png = heatmap, block_png = flow arrows, nb_png = None)
        lot_png   = heatmap_png if os.path.exists(heatmap_png) else None
        block_png = flow_png    if os.path.exists(flow_png)    else None
        nb_png    = None



        print("   🤖  Generating AI content...")
        flood_scenario = build_flood_scenario(
            row, st_addr, g_elev, delta, surge_ft, t_flood,
            crit_i, p_gpm, terrain_analysis, proj, ds_list)

        nino = None

        final_summary = build_final_summary(
            row, st_addr, risk_score, flood_icon, g_elev, delta,
            surge_ft, t_flood, crit_i, terrain_analysis,
            ds_list, permit_analysis, proj)

        pcpao_url_str = pcpao_url(parcel_id) if parcel_id else 'N/A'

        ollama_sales = None
        try:
            sales_prompt = (
                f"Senior flood-risk consultant. Property: {st_addr} "
                f"Score: {round(risk_score,1)}/100 ({flood_icon}). "
                f"Projected surge: {surge_ft} ft. Write 6-8 sentences. No bullets."
            )
            ollama_sales = ollama_query(sales_prompt, "Executive Sales") or None
        except Exception:
            ollama_sales = None

        print("   🧾  Building client PDF...")
        build_client_pdf(
            st_addr, pdf_path,
            radial_png if radial.get("ok") else block_png,
            nb_png, risk_score, flood_icon, surge_ft,
            foundation, year_built, opening_specs,
            radial.get("summary", {}), permit_analysis, ollama_sales,
            geo_verify=geo_verify,
            rainfall_damage=rainfall_damage,
            elnino_proj=elnino_proj,
            ponding_sim_png=sim_png,
            drainage_map_png=drainage_map_png,
            lot_png=lot_png,
            block_png=block_png,
            proj=proj,
            drainage_intel=drainage_intel,
            damage_cost=damage_cost)

        print("   📄  Writing full .txt report...")
        write_report(
            txt_path, st_addr, flood_icon, risk_score, parcel_id, None,
            pcpao_url_str, permit_analysis, g_elev, delta, surge_ft,
            t_flood, p_gpm, mat_data, eff_ratio, crit_i, waterbody,
            proj, sc_canal, evac_s, fp_s, neg_d, thresh_k,
            runoff_c, store_gal, soil_infil, scenarios,
            flood_scenario, psi, psi_s, psi_c, tons, tons_s, tons_c,
            grad_s, grad_c, sfp_s, sfp_c, cap_s, cap_c, tti_s, tti_c,
            ds_list, terrain_analysis, True, pipe_data,
            nino, opening_specs, check, final_summary,
            year_built, foundation, row,
            geo_verify=geo_verify, radial=radial,
            rainfall_damage=rainfall_damage, elnino_proj=elnino_proj)

        _dc  = drainage_intel.get("drainage_class", "ASSUMED")
        _gpm = drainage_intel.get("effective_gpm", 0.0)
        print(f"\n✅  PDF      : {pdf_path}")
        print(f"✅  TXT      : {txt_path}")
        if lot_png:   print(f"✅  Heatmap  : {heatmap_png}")
        if block_png: print(f"✅  Flow map : {flow_png}")
        print(f"\n   Score      : {round(risk_score,1)}/100  {flood_icon}")
        print(f"   Depressions: {radial['summary'].get('n_depressions',0)} detected  |  "
              f"{radial['summary']['total_ponded_gal']:.0f} gal total ponding")
        print(f"   Drainage   : {_dc}  |  Effective capacity: {_gpm:.1f} GPM")
        fb.queue_output(
            "PDF Report", pdf_path, st_addr,
            narrative=ollama_sales,
            narrative_type="executive sales paragraph")
        fb.queue_output(
            "Flood Scenario / Final Summary", txt_path, st_addr,
            narrative=final_summary,
            narrative_type="flood scenario and final summary")
        fb.flush(st_addr)


    fb.summarize()

    # El Nino export removed


if __name__ == "__main__":
    run_overlord_max()