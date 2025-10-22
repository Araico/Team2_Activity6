# app.py
# ------------------------------------------------------------------
# Streamlit + Folium + Nominatim (POIs dentro del viewbox de la CDMX)
# - POIs: bancos, cajeros, coworking, museos, hoteles, atracciones
# - Búsqueda por nombre (Search) sobre todos los POIs agregados
# - Render robusto (folium_static) + fallback HTML
# ------------------------------------------------------------------
import os, re, unicodedata, time, requests, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import folium_static

import folium
from folium.features import GeoJson, GeoJsonTooltip
from folium.plugins import (
    Search,
    HeatMapWithTime,
    Fullscreen,
    MiniMap,
    MousePosition,
    MeasureControl,
)
from branca.colormap import linear

# =============== Config ===============
st.set_page_config(page_title="Security Challenge — POIs Nominatim", layout="wide")
BASE_DIR = Path(__file__).parent

# =============== Sidebar ===============
st.sidebar.title("⚙️ Parámetros")
COLONIAS_PATH = st.sidebar.text_input(
    "Colonias (SHP/GeoJSON/ZIP)", str(BASE_DIR / "colonias.shp")
)
METRO_CSV = st.sidebar.text_input("Metro CSV", str(BASE_DIR / "metro_data.csv"))
CRIMES_CSV = st.sidebar.text_input(
    "Crímenes CSV", str(BASE_DIR / "FGJ_CLEAN_Team2.csv")
)

PLACE = st.sidebar.text_input("Área Nominatim (PLACE)", "Ciudad de México")
st.sidebar.caption("POIs personalizados (se usan *dentro* del viewbox de Nominatim)")
CUSTOM_POI_TEXT = st.sidebar.text_input(
    "POIs personalizados (coma-separados)", "alamedas,bellas artes,zocalo,bancos"
)
CUSTOM_POIS = [q.strip() for q in CUSTOM_POI_TEXT.split(",") if q.strip()]

st.sidebar.markdown("---")
st.sidebar.caption("Sube archivos para sobreescribir rutas (opcional)")
up_colonias = st.sidebar.file_uploader(
    "Colonias (SHP/GeoJSON/ZIP)", type=["shp", "geojson", "zip"]
)
up_metro = st.sidebar.file_uploader("Metro CSV", type=["csv"])
up_crimes = st.sidebar.file_uploader("Crímenes CSV", type=["csv"])


# =============== Helpers ===============
def pick_col_ci(cols, candidates):
    m = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in m:
            return m[k.lower()]
    return None


def clean_parens(x):
    if not isinstance(x, str):
        return x
    return re.sub(r"\s*\([^)]*\)", "", x).strip()


def key_norm_str(x):
    if not isinstance(x, str):
        return ""
    s = unicodedata.normalize("NFD", x)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = s.upper().strip()
    return re.sub(r"\s+", " ", s)


def key_norm(sr):
    return sr.astype(str).map(key_norm_str)


def valid_bounds(b):
    return (
        b is not None
        and len(b) == 4
        and all(np.isfinite(v) for v in b)
        and (b[2] > b[0])
        and (b[3] > b[1])
    )


# ----- Nominatim utils -----
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "security-challenge-nominatim/1.0 (a01662243@tec.mx)"}


def nom_geocode(address: str, limit: int = 5, lang: str = "es"):
    params = {
        "q": address,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "accept-language": lang,
        "polygon_geojson": 1,
    }
    r = requests.get(
        f"{NOMINATIM_BASE}/search", params=params, headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    time.sleep(1)
    return r.json()


def nom_reverse(lat: float, lon: float, zoom: int = 18, lang: str = "es"):
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": zoom,
        "addressdetails": 1,
        "accept-language": lang,
    }
    r = requests.get(
        f"{NOMINATIM_BASE}/reverse", params=params, headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    time.sleep(1)
    return r.json()


def bbox_from_result(item):
    # Nominatim boundingbox -> [south, north, west, east]
    if "boundingbox" not in item:
        return None
    south, north, west, east = map(float, item["boundingbox"])
    return south, north, west, east


def search_pois_in_viewbox(
    query: str, viewbox, bounded: bool = True, limit: int = 80, lang: str = "es"
):
    # viewbox en Nominatim (west, north, east, south)
    south, north, west, east = viewbox
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "accept-language": lang,
        "viewbox": f"{west},{north},{east},{south}",
        "bounded": 1 if bounded else 0,
    }
    r = requests.get(
        f"{NOMINATIM_BASE}/search", params=params, headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    time.sleep(1)
    return r.json()


def enrich_point(item):
    addr = item.get("address", {})
    return {
        "name": item.get("display_name", ""),
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "type": item.get("type", ""),
        "class": item.get("class", ""),
        "road": addr.get("road"),
        "neighbourhood": addr.get("neighbourhood") or addr.get("suburb"),
        "city": addr.get("city") or addr.get("town") or addr.get("village"),
        "state": addr.get("state"),
        "postcode": addr.get("postcode"),
        "country": addr.get("country"),
    }


def popup_html(props):
    fields = [
        ("Nombre", props.get("name")),
        ("Clase", props.get("class")),
        ("Tipo", props.get("type")),
        ("Calle", props.get("road")),
        ("Colonia/Vecindario", props.get("neighbourhood")),
        ("Ciudad", props.get("city")),
        ("Estado", props.get("state")),
        ("CP", props.get("postcode")),
        ("País", props.get("country")),
    ]
    rows = "".join(
        f"<tr><th style='text-align:left;padding-right:8px'>{k}</th><td>{v or '-'}</td></tr>"
        for k, v in fields
    )
    return f"<table>{rows}</table>"


def add_points_layer(m, points, name, color="#3388ff", show=True):
    if not points:
        return None
    fg = folium.FeatureGroup(name=name, show=show)
    for p in points:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=p.get("name", name),
            popup=folium.Popup(popup_html(p), max_width=360),
        ).add_to(fg)
    fg.add_to(m)
    return fg


# =============== Carga de datos base (colonias/metro/crímenes) ===============
with st.spinner("Cargando shapefile/CSV…"):
    # Colonias
    colonias_raw = gpd.read_file(up_colonias or COLONIAS_PATH).to_crs(4326)
    col_name_src = pick_col_ci(
        colonias_raw.columns,
        ["nomut", "nomdt", "nombre", "nom_col", "nomgeo", "name", "colonia"],
    )
    if col_name_src is None:
        st.stop()
    colonias_raw["Colony"] = colonias_raw[col_name_src].map(clean_parens)
    colonias = colonias_raw.dissolve(by="Colony", as_index=False)
    colonias["key_norm"] = key_norm(colonias["Colony"])
    col_m = colonias.to_crs(3857)
    colonias["Area_km²"] = (col_m.geometry.area / 1e6).values

    # Metro (opcional pero útil)
    mdf = pd.read_csv(up_metro or METRO_CSV)
    lon_m = pick_col_ci(mdf.columns, ["longitud", "lon", "longitude", "lng", "x"])
    lat_m = pick_col_ci(mdf.columns, ["latitud", "lat", "latitude", "y"])
    nam_m = pick_col_ci(
        mdf.columns, ["nombre", "name", "station", "estacion", "estación"]
    )
    xfer_m = pick_col_ci(
        mdf.columns, ["es_transbordo", "transbordo", "transfer", "is_transfer"]
    )
    if nam_m is None:
        mdf["station_name"] = np.arange(len(mdf)).astype(str)
    else:
        mdf = mdf.rename(columns={nam_m: "station_name"})
    if xfer_m is None:
        mdf["es_transbordo"] = 0
    else:
        mdf = mdf.rename(columns={xfer_m: "es_transbordo"})

    metro_all = gpd.GeoDataFrame(
        mdf[["station_name", "es_transbordo", lon_m, lat_m]].copy(),
        geometry=gpd.points_from_xy(mdf[lon_m], mdf[lat_m]),
        crs="EPSG:4326",
    )

    # Crímenes (para choropleth y heatmap)
    hdr = pd.read_csv(up_crimes or CRIMES_CSV, nrows=0)
    c_grp = pick_col_ci(
        hdr.columns, ["delito_grupo", "crime_group", "offense_group", "category"]
    )
    c_lon = pick_col_ci(hdr.columns, ["longitud", "longitude", "lon", "lng", "x"])
    c_lat = pick_col_ci(hdr.columns, ["latitud", "latitude", "lat", "y"])
    c_dat = pick_col_ci(
        hdr.columns,
        [
            "fecha",
            "fecha_hechos",
            "f_hecho",
            "date",
            "datetime",
            "created_at",
            "fechahora",
            "hora_hecho",
        ],
    )
    usecols = [c for c in [c_grp, c_lon, c_lat, c_dat] if c]
    cr_df = pd.read_csv(up_crimes or CRIMES_CSV, usecols=usecols, low_memory=False)

    raw_norm = key_norm(cr_df[c_grp]) if c_grp else pd.Series([], dtype=str)
    map_dict = {
        "ROBO TRANSEUNTE": "ROBBERY_PEDESTRIAN",
        "ROBO A TRANSEUNTE": "ROBBERY_PEDESTRIAN",
        "ROBBERY_PEDESTRIAN": "ROBBERY_PEDESTRIAN",
        "ROBO OBJETOS": "ROBBERY_OBJECT",
        "ROBO DE OBJETOS": "ROBBERY_OBJECT",
        "ROBBERY_OBJECT": "ROBBERY_OBJECT",
    }
    if c_grp:
        cr_df["_crime_norm"] = raw_norm.map(map_dict).fillna(raw_norm)
        cr_df = cr_df[
            cr_df["_crime_norm"].isin({"ROBBERY_PEDESTRIAN", "ROBBERY_OBJECT"})
        ].copy()

    cr_df["_lon"] = pd.to_numeric(cr_df[c_lon], errors="coerce")
    cr_df["_lat"] = pd.to_numeric(cr_df[c_lat], errors="coerce")
    cr_df = cr_df.dropna(subset=["_lon", "_lat"])

    if c_dat:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            cr_df["_ts"] = pd.to_datetime(
                cr_df[c_dat], errors="coerce", utc=True
            ).dt.tz_convert("America/Mexico_City")
    else:
        cr_df["_ts"] = pd.NaT

    gdf_crimes = gpd.GeoDataFrame(
        cr_df,
        geometry=gpd.points_from_xy(cr_df["_lon"], cr_df["_lat"]),
        crs="EPSG:4326",
    )

# =============== Choropleth (inc./km²) ===============
joined = gpd.sjoin(
    gdf_crimes, colonias[["key_norm", "geometry"]], how="inner", predicate="within"
)
counts = joined.groupby("key_norm").size().rename("Incidents").reset_index()
choropleth = (
    colonias[["Colony", "key_norm", "geometry", "Area_km²"]]
    .merge(counts, on="key_norm", how="left")
    .fillna({"Incidents": 0})
)
choropleth["Incidents_per_km²"] = (
    choropleth["Incidents"] / choropleth["Area_km²"].replace({0: np.nan})
).fillna(0.0)

# =============== Mapa base ===============
tb = choropleth.total_bounds if len(choropleth) else None
if not valid_bounds(tb):
    # CDMX por defecto
    minx, miny, maxx, maxy = (-99.35, 19.2, -98.9, 19.6)
else:
    minx, miny, maxx, maxy = tb

center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="cartodbpositron",
    control_scale=True,
)
Fullscreen().add_to(m)
MiniMap(toggle_display=True).add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
MousePosition(position="bottomright").add_to(m)

# Choropleth
vmin = float(choropleth["Incidents_per_km²"].min())
vmax = float(choropleth["Incidents_per_km²"].max())
cmap = linear.Purples_09.scale(vmin, vmax)
cmap.caption = "Incidentes por km² (robos seleccionados)"
_ch = choropleth.copy()
_ch["val"] = _ch["Incidents_per_km²"].astype(float)

clayer = folium.FeatureGroup(name="Densidad de robos (Inc./km²)", show=True)
GeoJson(
    data=_ch.to_json(),
    style_function=lambda f: {
        "fillColor": (
            cmap(f["properties"]["val"])
            if f["properties"]["val"] is not None
            else "#f3f0ff"
        ),
        "color": "#4b5563",
        "weight": 0.3,
        "fillOpacity": 0.85,
    },
    tooltip=GeoJsonTooltip(
        fields=["Colony", "Incidents", "Area_km²", "Incidents_per_km²"],
        aliases=["Colonia", "Incidentes (2 tipos)", "Área (km²)", "Incidentes/km²"],
        localize=True,
        sticky=True,
    ),
).add_to(clayer)
clayer.add_to(m)
cmap.add_to(m)

# Límites colonias
borders = folium.FeatureGroup(name="Colonias (límites)", show=False)
GeoJson(
    data=colonias[["Colony", "geometry"]].to_json(),
    style_function=lambda f: {"fillOpacity": 0.0, "color": "#7e22ce", "weight": 1.2},
    tooltip=GeoJsonTooltip(fields=["Colony"], aliases=["Colonia"]),
).add_to(borders)
borders.add_to(m)

# Metro (si hay puntos válidos)
metro_layer, added_any = (
    folium.FeatureGroup(name="Estaciones de Metro", show=False),
    False,
)
for _, r in metro_all.iterrows():
    if r.geometry is None or pd.isna(r.geometry.x) or pd.isna(r.geometry.y):
        continue
    folium.CircleMarker(
        location=[r.geometry.y, r.geometry.x],
        radius=4,
        color="#44403c",
        fill=True,
        fill_opacity=1.0,
        tooltip=f"{r.get('station_name','(station)')} | Transferencia: {int(r.get('es_transbordo',0))}",
    ).add_to(metro_layer)
    added_any = True
if added_any:
    metro_layer.add_to(m)

# Heatmap 24h (si hay timestamps)
if cr_df["_ts"].notna().any():
    tmax = pd.to_datetime(cr_df.loc[cr_df["_ts"].notna(), "_ts"]).max()
    tmin = tmax - pd.Timedelta(hours=24)
    recent = gdf_crimes[
        (gdf_crimes["_ts"] >= tmin) & (gdf_crimes["_ts"] <= tmax)
    ].copy()
    if len(recent):
        recent["hour"] = recent["_ts"].dt.floor("h")
        hours = sorted(recent["hour"].unique())
        heat_seq = [
            recent[recent["hour"] == h][["_lat", "_lon"]].values.tolist() for h in hours
        ]
        HeatMapWithTime(
            heat_seq,
            index=[str(h) for h in hours],
            auto_play=False,
            max_opacity=0.8,
            radius=9,
            name="Heatmap robos",
        ).add_to(m)

# =============== Nominatim: área y POIs (DENTRO del viewbox) ===============
st.markdown("#### 🧭 Mapa")

# 1) Buscar PLACE y dibujar polígono
try:
    candidates = nom_geocode(PLACE, limit=5, lang="es")
except Exception as e:
    candidates = []
    st.warning(f"Nominatim no disponible: {e}")

viewbox = None
if candidates:
    # Elegir MultiPolygon/Polygon preferentemente
    chosen = None
    for item in candidates:
        gj = item.get("geojson", {})
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            chosen = item
            break
    if chosen is None:
        chosen = candidates[0]

    lat0, lon0 = float(chosen["lat"]), float(chosen["lon"])
    viewbox = bbox_from_result(chosen)

    gj = chosen.get("geojson")
    if gj:
        folium.GeoJson(
            gj,
            name="Área de referencia (Nominatim)",
            style_function=lambda x: {
                "fillColor": "#FFB74D",
                "color": "#E64A19",
                "weight": 2,
                "fillOpacity": 0.20,
            },
        ).add_to(m)

    try:
        rev = nom_reverse(lat0, lon0, zoom=18, lang="es")
        folium.Marker(
            [lat0, lon0],
            tooltip="Centro (reverse geocoding)",
            popup=folium.Popup(
                f"<b>Centro:</b><br>{rev.get('display_name','')}", max_width=360
            ),
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(m)
    except Exception:
        pass

# 2) Consultas de POIs dentro del viewbox
poi_layers = []
all_points = []


def collect_points(queries, vb, color, name, show):
    pts = []
    if not vb:
        return None, pts
    for q in queries:
        try:
            res = search_pois_in_viewbox(q, viewbox=vb, limit=80, lang="es")
            pts.extend(enrich_point(r) for r in res)
        except Exception as e:
            st.write(f"Consulta '{q}' falló: {e}")
    if pts:
        layer = add_points_layer(m, pts, name=name, color=color, show=show)
        return layer, pts
    return None, pts


# Personalizados (incluye "bancos" por defecto)
layer_custom, pts_custom = collect_points(
    CUSTOM_POIS, viewbox, "#6ab04c", "POIs personalizados", True
)
poi_layers.append(layer_custom)
all_points.extend(pts_custom)

# Turismo
tourism_q = ["museum", "hotel", "tourist attraction"]
layer_tour, pts_tour = collect_points(
    tourism_q, viewbox, "#3388ff", "POIs Turismo", False
)
poi_layers.append(layer_tour)
all_points.extend(pts_tour)

# Económicos (bancos/cajero/coworking)
econ_q = ["bank", "cajero", "coworking"]
layer_econ, pts_econ = collect_points(
    econ_q, viewbox, "#9b59b6", "POIs Actividad Económica", False
)
poi_layers.append(layer_econ)
all_points.extend(pts_econ)

# 3) Capa de búsqueda por nombre (si hubo puntos)
if all_points:
    features = [
        {
            "type": "Feature",
            "properties": {"name": p.get("name", "(sin nombre)")},
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
        }
        for p in all_points
    ]
    gj_points = {"type": "FeatureCollection", "features": features}
    gj_layer = folium.GeoJson(
        gj_points,
        name="Búsqueda por nombre (POIs)",
        tooltip=folium.features.GeoJsonTooltip(fields=["name"], aliases=["Nombre"]),
    ).add_to(m)
    Search(
        layer=gj_layer,
        geom_type="Point",
        search_label="name",
        placeholder="Buscar POI por nombre…",
        collapsed=False,
    ).add_to(m)

# 4) Control de capas
folium.LayerControl(collapsed=False).add_to(m)

# 5) Render (siempre visible)
_rendered = False
try:
    folium_static(m, width=None, height=720)  # estable
    _rendered = True
except Exception as e:
    st.warning(f"No se pudo renderizar con folium_static. Fallback HTML. Detalle: {e}")

if not _rendered:
    html_map = m.get_root().render()
    components.html(html_map, height=720)

# 6) Export HTML
map_html_path = BASE_DIR / "combined_security_challenge_map.html"
m.save(str(map_html_path))
st.success(f"Mapa exportado a: {map_html_path.name}")

with open(map_html_path, "rb") as f:
    st.download_button(
        "⬇️ Descargar HTML del mapa",
        data=f,
        file_name=map_html_path.name,
        mime="text/html",
    )
