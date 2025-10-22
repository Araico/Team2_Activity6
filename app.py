import os, re, unicodedata, math, time, json, requests
import pandas as pd, numpy as np, geopandas as gpd
from datetime import timedelta
import folium
from folium import plugins
from folium.features import GeoJson, GeoJsonTooltip
from folium.plugins import MeasureControl, Search
from branca.colormap import linear
from shapely.geometry import shape

"""
## 3) Configure Paths

Update `BASE_DIR` to where your datasets live in Drive.
"""

# === PATHS (absolute, strings) ===
BASE_DIR     = os.path.dirname(__file__)
COLONIAS_SHP = os.path.join(BASE_DIR, 'colonias.shp')
METRO_CSV    = os.path.join(BASE_DIR, 'metro_data.csv')
CRIMES_OPT   = os.path.join(BASE_DIR, 'FGJ_CLEAN_Team2.csv')

"""
## 4) Helper Functions
"""

def pick_col_ci(cols, candidates):
    m = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in m:
            return m[k.lower()]
    return None

def clean_parens(x):
    import re
    if not isinstance(x, str):
        return x
    return re.sub(r'\s*\([^)]*\)', '', x).strip()

def key_norm_str(x):
    if not isinstance(x, str):
        return ''
    s = unicodedata.normalize('NFD', x)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = re.sub(r'[^\w\s]', ' ', s)
    s = s.upper().strip()
    return re.sub(r'\s+', ' ', s)

def key_norm(sr):
    return sr.astype(str).map(key_norm_str)

"""
## 5) Load Data (Colonias, Metro, Crimes) & Build Base Layers
"""

# Colonias
colonias_raw = gpd.read_file(COLONIAS_SHP).to_crs(4326)
col_name_src = pick_col_ci(colonias_raw.columns, ['nomut','nomdt','nombre','nom_col','nomgeo','name','colonia'])
colonias_raw['Colony'] = colonias_raw[col_name_src].map(clean_parens)
colonias = colonias_raw.dissolve(by='Colony', as_index=False)
colonias['key_norm'] = key_norm(colonias['Colony'])

col_m = colonias.to_crs(3857)
colonias['Area_km²'] = (col_m.geometry.area / 1e6).values

# Delegations (optional)
deleg_col = pick_col_ci(colonias_raw.columns, ['nomdt','alcaldia','delegacion','municipio','mun'])
deleg = None
if deleg_col is not None:
    deleg = (colonias_raw.assign(Deleg=colonias_raw[deleg_col].map(clean_parens))
             .dissolve(by='Deleg', as_index=False)[['Deleg','geometry']]
             .to_crs(4326))

# Metro
mdf = pd.read_csv(METRO_CSV)
lon_m = pick_col_ci(mdf.columns, ['longitud','lon','longitude','lng','x'])
lat_m = pick_col_ci(mdf.columns, ['latitud','lat','latitude','y'])
nam_m = pick_col_ci(mdf.columns, ['nombre','name','station','estacion','estación'])
xfer_m= pick_col_ci(mdf.columns, ['es_transbordo','transbordo','transfer','is_transfer'])

if nam_m is None: mdf['station_name'] = np.arange(len(mdf)).astype(str)
else:            mdf = mdf.rename(columns={nam_m:'station_name'})
if xfer_m is None: mdf['es_transbordo'] = 0
else:              mdf = mdf.rename(columns={xfer_m:'es_transbordo'})

mdf['station_key'] = key_norm(mdf['station_name'])
canon = (mdf.groupby('station_key')['station_name']
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
           .rename('station_canon'))
mdf = mdf.merge(canon, on='station_key', how='left')

metro_all = gpd.GeoDataFrame(
    mdf[['station_key','station_canon','es_transbordo', lon_m, lat_m]].copy(),
    geometry=gpd.points_from_xy(mdf[lon_m], mdf[lat_m]), crs='EPSG:4326')

# Crimes
candidates_date = ['fecha','fecha_hechos','f_hecho','date','datetime','created_at','fechahora','hora_hecho']
candidates_grp  = ['delito_grupo','crime_group','offense_group','category']
candidates_lon  = ['longitud','longitude','lon','lng','x']
candidates_lat  = ['latitud','latitude','lat','y']
candidates_wthr = ['conditions','weather','clima','condicion','condiciones']

hdr = pd.read_csv(CRIMES_OPT, nrows=0)
c_grp = pick_col_ci(hdr.columns, candidates_grp)
c_lon = pick_col_ci(hdr.columns, candidates_lon)
c_lat = pick_col_ci(hdr.columns, candidates_lat)
c_dat = pick_col_ci(hdr.columns, candidates_date)
c_wth = pick_col_ci(hdr.columns, candidates_wthr)

usecols = [c for c in [c_grp,c_lon,c_lat,c_dat,c_wth] if c is not None]
cr_df = pd.read_csv(CRIMES_OPT, usecols=usecols, low_memory=False)

raw_norm = key_norm(cr_df[c_grp])
map_dict = {
    'ROBO TRANSEUNTE': 'ROBBERY_PEDESTRIAN',
    'ROBO A TRANSEUNTE': 'ROBBERY_PEDESTRIAN',
    'ROBBERY_PEDESTRIAN': 'ROBBERY_PEDESTRIAN',
    'ROBO OBJETOS': 'ROBBERY_OBJECT',
    'ROBO DE OBJETOS': 'ROBBERY_OBJECT',
    'ROBBERY_OBJECT': 'ROBBERY_OBJECT',
}
cr_df['_crime_norm'] = raw_norm.map(map_dict).fillna(raw_norm)
cr_df = cr_df[cr_df['_crime_norm'].isin({'ROBBERY_PEDESTRIAN','ROBBERY_OBJECT'})].copy()

cr_df['_lon'] = pd.to_numeric(cr_df[c_lon], errors='coerce')
cr_df['_lat'] = pd.to_numeric(cr_df[c_lat], errors='coerce')
cr_df = cr_df.dropna(subset=['_lon','_lat'])

if c_dat is not None:
    cr_df['_ts'] = pd.to_datetime(cr_df[c_dat], errors='coerce', utc=True).dt.tz_convert('America/Mexico_City')
else:
    cr_df['_ts'] = pd.NaT

if c_wth is not None:
    cr_df['_weather'] = cr_df[c_wth].astype(str).str.strip()
else:
    cr_df['_weather'] = np.nan

gdf_crimes = gpd.GeoDataFrame(cr_df, geometry=gpd.points_from_xy(cr_df['_lon'], cr_df['_lat']), crs='EPSG:4326')

# Incidents per colony
joined = gpd.sjoin(gdf_crimes, colonias[['key_norm','geometry']], how='inner', predicate='within')
counts = joined.groupby('key_norm').size().rename('Incidents').reset_index()
choropleth = (colonias[['Colony','key_norm','geometry','Area_km²']]
              .merge(counts, on='key_norm', how='left')
              .fillna({'Incidents':0}))
choropleth['Incidents_per_km²'] = (choropleth['Incidents'] / choropleth['Area_km²'].replace({0:np.nan})).fillna(0.0)

# Base map
bounds = choropleth.total_bounds
minx, miny, maxx, maxy = bounds
center_lon, center_lat = (minx+maxx)/2, (miny+maxy)/2
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='cartodbpositron', control_scale=True)
m.fit_bounds([[miny,minx],[maxy,maxx]])
plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m)
plugins.MeasureControl(primary_length_unit='meters').add_to(m)
plugins.MousePosition(position='bottomright').add_to(m)

# Heatmap animation (24h window)
if cr_df['_ts'].notna().any():
    tmax = pd.to_datetime(cr_df.loc[cr_df['_ts'].notna(), '_ts']).max()
    tmin = tmax - pd.Timedelta(hours=24)
    recent = gdf_crimes[(gdf_crimes['_ts']>=tmin) & (gdf_crimes['_ts']<=tmax)].copy()
    if len(recent):
        recent['hour'] = recent['_ts'].dt.floor('H')
        hours = sorted(recent['hour'].unique())
        heat_seq = [recent[recent['hour']==h][['_lat','_lon']].values.tolist() for h in hours]
        plugins.HeatMapWithTime(
            heat_seq, index=[str(h) for h in hours], auto_play=False, max_opacity=0.8, radius=9,
            name="Heatmap robos"
        ).add_to(m)

# Colonias (límites)
colonies_layer = folium.FeatureGroup(name='Colonias (límites)', show=False)
GeoJson(
    data=colonias[['Colony','geometry']].to_json(),
    style_function=lambda f: {"fillOpacity": 0.0, "color": "#7e22ce", "weight": 1.2},
    tooltip=GeoJsonTooltip(fields=['Colony'], aliases=['Colonia'])
).add_to(colonies_layer)
colonies_layer.add_to(m)

# Estaciones de Metro
metro_layer = folium.FeatureGroup(name='Estaciones de Metro', show=False)
for _, r in metro_all.iterrows():
    if r.geometry is None or pd.isna(r.geometry.x) or pd.isna(r.geometry.y):
        continue
    folium.CircleMarker(
        location=[r.geometry.y, r.geometry.x],
        radius=4, color='#44403c', fill=True, fill_opacity=1.0,
        tooltip=f"{r.get('station_canon','(station)')} | Transferencia: {int(r.get('es_transbordo',0))}"
    ).add_to(metro_layer)
metro_layer.add_to(m)

# Densidad de robos (Incidents/km²)
vmin = float(choropleth["Incidents_per_km²"].min())
vmax = float(choropleth["Incidents_per_km²"].max())
cmap = linear.Purples_09.scale(vmin, vmax)
cmap.caption = "Incidentes por km² (robos seleccionados)"

_ch = choropleth.copy()
_ch["val"] = _ch["Incidents_per_km²"].astype(float)

density_layer = folium.FeatureGroup(name="Densidad de robos (Inc./km²)", show=True)
GeoJson(
    data=_ch.to_json(),
    style_function=lambda f: {
        "fillColor": cmap(f["properties"]["val"]) if f["properties"]["val"] is not None else "#f3f0ff",
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
    name="Densidad de robos"
).add_to(density_layer)
density_layer.add_to(m)
cmap.add_to(m)  # legend

"""
## 6) Nominatim Utilities
"""

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "security-challenge-nominatim/1.0 (a01662243@tec.mx)"}

def nom_geocode(address: str, limit: int = 5, lang: str = "es"):
    params = {
        "q": address,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "accept-language": lang,
        "polygon_geojson": 1
    }
    r = requests.get(f"{NOMINATIM_BASE}/search", params=params, headers=HEADERS, timeout=30)
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
        "accept-language": lang
    }
    r = requests.get(f"{NOMINATIM_BASE}/reverse", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(1)
    return r.json()

def bbox_from_result(item):
    if "boundingbox" not in item:
        return None
    south, north, west, east = map(float, item["boundingbox"])
    return south, north, west, east

def search_pois_in_viewbox(query: str, viewbox, bounded: bool = True, limit: int = 80, lang: str = "es"):
    south, north, west, east = viewbox
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "accept-language": lang,
        "viewbox": f"{west},{north},{east},{south}",
        "bounded": 1 if bounded else 0
    }
    r = requests.get(f"{NOMINATIM_BASE}/search", params=params, headers=HEADERS, timeout=30)
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

def add_points_layer(m, points, name, color='#3388ff', show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for p in points:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color=color, fill=True, fill_opacity=0.9,
            tooltip=p.get("name", name),
            popup=folium.Popup(popup_html(p), max_width=360)
        ).add_to(fg)
    fg.add_to(m)
    return fg

"""
## 7) Run Nominatim Queries & Build POI Layers
"""

# Choose PLACE for bbox/polygon
PLACE = "Ciudad de México"  # change if needed, e.g. "Coyoacán, CDMX"

candidates = nom_geocode(PLACE, limit=5, lang="es")
if not candidates:
    raise RuntimeError("No results for the given PLACE. Try a more specific query.")

# Prefer polygon result
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

# Boundary polygon
gj = chosen.get("geojson")
if gj:
    folium.GeoJson(
        gj,
        name="Área de referencia (Nominatim)",
        style_function=lambda x: {"fillColor": "#FFB74D", "color": "#E64A19", "weight": 2, "fillOpacity": 0.20},
        highlight_function=lambda x: {"weight": 3}
    ).add_to(m)

# Reverse geocoding marker
try:
    rev = nom_reverse(lat0, lon0, zoom=18, lang="es")
    center_name = rev.get("display_name", "Centro del área")
    folium.Marker(
        [lat0, lon0],
        tooltip="Reverse geocoding (centro)",
        popup=folium.Popup(f"<b>Centro:</b><br>{center_name}", max_width=360),
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)
except Exception as e:
    print("Reverse geocoding failed:", e)

# Custom POIs (required)
CUSTOM_POI_QUERIES = ["alamedas", "bellas artes", "zocalo", "bancos"]

custom_points = []
if viewbox:
    for q in CUSTOM_POI_QUERIES:
        try:
            res = search_pois_in_viewbox(q, viewbox=viewbox, limit=80, lang="es")
            custom_points.extend(enrich_point(r) for r in res)
        except Exception as e:
            print(f"Búsqueda personalizada '{q}' falló: {e}")

if custom_points:
    custom_fg = add_points_layer(m, custom_points, name="POIs personalizados (Alamedas/Bellas Artes/Zócalo/Bancos)", color="#6ab04c", show=True)

# Optional broader POIs
POI_TOURISM = ["museum", "hotel", "tourist attraction"]
POI_ECON    = ["bank", "cajero","coworking"]

tourism_points, econ_points = [], []
if viewbox:
    for q in POI_TOURISM:
        try:
            res = search_pois_in_viewbox(q, viewbox=viewbox, limit=80, lang="es")
            tourism_points.extend(enrich_point(r) for r in res)
        except Exception as e:
            print(f"Tourism query '{q}' failed: {e}")
    for q in POI_ECON:
        try:
            res = search_pois_in_viewbox(q, viewbox=viewbox, limit=80, lang="es")
            econ_points.extend(enrich_point(r) for r in res)
        except Exception as e:
            print(f"Economic query '{q}' failed: {e}")

if tourism_points:
    add_points_layer(m, tourism_points, name="POIs Turismo", color="#3388ff", show=False)
if econ_points:
    add_points_layer(m, econ_points,    name="POIs Actividad Económica", color="#9b59b6", show=False)

# Search Control for POIs
features = []
def to_feature(p):
    return {
        "type":"Feature",
        "properties":{"name": p.get("name","(sin nombre)")},
        "geometry":{"type":"Point","coordinates":[p["lon"], p["lat"]]}
    }

for coll in [custom_points, tourism_points, econ_points]:
    for p in coll:
        features.append(to_feature(p))

if features:
    gj_points = {"type":"FeatureCollection","features":features}
    gj_layer = folium.GeoJson(
        gj_points,
        name="Búsqueda por nombre (POIs)",
        tooltip=folium.features.GeoJsonTooltip(fields=["name"], aliases=["Nombre"])
    ).add_to(m)
    Search(
        layer=gj_layer,
        geom_type='Point',
        search_label='name',
        placeholder='Buscar POI por nombre…',
        collapsed=False
    ).add_to(m)

"""
## 8) Layer Control, Save & Download Map
"""

folium.LayerControl(collapsed=False).add_to(m)
map_html_path = os.path.join(BASE_DIR, 'combined_security_challenge_map.html')
m.save(map_html_path)
print(f"Mapa guardado en: {map_html_path}")
