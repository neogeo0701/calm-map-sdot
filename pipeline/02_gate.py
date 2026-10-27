"""1단계 게이트: 센서 반경 100m(필요 시 150m) 이내 공원·놀이터, 소음 결측률, 기간 확인.

OSM 원본(data/raw/osm_seoul_parks_playgrounds.json)은 Overpass API에서 아래 질의로 받았다.
  [out:json][timeout:240][bbox:37.41,126.76,37.72,127.19];(nwr["leisure"="playground"];nwr["leisure"="park"];);out geom;
"""
import json
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
from pyproj import Transformer

to_m = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True).transform

# --- 센서 위치
loc = pd.read_excel("data/raw/sdot_locations.xlsx")
loc.columns = ["no", "serial", "address", "crs", "lat", "lon", "old_serial", "old_serial_in_data"]
loc = loc.dropna(subset=["lat", "lon"])

# --- 소음 결측률 (데이터 기간 전체 시간 수 대비 소음 평균이 있는 시간)
df = pd.read_pickle("data/processed/sdot_hourly_all.pkl")
t0, t1 = df["measured_at"].min(), df["measured_at"].max()
expected_hours = int((t1 - t0) / pd.Timedelta(hours=1)) + 1
obs = df.dropna(subset=["noise_avg"]).groupby("serial").size()
meta = df.groupby("serial")[["area_type", "district", "dong"]].agg("last")

def data_serial(r):
    if r["serial"] in obs.index:
        return r["serial"]
    for c in ["old_serial_in_data", "old_serial"]:
        if isinstance(r[c], str) and r[c] in obs.index:
            return r[c]
    return r["serial"]

loc["data_serial"] = loc.apply(data_serial, axis=1)
loc["noise_hours"] = loc["data_serial"].map(obs).fillna(0).astype(int)
loc["missing_rate"] = 1 - loc["noise_hours"] / expected_hours
loc = loc.join(meta, on="data_serial")

# --- OSM 공원·놀이터 도형
osm = json.load(open("data/raw/osm_seoul_parks_playgrounds.json", encoding="utf-8"))["elements"]

def geom_of(e):
    if e["type"] == "node":
        return Point(to_m(e["lon"], e["lat"]))
    if e["type"] == "way":
        pts = [to_m(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            return Polygon(pts)
        return LineString(pts)
    lines = [LineString([to_m(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
    polys = list(polygonize(unary_union(lines))) if lines else []
    return MultiPolygon(polys) if polys else (unary_union(lines) if lines else None)

places = []
for e in osm:
    g = geom_of(e)
    if g is None or g.is_empty:
        continue
    tags = e.get("tags", {})
    places.append({"osm_id": f"{e['type']}/{e['id']}", "kind": tags.get("leisure"),
                   "name": tags.get("name:ko") or tags.get("name"), "geom": g})
tree = STRtree([p["geom"] for p in places])

rows = []
for _, s in loc.iterrows():
    sp = Point(to_m(s["lon"], s["lat"]))
    for i in tree.query(sp.buffer(150)):
        p = places[i]
        d = p["geom"].distance(sp)
        if d <= 150:
            rows.append({"serial": s["serial"], "data_serial": s["data_serial"], "address": s["address"],
                         "sensor_lat": s["lat"], "sensor_lon": s["lon"], "area_type": s["area_type"],
                         "district": s["district"], "missing_rate": round(s["missing_rate"], 4),
                         "osm_id": p["osm_id"], "kind": p["kind"], "name": p["name"],
                         "dist_m": round(d, 1),
                         "place_lat": None, "place_lon": None})
pairs = pd.DataFrame(rows)
# 장소 대표 좌표(지도 표시용)
back = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True).transform
cent = {p["osm_id"]: back(*p["geom"].representative_point().coords[0]) for p in places}
pairs["place_lon"] = pairs["osm_id"].map(lambda k: cent[k][0])
pairs["place_lat"] = pairs["osm_id"].map(lambda k: cent[k][1])
pairs.to_csv("data/processed/gate_pairs_150m.csv", index=False, encoding="utf-8-sig")
loc.to_csv("data/processed/sensor_missing.csv", index=False, encoding="utf-8-sig")

summary = {"period": [str(t0), str(t1)], "expected_hours": expected_hours,
           "sensors_in_location_file": int(len(loc)), "sensors_with_noise": int((loc.noise_hours > 0).sum())}
for r in (100, 150):
    p = pairs[pairs.dist_m <= r]
    ok = p[p.missing_rate <= 0.20]
    summary[f"r{r}"] = {
        "pairs": int(len(p)), "places": int(p.osm_id.nunique()), "sensors": int(p.serial.nunique()),
        "places_named": int(p[p.name.notna()].osm_id.nunique()),
        "places_with_sensor_missing_le_20pct": int(ok.osm_id.nunique()),
        "named_places_with_sensor_missing_le_20pct": int(ok[ok.name.notna()].osm_id.nunique()),
        "sensors_missing_le_20pct": int(ok.serial.nunique()),
        "by_district_named_ok": ok[ok.name.notna()].groupby("district").osm_id.nunique().sort_values(ascending=False).head(10).to_dict(),
    }
json.dump(summary, open("data/processed/gate_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
