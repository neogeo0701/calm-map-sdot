"""게이트 통과 후 송파구 선정 장소 5곳을 확정한다(센서–장소 경계 거리 30m 이내, 서로 다른 센서)."""
import json
import pandas as pd

SITES = [  # (OSM 이름, 센서 시리얼) — 근거: data/processed/gate_pairs_150m.csv
    ("송파근린공원", "V02Q1940775"),
    ("삼전공원", "V02Q1940767"),
    ("큰솔 어린이공원", "V02Q1940736"),
    ("거여공원", "V02Q1940757"),
    ("현림어린이공원", "V02Q1940717"),
]

pairs = pd.read_csv("data/processed/gate_pairs_150m.csv")
seoul = pd.read_excel("data/raw/seoul_parks.xlsx", header=None).iloc[2:]
seoul_names = set(seoul[3].astype(str).str.replace(" ", ""))

out = []
for i, (name, serial) in enumerate(SITES, 1):
    r = pairs[(pairs.name == name) & (pairs.serial == serial)].sort_values("dist_m").iloc[0]
    out.append({
        "site_id": f"S{i}", "name": name, "kind": r.kind, "osm_id": r.osm_id,
        "place_lat": round(r.place_lat, 6), "place_lon": round(r.place_lon, 6),
        "sensor_serial": serial, "sensor_address": r.address,
        "sensor_lat": r.sensor_lat, "sensor_lon": r.sensor_lon, "sensor_area_type": r.area_type,
        "dist_m": float(r.dist_m), "missing_rate": float(r.missing_rate),
        "in_seoul_major_parks_list": name.replace(" ", "") in seoul_names,
    })
json.dump(out, open("data/processed/selected_sites.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(pd.DataFrame(out)[["site_id", "name", "sensor_serial", "dist_m", "missing_rate", "sensor_area_type", "in_seoul_major_parks_list"]].to_string(index=False))
