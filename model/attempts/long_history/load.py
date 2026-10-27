"""장기 이력: 연도별 zip(2024, 2025)과 2026 주간 CSV에서 선정 5개 센서의 소음·기온·습도만 뽑아 시간 단위로 합친다.
결측은 보간하지 않는다. 사용법(저장소 루트에서): python model/attempts/long_history/load.py"""
import glob
import io
import json
import os
import zipfile

import pandas as pd

HERE = "model/attempts/long_history"
SEL = [s["sensor_serial"] for s in json.load(open("data/processed/selected_sites.json", encoding="utf-8"))]
# 연도마다 열 이름이 조금씩 달라 키워드로 찾는다
WANT = {"serial": ["시리얼"], "measured_at": ["측정시간"], "temp": ["온도 평균"], "humid": ["습도 평균"],
        "noise_max": ["소음 최대"], "noise_avg": ["소음 평균"], "noise_min": ["소음 최소"]}


def pick_cols(cols):
    out = {}
    for k, keys in WANT.items():
        for c in cols:
            if any(key in str(c) for key in keys):
                out[c] = k
                break
    return out


def read_csv_bytes(b, name):
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            head = pd.read_csv(io.BytesIO(b), encoding=enc, nrows=0, index_col=False)
            break
        except UnicodeDecodeError:
            continue
    m = pick_cols(head.columns)
    if "serial" not in m.values() or "noise_avg" not in m.values():
        print("  열을 찾지 못함:", name, list(head.columns)[:12])
        return None
    chunks = []
    for ch in pd.read_csv(io.BytesIO(b), encoding=enc, usecols=list(m), dtype=str, index_col=False, chunksize=500_000):
        ch = ch.rename(columns=m)
        chunks.append(ch[ch.serial.isin(SEL)])
    return pd.concat(chunks) if chunks else None


frames, log = [], []
sources = sorted(glob.glob(f"{HERE}/raw/*.zip")) + sorted(glob.glob(f"{HERE}/raw/*.csv")) + sorted(glob.glob("data/raw/S-DoT_NATURE_*.csv"))
for src in sources:
    if src.endswith(".zip"):
        z = zipfile.ZipFile(src)
        for info in z.infolist():
            if info.filename.lower().endswith(".csv"):
                df = read_csv_bytes(z.read(info), info.filename)
                if df is not None:
                    frames.append(df)
                    log.append((os.path.basename(src) + ":" + info.filename.encode("cp437", "ignore").decode("cp949", "ignore"), len(df)))
    else:
        df = read_csv_bytes(open(src, "rb").read(), src)
        if df is not None:
            frames.append(df)
            log.append((os.path.basename(src), len(df)))
    print(src, "누적 행", sum(n for _, n in log))

d = pd.concat(frames, ignore_index=True)
d["measured_at"] = pd.to_datetime(d.measured_at.str.replace("_", " "), errors="coerce").dt.floor("h")
for c in ["temp", "humid", "noise_max", "noise_avg", "noise_min"]:
    if c in d:
        d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["measured_at"]).drop_duplicates(["serial", "measured_at"], keep="last")
d.to_pickle(f"{HERE}/hourly_5sensors.pkl")
cov = d.dropna(subset=["noise_avg"]).groupby("serial").measured_at.agg(["min", "max", "count"])
print(cov)
json.dump({"sources": log, "coverage": {k: {"min": str(v["min"]), "max": str(v["max"]), "hours_with_noise": int(v["count"])} for k, v in cov.iterrows()}},
          open(f"{HERE}/coverage.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
