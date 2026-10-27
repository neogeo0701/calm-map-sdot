"""매일 1회(GitHub Actions) 실행: 공개 주간 CSV로 5개 센서 이력을 갱신하고 화면 예측(public/predictions.json)을 다시 만든다.
인증키 불필요(열린데이터광장 파일 내려받기). 결측은 보간하지 않는다.

- 이력: data/live/hourly_5sensors.csv.gz (처음에는 model/attempts/long_history/hourly_5sensors.pkl에서 만든다)
- 유효기간: 실행일(한국시간) 00시 ~ +6일 23시
- 방법: pipeline/live_config.json (소음 수준·돌발 지수 각각 "ai" | "lastweek" | "baseline" | "recent4w")
- 오차·신뢰도·성능표·등급 기준은 검증된 결과(기존 predictions.json 값)를 그대로 둔다
사용법(저장소 루트): python pipeline/daily_update.py [--no-fetch] [--today YYYY-MM-DD]
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

LIVE = "data/live/hourly_5sensors.csv.gz"
SEED_PKL = "model/attempts/long_history/hourly_5sensors.pkl"
PRED = "public/predictions.json"
CFG = json.load(open("pipeline/live_config.json", encoding="utf-8"))
DATASET = "https://data.seoul.go.kr/dataList/OA-15969/S/1/datasetView.do"
DOWNLOAD = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
KST = pd.Timedelta(hours=9)
# 장기 이력 S1 설정(DECISIONS.md) — 바꾸지 않음
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
LAGS, YOY = (168, 336), 8736
COLS = {"시리얼": "serial", "측정시간": "measured_at", "온도 평균": "temp", "습도 평균": "humid",
        "소음 최대": "noise_max", "소음 평균": "noise_avg", "소음 최소": "noise_min"}

sites = json.load(open("data/processed/selected_sites.json", encoding="utf-8"))
SEL = [s["sensor_serial"] for s in sites]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. 이력 갱신
def load_history():
    if os.path.exists(LIVE):
        h = pd.read_csv(LIVE, parse_dates=["measured_at"])
    else:
        h = pd.read_pickle(SEED_PKL)[["serial", "measured_at", "temp", "humid", "noise_max", "noise_avg", "noise_min"]]
    return h


def list_weekly_files():
    html = urllib.request.urlopen(urllib.request.Request(DATASET, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8", "replace")
    out = []
    for m in re.finditer(r"title=\"(S-DoT_NATURE_(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.csv)\" onclick=\"javascript:downloadFile\('(\d+)'\)", html):
        name, y, m1, d1, m2, d2, seq = m.groups()
        start = pd.Timestamp(f"{y}-{m1}-{d1}")
        end = pd.Timestamp(f"{int(y) + (1 if int(m2) < int(m1) else 0)}-{m2}-{d2}") + pd.Timedelta(hours=23)
        out.append((name, seq, start, end))
    return sorted(set(out), key=lambda x: x[2])


def download_csv(seq):
    body = urllib.parse.urlencode({"infId": "OA-15969", "seqNo": "", "seq": seq, "infSeq": "3"}).encode()
    req = urllib.request.Request(DOWNLOAD, data=body, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=600).read()


def parse_csv(b):
    head = pd.read_csv(io.BytesIO(b), encoding="cp949", nrows=0, index_col=False)
    m = {}
    for k, v in COLS.items():
        c = next((c for c in head.columns if k in str(c)), None)
        if c is not None:
            m[c] = v
    df = pd.concat(ch.rename(columns=m)[lambda x: x.serial.isin(SEL)] for ch in
                   pd.read_csv(io.BytesIO(b), encoding="cp949", usecols=list(m), dtype=str, index_col=False, chunksize=500_000))
    df["measured_at"] = pd.to_datetime(df.measured_at.str.replace("_", " "), errors="coerce").dt.floor("h")
    for c in ["temp", "humid", "noise_max", "noise_avg", "noise_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["measured_at"])


def update_history(h, fetch=True):
    last = h.measured_at.max()
    added = []
    if fetch:
        for name, seq, start, end in list_weekly_files():
            if end <= last:
                continue
            log("내려받기:", name)
            try:
                df = parse_csv(download_csv(seq))
            except Exception as e:  # 한 파일 실패가 전체를 멈추지 않게
                log("  실패:", name, e)
                continue
            h = pd.concat([h, df[h.columns]], ignore_index=True)
            added.append(name)
    h = h.drop_duplicates(["serial", "measured_at"], keep="last").sort_values(["serial", "measured_at"])
    os.makedirs(os.path.dirname(LIVE), exist_ok=True)
    # gzip 헤더 시각을 0으로 고정해 내용이 같으면 파일도 같게(새 자료가 없는 날 불필요한 커밋 방지)
    data = h.to_csv(index=False, float_format="%.4g").encode("utf-8")
    with open(LIVE, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        f.write(data)
    return h, added


# ---------------------------------------------------------------- 2. 예측
def build_frame(h, end):
    t0 = h.measured_at.min().floor("D")
    t1 = h.dropna(subset=["noise_avg"]).measured_at.max()
    hours = pd.date_range(t0, end, freq="h")
    grid = pd.MultiIndex.from_product([SEL, hours], names=["serial", "measured_at"])
    d = h[h.measured_at <= t1].set_index(["serial", "measured_at"])[["noise_avg", "noise_max", "temp", "humid"]].reindex(grid).reset_index()
    d["level"], d["spike"] = d.noise_avg, d.noise_max - d.noise_avg
    kr = holidays.KR(years=range(t0.year, end.year + 1))
    d["dow"], d["hour"] = d.measured_at.dt.dayofweek, d.measured_at.dt.hour
    d["weekend"] = (d.dow >= 5).astype(int)
    d["holiday"] = d.measured_at.dt.date.map(lambda x: int(x in kr))
    d["month"] = d.measured_at.dt.month
    d["week"] = d.measured_at.dt.isocalendar().week.astype(int)
    d["sensor"] = pd.Categorical(d.serial, categories=SEL)
    return d, t0, t1, len(hours)


def forecast(d, t0, t1, N, target, method):
    S = len(SEL)
    arr = d[target].to_numpy().reshape(S, N)
    k = int((t1 + pd.Timedelta(hours=1) - t0) / pd.Timedelta(hours=1))
    if method == "lastweek":
        last = arr[:, k - 168:k]
        out = np.full_like(arr, np.nan)
        for j in range(k, N):
            out[:, j] = last[:, (j - k) % 168]
        return out
    if method in ("baseline", "recent4w"):
        hist = d[(d.measured_at <= t1) & d[target].notna()]
        if method == "recent4w":
            hist = hist[hist.measured_at > t1 - pd.Timedelta(days=28)]
        g = hist.groupby(["serial", "dow", "hour"])[target].mean().rename("v")
        return d.join(g, on=["serial", "dow", "hour"])["v"].to_numpy().reshape(S, N)
    # AI(S1): 계절성 + 1·2주 전 같은 시각 + 1주 전 주간 평균, 재귀 예측
    yoy = np.full_like(arr, np.nan)
    yoy[:, YOY:] = arr[:, :-YOY]
    wk = np.full_like(arr, np.nan)
    with np.errstate(all="ignore"):
        for j in range(YOY + 84, N):
            wk[:, j] = np.nanmean(arr[:, j - YOY - 84:j - YOY + 84], axis=1)

    def roll(a, j):
        w = a[:, max(j - 335, 0):j - 167]
        with np.errstate(all="ignore"):
            return np.nanmean(w, axis=1) if w.shape[1] else np.full(S, np.nan)

    X = d.copy()
    X[f"{target}_yoy"], X[f"{target}_yoy_week"] = yoy.reshape(-1), wk.reshape(-1)
    for l in LAGS:
        sh = np.full_like(arr, np.nan)
        sh[:, l:] = arr[:, :-l]
        X[f"{target}_lag{l}"] = sh.reshape(-1)
    r = np.full_like(arr, np.nan)
    for j in range(N):
        r[:, j] = roll(arr, j)
    X[f"{target}_roll168"] = r.reshape(-1)
    feats = ["dow", "hour", "weekend", "holiday", "sensor", "temp", "humid", "month", "week", f"{target}_yoy", f"{target}_yoy_week",
             f"{target}_lag168", f"{target}_lag336", f"{target}_roll168"]
    tr = X[(X.measured_at <= t1) & X[target].notna()]
    model = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], tr[target], categorical_feature=["sensor"])
    for c in ["temp", "humid"]:  # 미래 기온·습도 = 직전 7일 같은 시각
        a = X[c].to_numpy().reshape(S, N).copy()
        last = a[:, k - 168:k].copy()
        for j in range(k, N):
            a[:, j] = last[:, (j - k) % 168]
        X[c] = a.reshape(-1)
    static = [c for c in feats if c != "sensor" and "_lag" not in c and "_roll" not in c]
    block = {c: X[c].to_numpy().reshape(S, N) for c in static}
    vals = arr.copy()
    vals[:, k:] = np.nan
    out = np.full_like(arr, np.nan)
    sc = pd.Categorical(SEL, categories=SEL)
    for j in range(k, N):
        row = pd.DataFrame({c: block[c][:, j] for c in static})
        row["sensor"] = sc
        for l in LAGS:
            row[f"{target}_lag{l}"] = vals[:, j - l]
        row[f"{target}_roll168"] = roll(vals, j)
        p = model.predict(row[feats])
        out[:, j], vals[:, j] = p, p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--today", help="유효기간 시작일(한국시간). 기본: 오늘")
    a = ap.parse_args()
    today = pd.Timestamp(a.today) if a.today else (pd.Timestamp.utcnow().tz_localize(None) + KST).normalize()
    win0, win1 = today, today + pd.Timedelta(days=6, hours=23)

    h, added = update_history(load_history(), fetch=not a.no_fetch)
    d, t0, t1, N = build_frame(h, win1)
    S = len(SEL)
    vals = {t: forecast(d, t0, t1, N, t, CFG[t]) for t in ("level", "spike")}
    base = {t: forecast(d, t0, t1, N, t, "baseline") for t in ("level", "spike")}
    cnt = {}
    for t in ("level", "spike"):
        hist = d[(d.measured_at <= t1) & d[t].notna()]
        cnt[t] = d.join(hist.groupby(["serial", "dow", "hour"])[t].count().rename("n"), on=["serial", "dow", "hour"])["n"].fillna(0).to_numpy().reshape(S, N)
    k1 = int((t1 + pd.Timedelta(hours=1) - t0) / pd.Timedelta(hours=1))
    cov = {t: (~np.isnan(d[t].to_numpy().reshape(S, N)[:, k1 - 336:k1])).mean(axis=1) for t in ("level", "spike")}
    times = pd.date_range(t0, win1, freq="h")
    w0 = int((win0 - t0) / pd.Timedelta(hours=1))

    P = json.load(open(PRED, encoding="utf-8"))
    for si, site in enumerate(P["sites"]):
        r = h[(h.serial == SEL[si]) & h.noise_avg.notna()].sort_values("measured_at")
        last = r.iloc[-1]
        wk = r[r.measured_at > t1 - pd.Timedelta(hours=168)]
        site["recent"] = {"t": last.measured_at.strftime("%Y-%m-%dT%H:00"), "level": float(last.noise_avg), "max": float(last.noise_max),
                          "min": float(last.noise_min), "spike": round(float(last.noise_max - last.noise_avg), 1)}
        site["recent_week"] = {"level": round(float(wk.noise_avg.mean()), 1), "spike": round(float((wk.noise_max - wk.noise_avg).mean()), 1)}
        fc = []
        for j in range(w0, N):
            e = {"t": times[j].strftime("%Y-%m-%dT%H:00")}
            for t in ("level", "spike"):
                v = vals[t][si, j]
                ok = not np.isnan(v) and (CFG[t] != "ai" or cov[t][si] >= 0.5)
                e[t] = round(float(v), 1) if ok else None
                bv = base[t][si, j]
                e[f"{t}_baseline"] = round(float(bv), 1) if (cnt[t][si, j] >= 3 and not np.isnan(bv)) else None
            fc.append(e)
        site["forecast"] = fc
        # 화면에 쓰는 방법의 테스트 평균 오차와 신뢰도(고정값 mae_all에서 고름)
        for t in ("level", "spike"):
            ma = site["mae_all"][t]
            site["mae"][t] = {**ma, "used": ma[CFG[t]]}
            u = ma[CFG[t]]
            site["confidence"][t] = "높음" if u < 1.5 else ("보통" if u <= 3.0 else "낮음")
    P["generated_at"] = (pd.Timestamp.utcnow().tz_localize(None) + KST).strftime("%Y-%m-%dT%H:%M")
    P["data_period"] = [str(t0)[:16], str(t1)[:16]]
    P["forecast_period"] = [str(win0)[:16], str(win1)[:16]]
    P["valid_until"] = str(win1)[:16]
    P["level_source_week"] = [str((t1 - pd.Timedelta(hours=167)).date()), str(t1.date())]
    P["horizon_now_hours"] = [int((win0 - t1) / pd.Timedelta(hours=1)), int((win1 - t1) / pd.Timedelta(hours=1))]
    P["live_methods"] = {t: CFG[t] for t in ("level", "spike")}
    P["update"] = {"source": "서울 열린데이터광장 OA-15969 주간 CSV(공개 파일)", "added_files": added, "schedule": "매일 06:10(한국시간)"}
    json.dump(P, open(PRED, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    log("마지막 실측", t1, "유효기간", win0.date(), "~", win1.date(), "추가 파일", added)


if __name__ == "__main__":
    main()
