"""2단계: 소음 수준(시간 평균)과 돌발 소음 지수(시간 최대 - 시간 평균)를 예측하고 기준 모델 대비 MAE를 검증한다.

평가
  E1  1시간 앞 예측: 직전 1·24·168시간 실측을 그대로 넣는다(핸드오프 명세).
  E2  재귀 다중 시점 예측: 학습 끝 시각부터 테스트 끝까지 예측값을 다시 입력으로 넣어 이어서 예측한다.
      서비스는 데이터 마지막 시각에서 217~384시간 뒤를 보여주므로 이 구간 MAE로 서비스 모델을 고른다.
결측은 보간하지 않는다. 실측이 없는 시간은 학습·평가에서 빠지고, 입력 지연값이 없으면 NaN 그대로 LightGBM에 들어간다.

사용법: python pipeline/04_model.py <attempt>   (attempt: base | try1 | try2)
"""
import json
import os
import sys

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

ATTEMPT = sys.argv[1] if len(sys.argv) > 1 else "base"
BASE_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
CONFIGS = {
    "base": dict(params=BASE_PARAMS, lags=(1, 24, 168), roll=False, train_pool="selected"),
    # 개선 1: 재귀 오차 누적을 줄이려고 직전 1·24시간을 빼고 1·2주 전 같은 시각과 1주 전 한 주 평균을 넣음
    "try1": dict(params=BASE_PARAMS, lags=(168, 336), roll=True, train_pool="selected"),
    # 개선 2: 개선 1 + 송파구 전체 센서로 학습 데이터 확대 + 더 작은 나무·더 많은 반복
    "try2": dict(params={**BASE_PARAMS, "n_estimators": 800, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 40},
                 lags=(168, 336), roll=True, train_pool="Songpa-gu"),
}
cfg = CONFIGS[ATTEMPT]
SERVICE_START, SERVICE_END = pd.Timestamp("2026-10-28 00:00"), pd.Timestamp("2026-11-03 23:00")
TARGETS = {"level": "소음 수준(시간 평균, dB)", "spike": "돌발 소음 지수(시간 최대 - 시간 평균, dB)"}
KR_HOL = holidays.KR(years=[2026])

sites = json.load(open("data/processed/selected_sites.json", encoding="utf-8"))
sel = [s["sensor_serial"] for s in sites]
raw = pd.read_pickle("data/processed/sdot_hourly_all.pkl")
if cfg["train_pool"] == "selected":
    pool = sel
else:
    pool = sel + sorted(set(raw.loc[raw.district == cfg["train_pool"], "serial"]) - set(sel))
raw = raw[raw.serial.isin(pool)]
t0, t1 = raw.measured_at.min(), raw.measured_at.max()
t_split = (t0 + (t1 - t0) * 0.75).floor("h")
hours = pd.date_range(t0, SERVICE_END, freq="h")
N, S = len(hours), len(pool)

# 센서 x 시간 격자 (결측은 NaN으로 남김)
grid = pd.MultiIndex.from_product([pool, hours], names=["serial", "measured_at"])
d = raw.set_index(["serial", "measured_at"])[["noise_avg", "noise_max", "temp", "humid"]].reindex(grid).reset_index()
d["level"] = d["noise_avg"]
d["spike"] = d["noise_max"] - d["noise_avg"]
d["dow"] = d.measured_at.dt.dayofweek
d["hour"] = d.measured_at.dt.hour
d["weekend"] = (d.dow >= 5).astype(int)
d["holiday"] = d.measured_at.dt.date.map(lambda x: int(x in KR_HOL))
d["sensor"] = pd.Categorical(d.serial, categories=pool)
# MultiIndex.from_product 순서 = pool 순서 x 시간 순서 → reshape(S, N) 가능


def lag_cols(t):
    return [f"{t}_lag{l}" for l in cfg["lags"]] + ([f"{t}_roll168"] if cfg["roll"] else [])


def roll168(arr, j):
    """j 시각 기준 168~335시간 전 한 주(168시간)의 평균. 실측이 없는 시간은 빼고 평균."""
    w = arr[:, max(j - 335, 0):j - 167]
    with np.errstate(all="ignore"):
        return np.nanmean(w, axis=1) if w.shape[1] else np.full(arr.shape[0], np.nan)


BASE_FEATS = ["dow", "hour", "weekend", "holiday", "sensor", "temp", "humid"]
FEATS = {t: BASE_FEATS + lag_cols(t) for t in TARGETS}


def add_lags(frame, t):
    arr = frame[t].to_numpy().reshape(S, N)
    for l in cfg["lags"]:
        sh = np.full_like(arr, np.nan)
        sh[:, l:] = arr[:, :-l]
        frame[f"{t}_lag{l}"] = sh.reshape(-1)
    if cfg["roll"]:
        r = np.full_like(arr, np.nan)
        for j in range(N):
            r[:, j] = roll168(arr, j)
        frame[f"{t}_roll168"] = r.reshape(-1)


def idx(ts):
    return int((ts - t0) / pd.Timedelta(hours=1))


def weather_proxy(frame, cutoff):
    """cutoff 이후 기온·습도를 cutoff 직전 7일 같은 시각 실측으로 대체한다(미래 기상은 모름)."""
    out = frame.copy()
    k = idx(cutoff)
    for c in ["temp", "humid"]:
        arr = frame[c].to_numpy().reshape(S, N).copy()
        last_week = arr[:, k - 168:k].copy()
        for j in range(k, N):
            arr[:, j] = last_week[:, (j - k) % 168]
        out[c] = arr.reshape(-1)
    return out


def fit(frame, t, until):
    m = frame[(frame.measured_at < until) & frame[t].notna()]
    model = lgb.LGBMRegressor(**cfg["params"])
    model.fit(m[FEATS[t]], m[t], categorical_feature=["sensor"])
    return model


def baseline(frame, t, until):
    """기준 모델: 센서별 요일 x 시간 평균(학습 구간)."""
    m = frame[(frame.measured_at < until) & frame[t].notna()]
    g = m.groupby(["serial", "dow", "hour"])[t].agg(["mean", "count"])
    j = frame.join(g, on=["serial", "dow", "hour"])
    return j["mean"].to_numpy(), j["count"].fillna(0).to_numpy()


def recent_baseline(frame, t, until, days=28):
    """참고 기준 1: 직전 days일만 쓴 센서별 요일 x 시간 평균(계절 변화 반영)."""
    m = frame[(frame.measured_at < until) & (frame.measured_at >= until - pd.Timedelta(days=days)) & frame[t].notna()]
    g = m.groupby(["serial", "dow", "hour"])[t].mean().rename("r")
    return frame.join(g, on=["serial", "dow", "hour"])["r"].to_numpy()


def last_week_naive(frame, t, until):
    """참고 기준 2: 직전 7일 실측을 그대로 반복."""
    arr = frame[t].to_numpy().reshape(S, N).copy()
    k = idx(until)
    last = arr[:, k - 168:k].copy()
    out = np.full_like(arr, np.nan)
    for j in range(k, N):
        out[:, j] = last[:, (j - k) % 168]
    return out.reshape(-1)


def recursive(frame, t, model, start):
    """start 시각부터 끝까지 1시간씩 예측하고, 예측값을 다음 시각의 지연 입력으로 쓴다."""
    vals = frame[t].to_numpy().reshape(S, N).copy()
    k0 = idx(start)
    vals[:, k0:] = np.nan
    base_block = {c: frame[c].to_numpy().reshape(S, N) for c in BASE_FEATS if c != "sensor"}
    sensor_col = pd.Categorical(pool, categories=pool)
    pred = np.full((S, N), np.nan)
    for j in range(k0, N):
        row = pd.DataFrame({c: base_block[c][:, j] for c in base_block})
        row["sensor"] = sensor_col
        for l in cfg["lags"]:
            row[f"{t}_lag{l}"] = vals[:, j - l]
        if cfg["roll"]:
            row[f"{t}_roll168"] = roll168(vals, j)
        p = model.predict(row[FEATS[t]])
        pred[:, j] = p
        vals[:, j] = p
    return pred.reshape(-1)


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return round(float(np.mean(np.abs(a[m] - b[m]))), 4) if m.any() else None


def pct(b, a):
    return round(100 * (b - a) / b, 2) if (a is not None and b) else None


if __name__ == "__main__":
    h_lo, h_hi = idx(SERVICE_START) - idx(t1), idx(SERVICE_END) - idx(t1)
    metrics = {
        "attempt": ATTEMPT,
        "config": {"params": cfg["params"], "lags": list(cfg["lags"]), "train_pool": cfg["train_pool"], "features": FEATS},
        "data_period": [str(t0), str(t1)],
        "train_period": [str(t0), str(t_split - pd.Timedelta(hours=1))],
        "test_period": [str(t_split), str(t1)],
        "service_period": [str(SERVICE_START), str(SERVICE_END)],
        "service_horizon_hours": [h_lo, h_hi],
        "sensors_train_pool": S, "sensors_eval": sel, "targets": {},
    }
    in_test = ((d.measured_at >= t_split) & (d.measured_at <= t1) & d.serial.isin(sel)).to_numpy()
    horizon = ((d.measured_at - t_split) / pd.Timedelta(hours=1)).to_numpy() + 1
    test_out = d.loc[in_test, ["serial", "measured_at"]].copy()
    dw = weather_proxy(d, t_split)
    buckets = {"1-24h": (1, 24), "25-168h": (25, 168), "169-216h": (169, 216),
               f"service_{h_lo}-{h_hi}h": (h_lo, h_hi), f"{h_hi + 1}h+": (h_hi + 1, 10 ** 6)}

    for t, label in TARGETS.items():
        add_lags(d, t)
        model = fit(d, t, t_split)
        base, _ = baseline(d, t, t_split)
        e1 = model.predict(d[FEATS[t]])
        e2 = recursive(dw, t, model, t_split)
        r4 = recent_baseline(d, t, t_split)
        lw = last_week_naive(d, t, t_split)
        y = d[t].to_numpy()
        res = {"label": label, "n_test_hours": int((in_test & ~np.isnan(y)).sum())}
        o = {"baseline_mae": mae(y[in_test], base[in_test]), "ai_1h_mae": mae(y[in_test], e1[in_test]),
             "ai_recursive_mae": mae(y[in_test], e2[in_test]),
             "ref_recent4w_mean_mae": mae(y[in_test], r4[in_test]), "ref_last_week_naive_mae": mae(y[in_test], lw[in_test])}
        o["ai_1h_improvement_pct"] = pct(o["baseline_mae"], o["ai_1h_mae"])
        o["ai_recursive_improvement_pct"] = pct(o["baseline_mae"], o["ai_recursive_mae"])
        res["overall"] = o
        res["by_horizon_recursive"] = {}
        for name, (lo, hi) in buckets.items():
            bm = in_test & (horizon >= lo) & (horizon <= hi)
            b, a = mae(y[bm], base[bm]), mae(y[bm], e2[bm])
            res["by_horizon_recursive"][name] = {"n": int((bm & ~np.isnan(y)).sum()), "baseline_mae": b,
                                                 "ai_recursive_mae": a, "ai_improvement_pct": pct(b, a),
                                                 "ref_recent4w_mean_mae": mae(y[bm], r4[bm]),
                                                 "ref_last_week_naive_mae": mae(y[bm], lw[bm])}
        sv = res["by_horizon_recursive"][f"service_{h_lo}-{h_hi}h"]
        res["by_sensor"] = {}
        for s in sel:
            sm = in_test & (d.serial == s).to_numpy()
            ss = sm & (horizon >= h_lo) & (horizon <= h_hi)
            res["by_sensor"][s] = {"baseline_mae": mae(y[sm], base[sm]), "ai_1h_mae": mae(y[sm], e1[sm]),
                                   "ai_recursive_mae": mae(y[sm], e2[sm]),
                                   "service_horizon_baseline_mae": mae(y[ss], base[ss]),
                                   "service_horizon_ai_mae": mae(y[ss], e2[ss]),
                                   "service_horizon_ref_recent4w_mae": mae(y[ss], r4[ss])}
        res["service_model"] = "ai" if sv["ai_recursive_mae"] < sv["baseline_mae"] else "baseline"
        gain = model.booster_.feature_importance("gain")
        res["feature_importance_gain"] = {f: round(float(g), 1) for f, g in sorted(zip(FEATS[t], gain), key=lambda x: -x[1])}
        metrics["targets"][t] = res
        for k, v in {"actual": y, "baseline": base, "ai_1h": e1, "ai_recursive": e2, "ref_recent4w": r4}.items():
            test_out[f"{t}_{k}"] = np.round(v[in_test], 3)
        print(t, json.dumps(o, ensure_ascii=False))
        for k, v in res["by_horizon_recursive"].items():
            print("   ", k, v)
        print("    service model:", res["service_model"])

    out_dir = f"model/attempts/{ATTEMPT}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(metrics, open(f"{out_dir}/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    test_out.to_csv(f"{out_dir}/test_predictions.csv", index=False, encoding="utf-8-sig")
