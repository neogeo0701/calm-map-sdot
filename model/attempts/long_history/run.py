"""장기 이력 실험: 계절성 피처(월, 주차, 전년 동기 값)를 더한 LightGBM을 학습/검증/테스트 3분할로 평가한다.
분할·후보·선택 규칙은 DECISIONS.md(2026-10-27 장기 이력 실험)에 테스트 전에 기록했다.

사용법(저장소 루트에서):
  python model/attempts/long_history/run.py val    # 후보 3개 × 대상 2개 검증
  python model/attempts/long_history/run.py test   # 검증으로 고른 후보만 대상별 1회 테스트
"""
import json
import os
import sys

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = "model/attempts/long_history"
PHASE = sys.argv[1]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)  # 기존 try1과 같음
CONFIGS = {
    "S1": dict(lags=(168, 336), roll=True),   # 기존 try1 지연값 + 계절성
    "S2": dict(lags=(1, 24, 168), roll=False),  # 기존 base 지연값 + 계절성
    "S3": dict(lags=(), roll=False),          # 계절성만
}
TARGETS = ["level", "spike"]
YOY = 8736  # 364일 = 같은 요일
T_VAL, T_TEST = pd.Timestamp("2026-09-09 18:00"), pd.Timestamp("2026-09-24 11:00")  # 기존과 같은 구간
SERVICE_END = pd.Timestamp("2026-11-03 23:00")
H_LO, H_HI = 217, 384

sites = json.load(open("data/processed/selected_sites.json", encoding="utf-8"))
sel = [s["sensor_serial"] for s in sites]
raw = pd.read_pickle(f"{HERE}/hourly_5sensors.pkl")
t0 = raw.measured_at.min().floor("D")
t1 = pd.Timestamp("2026-10-18 23:00")
hours = pd.date_range(t0, SERVICE_END, freq="h")
N, S = len(hours), len(sel)
KR_HOL = holidays.KR(years=range(t0.year, 2027))

grid = pd.MultiIndex.from_product([sel, hours], names=["serial", "measured_at"])
d = raw[raw.measured_at <= t1].set_index(["serial", "measured_at"])[["noise_avg", "noise_max", "temp", "humid"]].reindex(grid).reset_index()
d["level"] = d.noise_avg
d["spike"] = d.noise_max - d.noise_avg
d["dow"] = d.measured_at.dt.dayofweek
d["hour"] = d.measured_at.dt.hour
d["weekend"] = (d.dow >= 5).astype(int)
d["holiday"] = d.measured_at.dt.date.map(lambda x: int(x in KR_HOL))
d["month"] = d.measured_at.dt.month
d["week"] = d.measured_at.dt.isocalendar().week.astype(int)
d["sensor"] = pd.Categorical(d.serial, categories=sel)


def idx(ts):
    return int((ts - t0) / pd.Timedelta(hours=1))


# 전년 동기 값(실측만, 재귀 구간보다 항상 과거라 예측값이 섞이지 않음)
for t in TARGETS:
    arr = d[t].to_numpy().reshape(S, N)
    yoy = np.full_like(arr, np.nan)
    yoy[:, YOY:] = arr[:, :-YOY]
    d[f"{t}_yoy"] = yoy.reshape(-1)
    wk = np.full_like(arr, np.nan)
    with np.errstate(all="ignore"):
        for j in range(YOY + 84, N):
            wk[:, j] = np.nanmean(arr[:, j - YOY - 84:j - YOY + 84], axis=1)
    d[f"{t}_yoy_week"] = wk.reshape(-1)

BASE = ["dow", "hour", "weekend", "holiday", "sensor", "temp", "humid", "month", "week"]


def feats(cfg, t):
    return BASE + [f"{t}_yoy", f"{t}_yoy_week"] + [f"{t}_lag{l}" for l in cfg["lags"]] + ([f"{t}_roll168"] if cfg["roll"] else [])


def roll168(arr, j):
    w = arr[:, max(j - 335, 0):j - 167]
    with np.errstate(all="ignore"):
        return np.nanmean(w, axis=1) if w.shape[1] else np.full(arr.shape[0], np.nan)


def add_lags(frame, cfg, t):
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


def weather_proxy(frame, cutoff):
    out = frame.copy()
    k = idx(cutoff)
    for c in ["temp", "humid"]:
        arr = frame[c].to_numpy().reshape(S, N).copy()
        last = arr[:, k - 168:k].copy()
        for j in range(k, N):
            arr[:, j] = last[:, (j - k) % 168]
        out[c] = arr.reshape(-1)
    return out


def recursive(frame, cfg, t, model, start):
    vals = frame[t].to_numpy().reshape(S, N).copy()
    k0 = idx(start)
    vals[:, k0:] = np.nan
    static = [c for c in feats(cfg, t) if c != "sensor" and "_lag" not in c and "_roll" not in c]
    block = {c: frame[c].to_numpy().reshape(S, N) for c in static}
    sensor_col = pd.Categorical(sel, categories=sel)
    pred = np.full((S, N), np.nan)
    for j in range(k0, N):
        row = pd.DataFrame({c: block[c][:, j] for c in static})
        row["sensor"] = sensor_col
        for l in cfg["lags"]:
            row[f"{t}_lag{l}"] = vals[:, j - l]
        if cfg["roll"]:
            row[f"{t}_roll168"] = roll168(vals, j)
        p = model.predict(row[feats(cfg, t)])
        pred[:, j] = p
        vals[:, j] = p
    return pred.reshape(-1)


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return round(float(np.mean(np.abs(a[m] - b[m]))), 4) if m.any() else None


def simple_methods(t, start):
    y = d[t].to_numpy()
    tr = d[(d.measured_at < start) & d[t].notna()]
    g = tr.groupby(["serial", "dow", "hour"])[t].mean().rename("b")
    base = d.join(g, on=["serial", "dow", "hour"])["b"].to_numpy()
    r4 = tr[tr.measured_at >= start - pd.Timedelta(days=28)].groupby(["serial", "dow", "hour"])[t].mean().rename("r")
    rec = d.join(r4, on=["serial", "dow", "hour"])["r"].to_numpy()
    arr = y.reshape(S, N)
    k = idx(start)
    last = arr[:, k - 168:k].copy()
    lw = np.full_like(arr, np.nan)
    for j in range(k, N):
        lw[:, j] = last[:, (j - k) % 168]
    return {"baseline": base, "ref_recent4w_mean": rec, "ref_last_week_naive": lw.reshape(-1), "ref_last_year_same": d[f"{t}_yoy"].to_numpy()}


def evaluate(start, end, cfg_by_target):
    """start 이전으로 학습 → [start, end] 재귀 예측 평가. 서비스 거리(217~384h) MAE와 센서별 값."""
    horizon = ((d.measured_at - start) / pd.Timedelta(hours=1)).to_numpy() + 1
    in_eval = ((d.measured_at >= start) & (d.measured_at <= end)).to_numpy()
    in_svc = in_eval & (horizon >= H_LO) & (horizon <= H_HI)
    dw = weather_proxy(d, start)
    out, preds = {}, {}
    for t, names in cfg_by_target.items():
        y = d[t].to_numpy()
        sm = simple_methods(t, start)
        res = {"n_service_hours": int((in_svc & ~np.isnan(y)).sum()),
               "hours_covered": [int(horizon[in_svc & ~np.isnan(y)].min()), int(horizon[in_svc & ~np.isnan(y)].max())],
               "service_horizon_mae": {k: mae(y[in_svc], v[in_svc]) for k, v in sm.items()},
               "overall_mae": {k: mae(y[in_eval], v[in_eval]) for k, v in sm.items()}, "by_sensor": {}, "feature_importance_gain": {}}
        for name in names:
            cfg = CONFIGS[name]
            add_lags(d, cfg, t)
            for c in [c for c in d.columns if c.startswith(f"{t}_lag") or c.startswith(f"{t}_roll")]:
                dw[c] = d[c]
            tr = d[(d.measured_at < start) & d[t].notna()]
            model = lgb.LGBMRegressor(**PARAMS).fit(tr[feats(cfg, t)], tr[t], categorical_feature=["sensor"])
            p = recursive(dw, cfg, t, model, start)
            res["service_horizon_mae"][f"ai_{name}"] = mae(y[in_svc], p[in_svc])
            res["overall_mae"][f"ai_{name}"] = mae(y[in_eval], p[in_eval])
            res["feature_importance_gain"][name] = {f: round(float(g), 1) for f, g in sorted(zip(feats(cfg, t), model.booster_.feature_importance("gain")), key=lambda x: -x[1])}
            preds[f"{t}_ai_{name}"] = p
            res["train_rows"] = int(len(tr))
        for s in sel:
            ss = in_svc & (d.serial == s).to_numpy()
            res["by_sensor"][s] = {**{k: mae(y[ss], v[ss]) for k, v in sm.items()},
                                   **{k[len(t) + 1:]: mae(y[ss], v[ss]) for k, v in preds.items() if k.startswith(t + "_")}}
        preds[f"{t}_actual"] = y
        for k, v in sm.items():
            preds[f"{t}_{k}"] = v
        out[t] = res
        print(t, json.dumps(res["service_horizon_mae"], ensure_ascii=False))
    pr = d.loc[in_eval, ["serial", "measured_at"]].copy()
    for k, v in preds.items():
        pr[k] = np.round(v[in_eval], 3)
    return out, pr


meta = {"data_start": str(t0), "data_end": str(t1), "split": {"train": [str(t0), str(T_VAL - pd.Timedelta(hours=1))],
        "val": [str(T_VAL), str(T_TEST - pd.Timedelta(hours=1))], "test": [str(T_TEST), str(t1)]},
        "train_hours_with_noise": {s: int(d[(d.serial == s) & (d.measured_at < T_VAL)].level.notna().sum()) for s in sel},
        "params": PARAMS, "configs": {k: {"lags": list(v["lags"]), "roll168": v["roll"]} for k, v in CONFIGS.items()},
        "seasonal_features": ["month", "week", "{t}_yoy(8736h 전)", "{t}_yoy_week(8736h 전 ±84h 평균)"]}

if PHASE == "val":
    res, pr = evaluate(T_VAL, T_TEST - pd.Timedelta(hours=1), {t: list(CONFIGS) for t in TARGETS})
    chosen = {t: min((k for k in res[t]["service_horizon_mae"] if k.startswith("ai_")), key=res[t]["service_horizon_mae"].get) for t in TARGETS}
    json.dump({**meta, "phase": "val", "results": res, "chosen_by_rule": chosen}, open(f"{HERE}/val_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pr.to_csv(f"{HERE}/val_predictions.csv", index=False, encoding="utf-8-sig")
    print("chosen:", chosen)
elif PHASE == "test":
    assert not os.path.exists(f"{HERE}/test_metrics.json"), "테스트는 1회만"
    chosen = json.load(open(f"{HERE}/val_metrics.json", encoding="utf-8"))["chosen_by_rule"]
    res, pr = evaluate(T_TEST, t1, {t: [chosen[t][3:]] for t in TARGETS})
    json.dump({**meta, "phase": "test", "chosen_by_rule": chosen, "results": res}, open(f"{HERE}/test_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pr.to_csv(f"{HERE}/test_predictions.csv", index=False, encoding="utf-8-sig")
