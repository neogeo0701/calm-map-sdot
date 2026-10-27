"""순차 백테스트(설계는 DECISIONS.md "순차 백테스트 — 실행 전 고정"에 먼저 기록).
기준일 20개마다 그 이전 실측만으로 학습·계산하고 9~15일 뒤 9~21시를 예측해, 앱과 같은 방식으로 추천을 만든 뒤 실측과 비교한다.
사용법(저장소 루트): python model/attempts/backtest/backtest.py
"""
import importlib.util
import json
import sys
from itertools import product

import numpy as np
import pandas as pd

OUT = "model/attempts/backtest"
sys.argv = ["run.py", "none"]
spec = importlib.util.spec_from_file_location("lh", "model/attempts/long_history/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
d, sel, S, N = R.d, R.sel, R.S, R.N
names = {s["sensor_serial"]: s["name"] for s in R.sites}

ORIGINS = [pd.Timestamp("2024-06-03") + pd.Timedelta(days=42 * i) for i in range(20)]
assert ORIGINS[-1] == pd.Timestamp("2026-08-10")
assert ORIGINS[-1] + pd.Timedelta(days=16) < pd.Timestamp("2026-09-24 11:00")  # 기존 테스트 구간과 겹치지 않음
HOURS = list(range(9, 21))
SLOTS = {"9-12": (9, 12), "12-15": (12, 15), "15-18": (15, 18), "18-21": (18, 21)}
W = {"low": 0.5, "mid": 1.0, "high": 2.0}
WEIGHTS = list(product(W, W))
CUTS = {"level": (41.0, 45.0), "spike": (3.0, 4.0)}  # 화면과 같은 등급 기준(고정)
CFG = R.CONFIGS["S1"]
for t in ("level", "spike"):
    R.add_lags(d, CFG, t)  # 실측 기반 지연값(학습용). 예측 구간에서는 재귀로 덮어씀
ACT = {t: d[t].to_numpy().reshape(S, N) for t in ("level", "spike")}
DOW = d.dow.to_numpy().reshape(S, N)[0]
HOUR = d.hour.to_numpy().reshape(S, N)[0]


def grade(v, t):
    a, b = CUTS[t]
    return np.where(np.isnan(v), -1, np.where(v < a, 0, np.where(v < b, 1, 2)))  # 2 = 많은 편


def ai_forecast(t, T, steps):
    """T 이전으로 학습한 S1 모델의 재귀 예측(T부터 steps시간)."""
    k0 = R.idx(T)
    tr = d[(d.measured_at < T) & d[t].notna()]
    model = R.lgb.LGBMRegressor(**R.PARAMS).fit(tr[R.feats(CFG, t)], tr[t], categorical_feature=["sensor"])
    dw = R.weather_proxy(d, T)
    static = [c for c in R.feats(CFG, t) if c != "sensor" and "_lag" not in c and "_roll" not in c]
    block = {c: dw[c].to_numpy().reshape(S, N) for c in static}
    vals = ACT[t].copy()
    vals[:, k0:] = np.nan
    out = np.full((S, steps), np.nan)
    sensor_col = pd.Categorical(sel, categories=sel)
    for i in range(steps):
        j = k0 + i
        row = pd.DataFrame({c: block[c][:, j] for c in static})
        row["sensor"] = sensor_col
        for l in CFG["lags"]:
            row[f"{t}_lag{l}"] = vals[:, j - l]
        if CFG["roll"]:
            row[f"{t}_roll168"] = R.roll168(vals, j)
        p = model.predict(row[R.feats(CFG, t)])
        out[:, i] = p
        vals[:, j] = p
    return out


def simple_forecasts(t, T, steps):
    k0 = R.idx(T)
    arr = ACT[t]
    hist = arr[:, :k0]
    dow_h, hr_h = DOW[:k0], HOUR[:k0]
    tgt_dow, tgt_hr = DOW[k0:k0 + steps], HOUR[k0:k0 + steps]

    def dh_mean(h, dw, hr):
        out = np.full((S, steps), np.nan)
        for dd in range(7):
            for hh in range(24):
                m = (dw == dd) & (hr == hh)
                if not m.any():
                    continue
                with np.errstate(all="ignore"):
                    v = np.nanmean(h[:, m], axis=1)
                out[:, (tgt_dow == dd) & (tgt_hr == hh)] = v[:, None]
        return out

    base = dh_mean(hist, dow_h, hr_h)
    r4 = dh_mean(hist[:, -28 * 24:], dow_h[-28 * 24:], hr_h[-28 * 24:])
    last = arr[:, k0 - 168:k0]
    lw = np.stack([last[:, i % 168] for i in range(steps)], axis=1)
    return {"baseline": base, "recent4w": r4, "lastweek": lw}


METHODS = ["screen", "ai", "baseline", "recent4w", "lastweek"]
LABEL = {"screen": "현재 화면(돌발 AI + 소음 마지막 관측 주)", "ai": "AI 단독(두 대상 모두 AI)", "baseline": "요일×시간 평균",
         "recent4w": "직전 4주 평균", "lastweek": "마지막 관측 주 값"}
# 기준일별 누적값: M1(hit,total), M2(overlap,units), M3(regret_sum, units, err_regret_sum, err_units), M4 spike/level(num, den)
rows = []
place_rows = []
for oi, T in enumerate(ORIGINS):
    steps = 16 * 24
    pred = {}
    ai = {t: ai_forecast(t, T, steps) for t in ("level", "spike")}
    simp = {t: simple_forecasts(t, T, steps) for t in ("level", "spike")}
    for m in ["baseline", "recent4w", "lastweek"]:
        pred[m] = {t: simp[t][m] for t in ("level", "spike")}
    pred["ai"] = ai
    pred["screen"] = {"level": simp["level"]["lastweek"], "spike": ai["spike"]}
    k0 = R.idx(T)
    act = {t: ACT[t][:, k0:k0 + steps] for t in ("level", "spike")}
    times = pd.date_range(T, periods=steps, freq="h")
    stat = {m: dict(m1h=0, m1n=0, m2o=0, m2n=0, m3s=0.0, m3n=0, m3es=0.0, m3en=0, m4sn=0, m4sd=0, m4ln=0, m4ld=0) for m in METHODS}
    pstat = {(m, s): dict(m1h=0, m1n=0, m4sn=0, m4sd=0) for m in METHODS for s in range(S)}
    for day in range(9, 16):
        idx_day = [i for i, ts in enumerate(times) if (ts.normalize() - T).days == day and ts.hour in HOURS]
        if not idx_day:
            continue
        ii = np.array(idx_day)
        hrs = HOUR[k0 + ii]
        A = {t: act[t][:, ii] for t in ("level", "spike")}
        ok = ~np.isnan(A["level"]) & ~np.isnan(A["spike"])
        for m in METHODS:
            for t in ("level", "spike"):
                ok &= ~np.isnan(pred[m][t][:, ii])
        for wl, ws in WEIGHTS:
            sc_a = W[wl] * (A["level"] - 35) + W[ws] * A["spike"]
            sc_p = {m: W[wl] * (pred[m]["level"][:, ii] - 35) + W[ws] * pred[m]["spike"][:, ii] for m in METHODS}
            # M1: 장소별 시간대 적중
            for s in range(S):
                a_slot, p_slot = {}, {m: {} for m in METHODS}
                for sl, (h0, h1) in SLOTS.items():
                    mk = ok[s] & (hrs >= h0) & (hrs < h1)
                    if mk.sum() >= 2:
                        a_slot[sl] = sc_a[s, mk].mean()
                        for m in METHODS:
                            p_slot[m][sl] = sc_p[m][s, mk].mean()
                if len(a_slot) < 2:
                    continue
                amin = min(a_slot.values())
                for m in METHODS:
                    pick = min(p_slot[m], key=p_slot[m].get)
                    hit = int(np.isclose(a_slot[pick], amin))
                    stat[m]["m1h"] += hit; stat[m]["m1n"] += 1
                    pstat[(m, s)]["m1h"] += hit; pstat[(m, s)]["m1n"] += 1
            # M2~M4: 시간대 단위 추천
            for sl, (h0, h1) in SLOTS.items():
                inslot = (hrs >= h0) & (hrs < h1)
                places = [s for s in range(S) if (ok[s] & inslot).any()]
                if len(places) < 3:
                    continue

                def top3(sc):
                    best = []
                    for s in places:
                        cols = np.where(ok[s] & inslot)[0]
                        c = cols[np.argmin(sc[s, cols])]
                        best.append((sc[s, c], s, c))
                    return sorted(best)[:3]

                act_top = top3(sc_a)
                act_min = min(sc_a[s, c] for s in places for c in np.where(ok[s] & inslot)[0])
                act_set = {s for _, s, _ in act_top}
                for m in METHODS:
                    pt = top3(sc_p[m])
                    st = stat[m]
                    st["m2o"] += len(act_set & {s for _, s, _ in pt}); st["m2n"] += 1
                    _, s1, c1 = pt[0]
                    reg = sc_a[s1, c1] - act_min
                    st["m3s"] += reg; st["m3n"] += 1
                    if reg > 1e-9:
                        st["m3es"] += reg; st["m3en"] += 1
                    for _, s_, c_ in pt:
                        if grade(np.array(A["spike"][s_, c_]), "spike") == 2:
                            st["m4sd"] += 1; pstat[(m, s_)]["m4sd"] += 1
                            if grade(np.array(pred[m]["spike"][s_, ii[c_]]), "spike") < 2:
                                st["m4sn"] += 1; pstat[(m, s_)]["m4sn"] += 1
                        if grade(np.array(A["level"][s_, c_]), "level") == 2:
                            st["m4ld"] += 1
                            if grade(np.array(pred[m]["level"][s_, ii[c_]]), "level") < 2:
                                st["m4ln"] += 1
    for m in METHODS:
        rows.append({"origin": str(T.date()), "method": m, **stat[m]})
    for (m, s), v in pstat.items():
        place_rows.append({"origin": str(T.date()), "method": m, "place": names[sel[s]], **v})
    print(T.date(), {m: (stat[m]["m1n"], round(stat[m]["m1h"] / max(stat[m]["m1n"], 1), 3), stat[m]["m4sd"]) for m in ["screen", "lastweek"]}, flush=True)

df = pd.DataFrame(rows)
pf = pd.DataFrame(place_rows)
df.to_csv(f"{OUT}/per_origin.csv", index=False, encoding="utf-8-sig")
pf.to_csv(f"{OUT}/per_origin_place.csv", index=False, encoding="utf-8-sig")


def pooled(g):
    s = g.sum(numeric_only=True)
    return {"M1": s.m1h / s.m1n if s.m1n else np.nan, "M2": s.m2o / (3 * s.m2n) if s.m2n else np.nan,
            "M3_all": s.m3s / s.m3n if s.m3n else np.nan, "M3_err": s.m3es / s.m3en if s.m3en else np.nan,
            "M3_err_rate": s.m3en / s.m3n if s.m3n else np.nan,
            "M4_spike": s.m4sn / s.m4sd if s.m4sd else np.nan, "M4_level": s.m4ln / s.m4ld if s.m4ld else np.nan,
            "n_M1": int(s.m1n), "n_units": int(s.m2n), "n_M4_spike": int(s.m4sd), "n_M4_level": int(s.m4ld)}


point = {m: pooled(df[df.method == m]) for m in METHODS}
rng = np.random.default_rng(0)
origins = df.origin.unique()
by = {m: df[df.method == m].set_index("origin") for m in METHODS}
B = 2000
boot = {m: [] for m in METHODS}
for b in range(B):
    samp = rng.choice(origins, size=len(origins), replace=True)
    for m in METHODS:
        boot[m].append(pooled(by[m].loc[samp]))
ci = {}
for ai_m in ["screen", "ai"]:
    for m in ["baseline", "recent4w", "lastweek"]:
        key = f"{ai_m}_minus_{m}"
        ci[key] = {}
        for met in ["M1", "M2", "M3_all", "M3_err", "M4_spike", "M4_level"]:
            diffs = np.array([boot[ai_m][b][met] - boot[m][b][met] for b in range(B)], dtype=float)
            diffs = diffs[~np.isnan(diffs)]
            ci[key][met] = {"diff": round(point[ai_m][met] - point[m][met], 4),
                            "lo": round(float(np.percentile(diffs, 2.5)), 4), "hi": round(float(np.percentile(diffs, 97.5)), 4)}

# 판정(사전 규칙): 판정 AI = screen. M1(높을수록 좋음) 또는 M4_spike(낮을수록 좋음)에서 단순 방법 3개 각각보다 유의
sig_m1 = all(ci[f"screen_minus_{m}"]["M1"]["lo"] > 0 for m in ["baseline", "recent4w", "lastweek"])
sig_m4 = all(ci[f"screen_minus_{m}"]["M4_spike"]["hi"] < 0 for m in ["baseline", "recent4w", "lastweek"])
best = sorted(METHODS, key=lambda m: (-point[m]["M1"], point[m]["M4_spike"]))[0]
place = {}
for m in METHODS:
    for p_, g in pf[pf.method == m].groupby("place"):
        s = g.sum(numeric_only=True)
        place.setdefault(p_, {})[m] = {"M1": round(s.m1h / s.m1n, 4) if s.m1n else None, "M4_spike": round(s.m4sn / s.m4sd, 4) if s.m4sd else None,
                                       "n_M1": int(s.m1n), "n_M4_spike": int(s.m4sd)}
per_origin_m1 = df.pivot(index="origin", columns="method", values="m1h") / df.pivot(index="origin", columns="method", values="m1n")
res = {"origins": [str(o.date()) for o in ORIGINS], "labels": LABEL, "point": {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in point[m].items()} for m in METHODS},
       "ci95_bootstrap_origins": ci, "bootstrap": {"B": B, "seed": 0, "unit": "기준일"},
       "verdict": {"significant_M1": sig_m1, "significant_M4_spike": sig_m4, "ai_claim_supported": bool(sig_m1 or sig_m4),
                   "best_method_by_rule": best},
       "by_place": place, "per_origin_M1": per_origin_m1.round(4).to_dict(orient="index")}
json.dump(res, open(f"{OUT}/backtest_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res["point"], ensure_ascii=False, indent=1))
print(json.dumps(res["verdict"], ensure_ascii=False))
