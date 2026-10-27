"""검증 구간에서 고른 방법으로 서비스 기간(2026-10-28~11-03) 시간별 값을 만들고,
metrics.json·그래프·샘플 CSV·public/predictions.json을 쓴다.

선택(테스트 전에 확정, DECISIONS.md): 대상별로 검증 구간 서비스 거리 MAE가 가장 낮은 후보.
  소음 수준      → 지난주 그대로(직전 7일 실측 반복)
  돌발 소음 지수 → AI(LightGBM, try1 설정)
"""
import importlib.util
import json
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

AI_ATTEMPTS = ["base", "try1", "try2"]
REF = {"baseline": "요일×시간 평균", "ref_recent4w_mean": "직전 4주 요일×시간 평균", "ref_last_week_naive": "지난주 그대로"}
SPIKE_AI = "try1"

sys.argv = ["04_model.py", SPIKE_AI, "test"]
spec = importlib.util.spec_from_file_location("m", "pipeline/04_model.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
d, t1, sel, sites = M.d, M.t1, M.sel, M.sites
site_name = {s["sensor_serial"]: s["name"] for s in sites}

val = {a: json.load(open(f"model/attempts/{a}_val/metrics.json", encoding="utf-8")) for a in AI_ATTEMPTS}
h_lo, h_hi = val["base"]["service_horizon_hours"]
svc_key = f"service_{h_lo}-{h_hi}h"

# --- 1) 검증 결과 표와 선택 (선택 규칙은 테스트 전에 확정)
validation, selected = {}, {}
for t in M.TARGETS:
    row = {f"ai_{a}": val[a]["targets"][t]["by_horizon_recursive"][svc_key]["ai_recursive_mae"] for a in AI_ATTEMPTS}
    b = val["base"]["targets"][t]["by_horizon_recursive"][svc_key]
    row.update({"baseline": b["baseline_mae"], "ref_recent4w_mean": b["ref_recent4w_mean_mae"], "ref_last_week_naive": b["ref_last_week_naive_mae"]})
    validation[t] = {"service_horizon_mae": row, "hours_covered": b["hours_covered"], "n": b["n"]}
    selected[t] = min(row, key=row.get)
best_ai = {t: min((k for k in validation[t]["service_horizon_mae"] if k.startswith("ai_")), key=validation[t]["service_horizon_mae"].get) for t in M.TARGETS}
assert selected == {"level": "ref_last_week_naive", "spike": "ai_try1"}, selected

# --- 2) 테스트 결과(대상별 1회 평가: level=try2, spike=try1)
test_runs = {"level": json.load(open(f"model/attempts/{best_ai['level'][3:]}_test/metrics.json", encoding="utf-8")),
             "spike": json.load(open(f"model/attempts/{best_ai['spike'][3:]}_test/metrics.json", encoding="utf-8"))}
t_test = pd.Timestamp(test_runs["spike"]["eval_period"][0])
horizon = ((d.measured_at - t_test) / pd.Timedelta(hours=1)).to_numpy() + 1
in_test = ((d.measured_at >= t_test) & (d.measured_at <= t1) & d.serial.isin(sel)).to_numpy()
in_svc = in_test & (horizon >= h_lo) & (horizon <= h_hi)

test = {}
for t in M.TARGETS:
    r = test_runs[t]["targets"][t]
    sv = r["by_horizon_recursive"][svc_key]
    y = d[t].to_numpy()
    lw = M.last_week_naive(d, t, t_test)
    base, _ = M.baseline(d, t, t_test)
    by_sensor = {}
    for s in sel:
        sm = in_svc & (d.serial == s).to_numpy()
        by_sensor[s] = {"ai": r["by_sensor"][s]["service_horizon_ai_mae"], "baseline": M.mae(y[sm], base[sm]),
                        "ref_last_week_naive": M.mae(y[sm], lw[sm])}
    test[t] = {
        "ai_attempt": best_ai[t][3:],
        "service_horizon_mae": {best_ai[t]: sv["ai_recursive_mae"], "baseline": sv["baseline_mae"],
                                "ref_recent4w_mean": sv["ref_recent4w_mean_mae"], "ref_last_week_naive": sv["ref_last_week_naive_mae"]},
        "n": sv["n"], "hours_covered": sv["hours_covered"],
        "overall_mae": {best_ai[t] + "_recursive": r["overall"]["ai_recursive_mae"], best_ai[t] + "_1h": r["overall"]["ai_1h_mae"],
                        "baseline": r["overall"]["baseline_mae"], "ref_recent4w_mean": r["overall"]["ref_recent4w_mean_mae"],
                        "ref_last_week_naive": r["overall"]["ref_last_week_naive_mae"]},
        "by_horizon": r["by_horizon_recursive"],
        "by_sensor_service_horizon": by_sensor,
        "feature_importance_gain": r["feature_importance_gain"],
    }
    test[t]["selected_mae"] = test[t]["service_horizon_mae"][selected[t] if selected[t] in test[t]["service_horizon_mae"] else best_ai[t]]

# --- 3) 서비스 값: 전체 실측(~t1) 기준
start = t1 + pd.Timedelta(hours=1)
svc = d[["serial", "measured_at", "dow", "hour"]].copy()
# 소음 수준: 지난주 그대로
svc["level_value"] = M.last_week_naive(d, "level", start)
# 돌발 소음 지수: AI(try1) 전체 기간 재학습 후 재귀 예측
dw = M.weather_proxy(d, start)
M.add_lags(d, "spike")
lag_cols = [c for c in d.columns if c.startswith("spike_")]
dw[lag_cols] = d[lag_cols]
spike_model = M.fit(d, "spike", start)
svc["spike_value"] = M.recursive(dw, "spike", spike_model, start)
importance = dict(zip(M.FEATS["spike"], spike_model.booster_.feature_importance("gain").tolist()))
for t in M.TARGETS:
    svc[f"{t}_baseline"], svc[f"{t}_baseline_n"] = M.baseline(d, t, start)
arr = d["spike"].to_numpy().reshape(M.S, M.N)
k = M.idx(start)
svc["spike_recent_coverage"] = np.repeat((~np.isnan(arr[:, k - 336:k])).mean(axis=1), M.N)
svc = svc[(svc.measured_at >= M.SERVICE_START) & (svc.measured_at <= M.SERVICE_END) & svc.serial.isin(sel)]

# --- 4) metrics.json
metrics = {
    "note": "MAE 단위 dB. 서비스 거리 = 데이터 마지막 시각으로부터 217~384시간 뒤(화면 예측과 같은 조건). "
            "시간순 3분할. 선택은 검증 구간으로만, 테스트는 대상별 1회 평가. 검증 구간 길이 한계로 검증은 217~342시간만 포함.",
    "split": test_runs["spike"]["split"],
    "service_period": [str(M.SERVICE_START), str(M.SERVICE_END)],
    "service_horizon_hours": [h_lo, h_hi],
    "candidates": {**{f"ai_{a}": f"LightGBM {a}" for a in AI_ATTEMPTS}, **REF},
    "ai_configs": {a: val[a]["config"] for a in AI_ATTEMPTS},
    "selection_rule": "대상별 검증 구간 서비스 거리 MAE 최저 후보(AI 3설정 + 기준 3개)",
    "validation": validation,
    "selected": selected,
    "best_ai_on_validation": best_ai,
    "test": test,
    "service_method": {"level": "지난주 그대로(직전 7일 실측 반복)", "spike": f"AI LightGBM {SPIKE_AI} 재귀 예측"},
}
json.dump(metrics, open("model/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# --- 5) 그래프
names = {"dow": "요일", "hour": "시간", "weekend": "주말", "holiday": "공휴일", "sensor": "센서 ID", "temp": "기온", "humid": "습도",
         "spike_lag168": "1주 전 같은 시각", "spike_lag336": "2주 전 같은 시각", "spike_roll168": "1주 전 주간 평균"}
fig, ax = plt.subplots(figsize=(8, 4.2))
tot = sum(importance.values())
items = sorted(importance.items(), key=lambda x: x[1])
ax.barh([names.get(k, k) for k, _ in items], [100 * v / tot for _, v in items], color="#3b6ea5")
ax.set_xlabel("중요도 (gain, %)")
ax.set_title("돌발 소음 지수 AI(LightGBM) 피처 중요도\n송파구 5개 센서, 2026-07-13~10-18 실측 학습", fontsize=11)
fig.tight_layout()
fig.savefig("model/feature_importance.png", dpi=150)

tp = {t: pd.read_csv(f"model/attempts/{best_ai[t][3:]}_test/test_predictions.csv", parse_dates=["measured_at"]) for t in M.TARGETS}
lw_level = pd.DataFrame({"serial": d.serial, "measured_at": d.measured_at, "lw": M.last_week_naive(d, "level", t_test)})
for t in M.TARGETS:
    q0 = tp[t]
    if t == "level":
        q0 = q0.merge(lw_level, on=["serial", "measured_at"], how="left")
    fig, axes = plt.subplots(len(sel), 1, figsize=(13, 2.3 * len(sel)), sharex=True)
    for ax, s in zip(axes, sel):
        q = q0[q0.serial == s]
        bs = test[t]["by_sensor_service_horizon"][s]
        ax.plot(q.measured_at, q[f"{t}_actual"], color="#222", lw=0.9, label="실측")
        if t == "level":
            ax.plot(q.measured_at, q["lw"], color="#2f6f5e", lw=1.1, label="지난주 그대로(서비스 사용)")
            ax.plot(q.measured_at, q[f"{t}_ai_recursive"], color="#d9822b", lw=0.9, alpha=0.8, label="AI(try2, 미사용)")
            title = f"{site_name[s]}  서비스 거리 MAE: 지난주 그대로 {bs['ref_last_week_naive']:.2f} / AI {bs['ai']:.2f} / 평균 {bs['baseline']:.2f} dB"
        else:
            ax.plot(q.measured_at, q[f"{t}_ai_recursive"], color="#d9822b", lw=1.1, label="AI 예측(서비스 사용)")
            title = f"{site_name[s]}  서비스 거리 MAE: AI {bs['ai']:.2f} / 평균 {bs['baseline']:.2f} / 지난주 그대로 {bs['ref_last_week_naive']:.2f} dB"
        ax.plot(q.measured_at, q[f"{t}_baseline"], color="#7aa6c2", lw=0.9, ls="--", label="요일×시간 평균")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylabel("dB")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    fig.suptitle(f"테스트 기간(1회 평가) 실측 vs 예측 — {M.TARGETS[t]} (실측 없는 시간은 빈칸)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"model/test_actual_vs_pred_{t}.png", dpi=130)

# --- 6) 샘플 CSV
out = svc.drop(columns=["spike_recent_coverage"]).copy()
out.insert(1, "site", out.serial.map(site_name))
out.round(2).to_csv("model/sample_predictions_2026-10-28_11-03.csv", index=False, encoding="utf-8-sig")


def confidence(m):
    if m is None:
        return "정보 부족"
    return "높음" if m < 1.5 else ("보통" if m <= 3.0 else "낮음")


# --- 7) public/predictions.json
raw = pd.read_pickle("data/processed/sdot_hourly_all.pkl")
pub_sites = []
for s in sites:
    ser = s["sensor_serial"]
    r = raw[(raw.serial == ser) & raw.noise_avg.notna()].sort_values("measured_at")
    last = r.iloc[-1]
    wk = r.set_index("measured_at").loc[t1 - pd.Timedelta(hours=167):t1]
    q = svc[svc.serial == ser].sort_values("measured_at")
    err = {"level": test["level"]["by_sensor_service_horizon"][ser]["ref_last_week_naive"],
           "spike": test["spike"]["by_sensor_service_horizon"][ser]["ai"]}
    fc = []
    for _, row in q.iterrows():
        e = {"t": row.measured_at.strftime("%Y-%m-%dT%H:00")}
        e["level"] = None if np.isnan(row.level_value) else round(float(row.level_value), 1)
        ok = row.spike_recent_coverage >= 0.5 and not np.isnan(row.spike_value)
        e["spike"] = round(float(row.spike_value), 1) if ok else None
        for t in M.TARGETS:
            e[f"{t}_baseline"] = round(float(row[f"{t}_baseline"]), 1) if row[f"{t}_baseline_n"] >= 3 else None
        fc.append(e)
    pub_sites.append({
        **{k: s[k] for k in ["site_id", "name", "kind", "osm_id", "place_lat", "place_lon", "sensor_serial",
                             "sensor_address", "sensor_lat", "sensor_lon", "dist_m"]},
        "recent": {"t": last.measured_at.strftime("%Y-%m-%dT%H:00"), "level": float(last.noise_avg),
                   "max": float(last.noise_max), "min": float(last.noise_min), "spike": round(float(last.noise_max - last.noise_avg), 1)},
        "recent_week": {"level": round(float(wk.noise_avg.mean()), 1), "spike": round(float((wk.noise_max - wk.noise_avg).mean()), 1)},
        "mae": {"level": {"used": err["level"], "baseline": test["level"]["by_sensor_service_horizon"][ser]["baseline"]},
                "spike": {"used": err["spike"], "baseline": test["spike"]["by_sensor_service_horizon"][ser]["baseline"]}},
        "confidence": {t: confidence(err[t]) for t in M.TARGETS},
        "forecast": fc,
    })

lw_week = [str(t1 - pd.Timedelta(hours=167))[:10], str(t1)[:10]]
pub = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M"),
    "data_source": "서울특별시 도시데이터 센서(S-DoT) 환경정보(서울 열린데이터광장 OA-15969)",
    "data_period": [str(M.t0)[:16], str(t1)[:16]],
    "forecast_period": [str(M.SERVICE_START)[:16], str(M.SERVICE_END)[:16]],
    "split": {k: [v[0][:16], v[1][:16]] for k, v in metrics["split"].items()},
    "horizon_hours": [h_lo, h_hi],
    "level_source_week": lw_week,
    "methods": metrics["service_method"],
    "test": {t: {"n": test[t]["n"], **test[t]["service_horizon_mae"]} for t in M.TARGETS},
    "validation": {t: validation[t]["service_horizon_mae"] for t in M.TARGETS},
    "sites": pub_sites,
}
json.dump(pub, open("public/predictions.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(json.dumps({"validation": validation, "selected": selected, "test": {t: test[t]["service_horizon_mae"] for t in test}}, ensure_ascii=False, indent=1))
print(json.dumps({s["name"]: [s["mae"], s["confidence"]] for s in pub_sites}, ensure_ascii=False))
print("level None:", sum(f["level"] is None for s in pub_sites for f in s["forecast"]), "spike None:", sum(f["spike"] is None for s in pub_sites for f in s["forecast"]))
