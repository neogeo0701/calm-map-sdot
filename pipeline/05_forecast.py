"""채택 모델(try1)을 전체 기간으로 다시 학습해 서비스 기간(2026-10-28~11-03) 시간별 예측을 만들고,
metrics.json·그래프·샘플 CSV·public/predictions.json을 쓴다."""
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

CHOSEN = "try1"
sys.argv = ["04_model.py", CHOSEN]
spec = importlib.util.spec_from_file_location("m", "pipeline/04_model.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

d, t1, sel, sites = M.d, M.t1, M.sel, M.sites
attempts = {a: json.load(open(f"model/attempts/{a}/metrics.json", encoding="utf-8")) for a in ["base", "try1", "try2"]}
chosen = attempts[CHOSEN]
h_lo, h_hi = chosen["service_horizon_hours"]
svc_key = f"service_{h_lo}-{h_hi}h"


def confidence(m):
    if m is None:
        return "정보 부족"
    return "높음" if m < 1.5 else ("보통" if m <= 3.0 else "낮음")


# --- 서비스 예측: 전체 실측(~t1)으로 재학습, t1 다음 시각부터 재귀 예측
start = t1 + pd.Timedelta(hours=1)
dw = M.weather_proxy(d, start)
svc = d[["serial", "measured_at", "dow", "hour"]].copy()
importances = {}
for t in M.TARGETS:
    M.add_lags(d, t)
    model = M.fit(d, t, start)
    dw[[c for c in d.columns if c.startswith(f"{t}_")]] = d[[c for c in d.columns if c.startswith(f"{t}_")]]
    svc[f"{t}_ai"] = M.recursive(dw, t, model, start)
    svc[f"{t}_baseline"], svc[f"{t}_baseline_n"] = M.baseline(d, t, start)
    importances[t] = dict(zip(M.FEATS[t], model.booster_.feature_importance("gain").tolist()))
    # 입력 정보 부족: 1·2주 전 값과 주간 평균의 근거가 되는 실측이 모두 없음
    arr = d[t].to_numpy().reshape(M.S, M.N)
    obs = ~np.isnan(arr)
    k = M.idx(start)
    recent_cov = obs[:, k - 336:k].mean(axis=1)  # 직전 2주 실측 비율
    svc[f"{t}_recent_coverage"] = np.repeat(recent_cov, M.N)

svc = svc[(svc.measured_at >= M.SERVICE_START) & (svc.measured_at <= M.SERVICE_END) & svc.serial.isin(sel)]

# --- metrics.json (채택 모델 + 시도 이력)
metrics = {
    "chosen_attempt": CHOSEN,
    "note": "MAE 단위 dB. baseline=센서별 요일x시간 평균(학습 구간). ai_recursive=서비스와 같은 재귀 예측. "
            "서비스 화면 값은 service 구간(데이터 마지막 시각으로부터 217~384시간 뒤) 조건과 같다. "
            "세 설정 중 채택에 테스트 구간을 사용했으므로 채택 모델 MAE는 약간 낙관적일 수 있다.",
    "attempts_summary": {a: {t: {"overall": m["targets"][t]["overall"], "service_horizon": m["targets"][t]["by_horizon_recursive"][svc_key]}
                             for t in M.TARGETS} for a, m in attempts.items()},
    **chosen,
}
json.dump(metrics, open("model/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# --- 피처 중요도 그래프 (전체 기간 재학습 모델, gain 비율)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = {"dow": "요일", "hour": "시간", "weekend": "주말", "holiday": "공휴일", "sensor": "센서 ID", "temp": "기온", "humid": "습도"}
for ax, (t, imp) in zip(axes, importances.items()):
    tot = sum(imp.values())
    items = sorted(imp.items(), key=lambda x: x[1])
    lab = [names.get(k, k.replace(f"{t}_lag168", "1주 전 같은 시각").replace(f"{t}_lag336", "2주 전 같은 시각").replace(f"{t}_roll168", "1주 전 주간 평균")) for k, _ in items]
    ax.barh(lab, [100 * v / tot for _, v in items], color="#3b6ea5")
    ax.set_title(M.TARGETS[t], fontsize=11)
    ax.set_xlabel("중요도 (gain, %)")
fig.suptitle("LightGBM 피처 중요도 (송파구 5개 센서, 2026-07-13~10-18 실측 학습)", fontsize=12)
fig.tight_layout()
fig.savefig("model/feature_importance.png", dpi=150)

# --- 테스트 기간 실측 vs 예측 그래프
tp = pd.read_csv(f"model/attempts/{CHOSEN}/test_predictions.csv", parse_dates=["measured_at"])
site_name = {s["sensor_serial"]: s["name"] for s in sites}
for t in M.TARGETS:
    fig, axes = plt.subplots(len(sel), 1, figsize=(13, 2.3 * len(sel)), sharex=True)
    for ax, s in zip(axes, sel):
        q = tp[tp.serial == s]
        ax.plot(q.measured_at, q[f"{t}_actual"], color="#222", lw=0.9, label="실측")
        ax.plot(q.measured_at, q[f"{t}_ai_recursive"], color="#d9822b", lw=1.1, label="AI 예측(재귀)")
        ax.plot(q.measured_at, q[f"{t}_baseline"], color="#7aa6c2", lw=0.9, ls="--", label="기준(요일x시간 평균)")
        bs = chosen["targets"][t]["by_sensor"][s]
        ax.set_title(f"{site_name[s]} ({s})  MAE: AI {bs['ai_recursive_mae']:.2f} / 기준 {bs['baseline_mae']:.2f} dB", fontsize=9.5, loc="left")
        ax.set_ylabel("dB")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    fig.suptitle(f"테스트 기간 실측 vs 예측 — {M.TARGETS[t]} (실측 없는 시간은 빈칸)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"model/test_actual_vs_pred_{t}.png", dpi=130)

# --- 샘플 예측 CSV (선택 제출물)
out = svc.copy()
out.insert(1, "site", out.serial.map(site_name))
out.round(2).to_csv("model/sample_predictions_2026-10-28_11-03.csv", index=False, encoding="utf-8-sig")

# --- 최근 실측 (카드용)
raw = pd.read_pickle("data/processed/sdot_hourly_all.pkl")
pub_sites = []
for s in sites:
    r = raw[(raw.serial == s["sensor_serial"]) & raw.noise_avg.notna()].sort_values("measured_at")
    last = r.iloc[-1]
    q = svc[svc.serial == s["sensor_serial"]].sort_values("measured_at")
    tm = {t: chosen["targets"][t]["by_sensor"][s["sensor_serial"]] for t in M.TARGETS}
    fc = []
    for _, row in q.iterrows():
        e = {"t": row.measured_at.strftime("%Y-%m-%dT%H:00")}
        for t in M.TARGETS:
            ok = row[f"{t}_recent_coverage"] >= 0.5 and not np.isnan(row[f"{t}_ai"])
            e[t] = round(float(row[f"{t}_ai"]), 1) if ok else None
            e[f"{t}_baseline"] = round(float(row[f"{t}_baseline"]), 1) if row[f"{t}_baseline_n"] >= 3 else None
        fc.append(e)
    pub_sites.append({
        **{k: s[k] for k in ["site_id", "name", "kind", "osm_id", "place_lat", "place_lon", "sensor_serial",
                             "sensor_address", "sensor_lat", "sensor_lon", "dist_m"]},
        "recent": {"t": last.measured_at.strftime("%Y-%m-%dT%H:00"), "level": float(last.noise_avg),
                   "max": float(last.noise_max), "min": float(last.noise_min),
                   "spike": round(float(last.noise_max - last.noise_avg), 1)},
        "recent_week": {t: round(float((r.set_index("measured_at").loc[t1 - pd.Timedelta(hours=167):t1].pipe(
            lambda x: x.noise_avg if t == "level" else x.noise_max - x.noise_avg)).mean()), 1) for t in M.TARGETS},
        "mae": {t: {"ai": tm[t]["service_horizon_ai_mae"], "baseline": tm[t]["service_horizon_baseline_mae"]} for t in M.TARGETS},
        "confidence": {t: confidence(tm[t]["service_horizon_ai_mae"]) for t in M.TARGETS},
        "forecast": fc,
    })

pub = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M"),
    "data_source": "서울특별시 도시데이터 센서(S-DoT) 환경정보(서울 열린데이터광장 OA-15969)",
    "data_period": [str(M.t0)[:16], str(t1)[:16]],
    "forecast_period": [str(M.SERVICE_START)[:16], str(M.SERVICE_END)[:16]],
    "train_period": [p[:16] for p in chosen["train_period"]],
    "test_period": [p[:16] for p in chosen["test_period"]],
    "horizon_hours": [h_lo, h_hi],
    "model": {"name": "LightGBM 회귀 (재귀 예측)", "attempt": CHOSEN,
              "service_mae": {t: chosen["targets"][t]["by_horizon_recursive"][svc_key] for t in M.TARGETS}},
    "sites": pub_sites,
}
json.dump(pub, open("public/predictions.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(json.dumps({s["name"]: {"recent": s["recent"], "mae": s["mae"], "conf": s["confidence"]} for s in pub_sites}, ensure_ascii=False, indent=1))
print(svc.groupby("serial")[["level_ai", "level_baseline", "spike_ai", "spike_baseline"]].mean().round(2))
