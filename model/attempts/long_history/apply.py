"""장기 이력 S1 모델을 돌발 소음 지수에만 적용한다(DECISIONS.md 교체 판정).
전체 확보 기간(2024-01 ~ 2026-10-18)으로 다시 학습해 2026-10-28 ~ 11-03 시간별 돌발 지수를 예측하고,
public/predictions.json의 돌발 지수 부분(값·같은 시각 요일×시간 평균·오차·신뢰도·성능표)만 바꾼다. 소음 수준은 그대로.
사용법(저장소 루트에서): python model/attempts/long_history/apply.py"""
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

HERE = "model/attempts/long_history"
sys.argv = ["run.py", "none"]
spec = importlib.util.spec_from_file_location("lh", f"{HERE}/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
d, sel, t1 = R.d, R.sel, R.t1
val = json.load(open(f"{HERE}/val_metrics.json", encoding="utf-8"))
test = json.load(open(f"{HERE}/test_metrics.json", encoding="utf-8"))
NAME = test["chosen_by_rule"]["spike"][3:]  # S1
cfg = R.CONFIGS[NAME]
T = test["results"]["spike"]

# --- 서비스 예측
start = t1 + pd.Timedelta(hours=1)
R.add_lags(d, cfg, "spike")
dw = R.weather_proxy(d, start)
for c in [c for c in d.columns if c.startswith("spike_lag") or c.startswith("spike_roll")]:
    dw[c] = d[c]
tr = d[(d.measured_at < start) & d.spike.notna()]
model = R.lgb.LGBMRegressor(**R.PARAMS).fit(tr[R.feats(cfg, "spike")], tr.spike, categorical_feature=["sensor"])
pred = R.recursive(dw, cfg, "spike", model, start)
g = tr.groupby(["serial", "dow", "hour"]).spike.agg(["mean", "count"])
j = d.join(g, on=["serial", "dow", "hour"])
arr = d.spike.to_numpy().reshape(R.S, R.N)
k = R.idx(start)
cov = np.repeat((~np.isnan(arr[:, k - 336:k])).mean(axis=1), R.N)
svc = pd.DataFrame({"serial": d.serial, "measured_at": d.measured_at, "spike_value": pred,
                    "spike_baseline": j["mean"].to_numpy(), "spike_baseline_n": j["count"].fillna(0).to_numpy(), "cov": cov})
svc = svc[(svc.measured_at >= pd.Timestamp("2026-10-28")) & (svc.measured_at <= R.SERVICE_END)]
svc.round(3).to_csv(f"{HERE}/service_spike_2026-10-28_11-03.csv", index=False, encoding="utf-8-sig")

# --- 그래프: 피처 중요도, 테스트 실측 vs 예측
names = {"dow": "요일", "hour": "시간", "weekend": "주말", "holiday": "공휴일", "sensor": "센서 ID", "temp": "기온", "humid": "습도",
         "month": "월", "week": "주차", "spike_yoy": "전년 동기 값", "spike_yoy_week": "전년 동기 주간 평균",
         "spike_lag168": "1주 전 같은 시각", "spike_lag336": "2주 전 같은 시각", "spike_roll168": "1주 전 주간 평균"}
imp = dict(zip(R.feats(cfg, "spike"), model.booster_.feature_importance("gain")))
items = sorted(imp.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh([names.get(a, a) for a, _ in items], [100 * v / sum(imp.values()) for _, v in items], color="#3b6ea5")
ax.set_xlabel("중요도 (gain, %)")
ax.set_title("돌발 소음 지수 AI(LightGBM, 계절성 피처) 피처 중요도\n송파구 5개 센서, 2024-01 ~ 2026-10-18 실측 학습", fontsize=11)
fig.tight_layout()
fig.savefig(f"{HERE}/feature_importance_spike.png", dpi=150)

tp = pd.read_csv(f"{HERE}/test_predictions.csv", parse_dates=["measured_at"])
site_name = {s["sensor_serial"]: s["name"] for s in R.sites}
fig, axes = plt.subplots(len(sel), 1, figsize=(13, 2.3 * len(sel)), sharex=True)
for ax, s in zip(axes, sel):
    q = tp[tp.serial == s]
    b = T["by_sensor"][s]
    ax.plot(q.measured_at, q.spike_actual, color="#222", lw=0.9, label="실측")
    ax.plot(q.measured_at, q[f"spike_ai_{NAME}"], color="#d9822b", lw=1.1, label="AI 예측(서비스 사용)")
    ax.plot(q.measured_at, q.spike_baseline, color="#7aa6c2", lw=0.9, ls="--", label="요일×시간 평균(2024~)")
    ax.set_title(f"{site_name[s]}  서비스 거리 MAE: AI {b[f'ai_{NAME}']:.2f} / 요일×시간 평균 {b['baseline']:.2f} / 지난주 그대로 {b['ref_last_week_naive']:.2f} dB", fontsize=9.5, loc="left")
    ax.set_ylabel("dB")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
fig.suptitle("테스트 기간(1회 평가) 돌발 소음 지수 — 장기 이력 계절성 모델 (실측 없는 시간은 빈칸)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{HERE}/test_actual_vs_pred_spike.png", dpi=130)
# 보고서·화면이 쓰는 그래프 위치에 사본(기존 3분할 그래프는 model/attempts/에 결과 파일로 남아 있음)
import shutil
shutil.copy(f"{HERE}/test_actual_vs_pred_spike.png", "model/test_actual_vs_pred_spike.png")
shutil.copy(f"{HERE}/feature_importance_spike.png", "model/feature_importance.png")


def confidence(m):
    return "정보 부족" if m is None else ("높음" if m < 1.5 else ("보통" if m <= 3.0 else "낮음"))


# --- public/predictions.json: 돌발 지수 부분만 교체
P = json.load(open("public/predictions.json", encoding="utf-8"))
for site in P["sites"]:
    ser = site["sensor_serial"]
    q = svc[svc.serial == ser].set_index("measured_at")
    for f in site["forecast"]:
        r = q.loc[pd.Timestamp(f["t"])]
        f["spike"] = round(float(r.spike_value), 1) if (r["cov"] >= 0.5 and not np.isnan(r.spike_value)) else None
        f["spike_baseline"] = round(float(r.spike_baseline), 1) if r.spike_baseline_n >= 3 else None
    b = T["by_sensor"][ser]
    site["mae"]["spike"] = {"used": b[f"ai_{NAME}"], "baseline": b["baseline"]}
    site["confidence"]["spike"] = confidence(b[f"ai_{NAME}"])
sm = T["service_horizon_mae"]
P["test"]["spike"] = {"n": T["n_service_hours"], "ai": sm[f"ai_{NAME}"], "baseline": sm["baseline"], "ref_recent4w_mean": sm["ref_recent4w_mean"],
                      "ref_last_week_naive": sm["ref_last_week_naive"], "ref_last_year_same": sm["ref_last_year_same"]}
P["test"]["level"]["ai"] = P["test"]["level"].pop("ai_try2", P["test"]["level"].get("ai"))
P["validation"]["spike"] = val["results"]["spike"]["service_horizon_mae"]
P["methods"]["spike"] = f"AI LightGBM 장기 이력 {NAME}(계절성 피처, 2024-01~ 학습) 재귀 예측"
P["sources"] = {"spike": "model/attempts/long_history/test_metrics.json", "level": "model/metrics.json"}
P["spike_history_start"] = str(R.t0)[:10]
P["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M")
json.dump(P, open("public/predictions.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("spike None:", sum(f["spike"] is None for s in P["sites"] for f in s["forecast"]))
print({s["name"]: (s["mae"]["spike"], s["confidence"]["spike"]) for s in P["sites"]})
print("top importance:", [names.get(a, a) for a, _ in items[::-1][:5]])
