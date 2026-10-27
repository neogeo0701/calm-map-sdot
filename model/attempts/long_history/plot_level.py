"""성능 표 기준 통일(2024-01 이후 학습)에 맞춘 테스트 기간 소음 수준 그래프. 사용법(저장소 루트): python model/attempts/long_history/plot_level.py"""
import json
import shutil

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = "model/attempts/long_history"
T = json.load(open(f"{HERE}/test_metrics.json", encoding="utf-8"))["results"]["level"]["by_sensor"]
sites = json.load(open("data/processed/selected_sites.json", encoding="utf-8"))
tp = pd.read_csv(f"{HERE}/test_predictions.csv", parse_dates=["measured_at"])
fig, axes = plt.subplots(len(sites), 1, figsize=(13, 2.3 * len(sites)), sharex=True)
for ax, s in zip(axes, sites):
    q = tp[tp.serial == s["sensor_serial"]]
    b = T[s["sensor_serial"]]
    ax.plot(q.measured_at, q.level_actual, color="#222", lw=0.9, label="실측")
    ax.plot(q.measured_at, q.level_ref_last_week_naive, color="#2f6f5e", lw=1.1, label="지난주 그대로(서비스 사용)")
    ax.plot(q.measured_at, q.level_ai_S1, color="#d9822b", lw=0.9, alpha=0.8, label="AI(미사용)")
    ax.plot(q.measured_at, q.level_baseline, color="#7aa6c2", lw=0.9, ls="--", label="요일×시간 평균(2024~)")
    ax.set_title(f"{s['name']}  서비스 거리 MAE: 지난주 그대로 {b['ref_last_week_naive']:.2f} / AI {b['ai_S1']:.2f} / 요일×시간 평균 {b['baseline']:.2f} dB", fontsize=9.5, loc="left")
    ax.set_ylabel("dB")
axes[0].legend(loc="upper right", fontsize=8, ncol=4)
fig.suptitle("테스트 기간(1회 평가) 소음 수준 — 모든 방법 2024-01 이후 학습 기준 (실측 없는 시간은 빈칸)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{HERE}/test_actual_vs_pred_level.png", dpi=130)
shutil.copy(f"{HERE}/test_actual_vs_pred_level.png", "model/test_actual_vs_pred_level.png")
print("ok")
