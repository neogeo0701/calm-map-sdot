"""화면에 쓰는 검증 결과(고정값)를 public/predictions.json에 넣는다. 매일 갱신(daily_update.py)은 이 값을 그대로 둔다.
- mae_all: 장기 이력 테스트(2026-09-24~10-18, 1회 평가)의 장소별 서비스 거리 평균 오차, 방법별
- backtest: 순차 백테스트(BACKTEST_REPORT.md) 요약
사용법(저장소 루트): python pipeline/set_static_results.py"""
import json

P = json.load(open("public/predictions.json", encoding="utf-8"))
LH = json.load(open("model/attempts/long_history/test_metrics.json", encoding="utf-8"))["results"]
BT = json.load(open("model/attempts/backtest/backtest_results.json", encoding="utf-8"))
# 화면 표시용 점 추정은 반올림 전 값(기준일별 합계에서 다시 계산)으로
import pandas as pd
g = pd.read_csv("model/attempts/backtest/per_origin.csv").groupby("method").sum(numeric_only=True)
POINT = {m: {"M1": r.m1h / r.m1n, "M2": r.m2o / (3 * r.m2n), "M3_all": r.m3s / r.m3n, "M4_spike": r.m4sn / r.m4sd, "M4_level": r.m4ln / r.m4ld}
         for m, r in g.iterrows()}
KEY = {"ai": "ai_S1", "baseline": "baseline", "recent4w": "ref_recent4w_mean", "lastweek": "ref_last_week_naive"}
for site in P["sites"]:
    ser = site["sensor_serial"]
    site["mae_all"] = {t: {m: LH[t]["by_sensor"][ser][k] for m, k in KEY.items()} for t in ("level", "spike")}
P["backtest"] = {
    "origins": len(BT["origins"]), "period": [BT["origins"][0], BT["origins"][-1]],
    "point": POINT,
    "ci_screen_minus": {m: {k: BT["ci95_bootstrap_origins"][f"screen_minus_{m}"][k] for k in ("M1", "M4_spike")} for m in ("baseline", "recent4w", "lastweek")},
    "verdict": BT["verdict"],
}
json.dump(P, open("public/predictions.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("ok", P["backtest"]["verdict"])
