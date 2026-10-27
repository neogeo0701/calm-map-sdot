"""데모 영상용 설명 화면(demo/slides.html)을 실제 데이터로 만든다. 코드 편집기 화면은 쓰지 않는다."""
import html
import json
import shutil

import pandas as pd

RAW = "data/raw/S-DoT_NATURE_2026.10.12-10.18.csv"
SERIAL = "V02Q1940736"  # 큰솔 어린이공원 센서
cols = ["시리얼", "측정시간", "자치구", "행정동", "온도 평균(℃)", "습도 평균(%)", "소음 최대(dB)", "소음 평균(dB)", "소음 최소(dB)"]
df = pd.read_csv(RAW, encoding="cp949", usecols=cols, dtype=str, index_col=False)
rows = df[(df["시리얼"] == SERIAL) & df["측정시간"].str.startswith("2026-10-18_1")][cols].head(9)
m = json.load(open("model/metrics.json", encoding="utf-8"))
T = {t: m["test"][t]["service_horizon_mae"] for t in ["level", "spike"]}
V = {t: m["validation"][t]["service_horizon_mae"] for t in ["level", "spike"]}
NT = m["test"]["spike"]["n"]
# 돌발 소음 지수는 장기 이력 실험 결과로 교체(DECISIONS.md)
LH = json.load(open("model/attempts/long_history/test_metrics.json", encoding="utf-8"))
_s = LH["results"]["spike"]["service_horizon_mae"]
T["spike"] = {"ai_try1": _s["ai_S1"], "baseline": _s["baseline"], "ref_recent4w_mean": _s["ref_recent4w_mean"], "ref_last_week_naive": _s["ref_last_week_naive"]}
shutil.copy("model/feature_importance.png", "demo/feature_importance.png")

# 영상용 넓은 그래프: 큰솔 어린이공원 센서 하나(테스트 기간 실측 vs 예측)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
tp = pd.read_csv("model/attempts/long_history/test_predictions.csv", parse_dates=["measured_at"]).rename(columns={"spike_ai_S1": "spike_ai_recursive"})
q = tp[tp.serial == SERIAL].set_index("measured_at")
q = q.loc["2026-10-05":"2026-10-18 23:00"]  # 마지막 2주만(영상에서 읽을 수 있게)
fig, ax = plt.subplots(figsize=(14, 5.2))
ax.plot(q.index, q.spike_actual, color="#222", lw=1.1, label="실측")
ax.plot(q.index, q.spike_ai_recursive, color="#d9822b", lw=1.6, label="AI 예측(재귀)")
ax.plot(q.index, q.spike_baseline, color="#7aa6c2", lw=1.1, ls="--", label="요일·시간 평균")
bs = LH["results"]["spike"]["by_sensor"][SERIAL]
ax.set_title(f"큰솔 어린이공원 센서 돌발 소음 지수(최대-평균, dB) · 테스트 MAE  AI {bs['ai_S1']:.2f} / 요일·시간 평균(2024~) {bs['baseline']:.2f}", fontsize=14, loc="left")
ax.set_ylabel("dB"); ax.legend(loc="upper right", ncol=3, fontsize=11); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("demo/test_actual_vs_pred_spike.png", dpi=130)

table = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>" + "".join(
    "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows.values) + "</table>"

page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>demo slides</title>
<style>
body{{margin:0;background:#f6f5f1;color:#1f2328;font-family:"Malgun Gothic",sans-serif}}
section{{width:1280px;height:720px;box-sizing:border-box;padding:40px 72px 110px;display:none;flex-direction:column;justify-content:center}}
section:target{{display:flex}}
h1{{font-size:46px;margin:0 0 18px}} h2{{font-size:34px;margin:0 0 16px}}
p,li{{font-size:24px;line-height:1.55}} .muted{{color:#5b6168;font-size:18px}}
.tag{{display:inline-block;font-size:16px;font-weight:700;border:2px solid #c4651c;color:#c4651c;border-radius:999px;padding:2px 12px;margin-bottom:14px}}
.obs{{border-color:#1f2328;color:#1f2328}}
table{{border-collapse:collapse;font-size:17px;background:#fff}} th,td{{border:1px solid #dcd9d1;padding:6px 10px;text-align:right}} th{{background:#eeece6}}
td:nth-child(n+7){{font-weight:700}}
img{{max-width:100%;max-height:470px;object-fit:contain;background:#fff;border:1px solid #dcd9d1}}
.big{{font-size:64px;font-weight:800;color:#2f6f5e}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:center}}
</style></head><body>
<section id="problem">
  <span class="tag">가상 이용 시나리오</span>
  <h1>"토요일 오후, 공원에 가도 괜찮을까?"</h1>
  <p>소리에 예민한 아이와 동네 공원에 가려는 보호자는 <b>언제 가야 덜 시끄러운지</b> 미리 알 방법이 없습니다.<br>
  한 번 힘든 외출을 겪으면 다음 외출을 포기하게 됩니다.</p>
  <p class="muted">발달장애인 25만5,207명(2021, 보건복지부 발달장애인 실태조사) · 발달장애아동 325명 보호자 설문에서 감각처리 문제 확인(김진경·이택영, 대한작업치료학회지 2010)<br>
  기존 해외 서비스: PIUMA(당사자·보호자 제보), HushMap(이용자 제보, 부족분은 AI 추정), KultureCity(시설 인증)</p>
</section>
<section id="idea">
  <h1>무던한 지도 — "갑작스러운 소리"가 적은 시간을 예측</h1>
  <p>평균 소음만으로는 갑작스러운 소리를 알 수 없습니다. 시간 최대 − 평균을 <b>돌발 소음 지수</b>로 정하고,<br>AI(LightGBM)가 송파구 공원·놀이터 5곳의 다음 주 시간별 돌발 지수를 예측합니다.<br>
  AI가 단순 방법보다 나은지 학습·검증·테스트로 나눠 확인하고, 보호자 조건에 맞는 방문 시간 3개를 근거와 함께 추천합니다.</p>
  <p class="muted">2026 AI 라이프 아이디어 챌린지 수상 후보작 「무던한 지도(Calm Map)」 기반 구현</p>
</section>
<section id="raw">
  <span class="tag obs">실측 원본</span>
  <h2>S-DoT 환경정보 원본 CSV — 큰솔 어린이공원 옆 센서 ({SERIAL})</h2>
  {table}
  <p class="muted">서울 열린데이터광장 OA-15969 「S-DoT_NATURE_2026.10.12-10.18.csv」의 실제 행(일부 열). 14주 · 200만 행에서 공원 경계 30m 이내 센서 5대를 골랐습니다.<br>
  소음 수준 = 시간 평균 · 돌발 소음 지수 = 시간 최대 − 시간 평균 · 결측은 보간하지 않음</p>
</section>
<section id="test">
  <span class="tag">AI 예측</span>
  <h2>테스트 마지막 2주: 돌발 소음 지수 실측(검정) vs AI(주황) vs 요일·시간 평균(파랑 점선)</h2>
  <img src="test_actual_vs_pred_spike.png">
</section>
<section id="mae">
  <h2>테스트(마지막 1회 평가) · 화면과 같은 9~16일 앞 예측의 오차 MAE</h2>
  <div class="grid">
    <div><p>돌발 소음 지수 — <b>AI 사용</b></p><div class="big">{T['spike']['ai_try1']:.2f} dB</div>
      <p>요일·시간 평균 {T['spike']['baseline']:.2f} · 직전 4주 {T['spike']['ref_recent4w_mean']:.2f} · 지난주 그대로 {T['spike']['ref_last_week_naive']:.2f} dB<br>
      <b>지난주 그대로·직전 4주보다 작고, 긴 기간(2024~) 요일·시간 평균과는 거의 같음</b> <span class="muted">(약 34개월 실측 + 계절 정보로 학습)</span></p></div>
    <div><p>소음 수준 — <b>AI 미사용</b></p><div class="big" style="color:#5b6168">{T['level']['ref_last_week_naive']:.2f} dB</div>
      <p>"지난주 그대로" {T['level']['ref_last_week_naive']:.2f} vs AI {T['level']['ai_try2']:.2f} dB<br>
      <b>AI가 단순 방법을 넘지 못함</b> → 화면은 지난주 같은 시각 실측값 사용</p></div>
  </div>
  <p class="muted">시간순 3분할(검증 2.1주 · 테스트 3.5주, 학습은 그 이전). 방법·교체 규칙은 테스트 전에 정하고 테스트는 한 번만 평가 · 5개 센서 · 실측 {NT}시간</p>
</section>
<section id="importance">
  <h2>돌발 소음 지수 AI가 주로 본 것: 1주 전 같은 시각, 1주 전 주간 평균, 시간대 (계절 정보 약 12%)</h2>
  <img src="feature_importance.png">
</section>
<section id="effect">
  <h1>기대효과와 다음 단계</h1>
  <ul>
    <li>보호자가 <b>시간</b>을 골라 갑작스러운 소리로 인한 외출 실패 가능성을 줄이는 근거를 얻음 — 장소에 낙인을 찍지 않음</li>
    <li>AI가 이긴 부분만 AI로, 진 부분은 단순 방법으로 — 검증 결과를 그대로 공개</li>
    <li>서울 S-DoT 센서 1,100여 대 · 공원 경계 100m 이내 이름 있는 공원·놀이터 306곳으로 같은 방법을 넓힐 수 있음</li>
    <li>다음 단계: Open API로 최신 데이터 반영(예측 거리 단축), 보호자 실제 방문 피드백으로 검증, 소음 수준 AI 개선</li>
  </ul>
  <p class="muted">이 결과는 S-DoT 데이터를 활용한 개인 연구 결과이며 서울시의 공식 입장이 아닙니다. 진단이나 보장이 아닙니다.</p>
</section>
</body></html>"""
open("demo/slides.html", "w", encoding="utf-8").write(page)
print("ok")
