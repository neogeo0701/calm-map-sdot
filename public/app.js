// 무던한 지도 — 정적 웹앱. 예측값은 pipeline/05_forecast.py가 미리 계산한 predictions.json만 읽는다(서버·키 없음).
"use strict";

const WEIGHT = { low: 0.5, mid: 1, high: 2 };
const FLOOR_DB = 35; // 센서 측정 하한(학습 데이터 최솟값)
const DOW = ["일", "월", "화", "수", "목", "금", "토"];
const $ = (s) => document.querySelector(s);

let DATA = null;
let map = null;

// 부담 점수: 낮을수록 무던함
function score(level, spike, w) {
  return WEIGHT[w.level] * (level - FLOOR_DB) + WEIGHT[w.spike] * spike;
}

// 선택한 날짜·시간대 안에서 장소별 가장 낮은 시간 1개씩 → 점수 순 상위 3곳
function recommend(cond) {
  const [h0, h1] = cond.slot.split("-").map(Number);
  const out = [];
  for (const site of DATA.sites) {
    let best = null;
    for (const f of site.forecast) {
      if (!f.t.startsWith(cond.day)) continue;
      const h = Number(f.t.slice(11, 13));
      if (h < h0 || h >= h1) continue;
      if (f.level == null || f.spike == null) continue; // 정보 부족은 후보에서 제외
      const s = score(f.level, f.spike, cond);
      if (!best || s < best.score) best = { site, f, hour: h, score: s };
    }
    if (best) out.push(best);
  }
  out.sort((a, b) => a.score - b.score);
  return out.slice(0, 3);
}

function readCond() {
  const fd = new FormData($("#cond"));
  return { level: fd.get("level"), spike: fd.get("spike"), day: fd.get("day"), slot: fd.get("slot") };
}

const fmtDay = (d) => {
  const dt = new Date(d + "T00:00:00");
  return `${dt.getMonth() + 1}/${dt.getDate()}(${DOW[dt.getDay()]})`;
};
const fmtTime = (t) => `${fmtDay(t.slice(0, 10))} ${Number(t.slice(11, 13))}시`;
const n1 = (v) => (v == null ? null : v.toFixed(1));

function buildDays() {
  const days = [...new Set(DATA.sites[0].forecast.map((f) => f.t.slice(0, 10)))];
  const sat = days.find((d) => new Date(d + "T00:00:00").getDay() === 6) || days[0];
  $("#days").innerHTML = days
    .map((d) => `<label><input type="radio" name="day" value="${d}" ${d === sat ? "checked" : ""}>${fmtDay(d)}</label>`)
    .join("");
}

function fact(cls, k, v, s) {
  return `<div class="fact ${cls}"><span class="k">${k}</span><span class="v">${v}</span>${s ? `<span class="s">${s}</span>` : ""}</div>`;
}

function dayBars(site, cond, bestHour) {
  const [h0, h1] = cond.slot.split("-").map(Number);
  const rows = site.forecast.filter((f) => f.t.startsWith(cond.day));
  const vals = rows.map((f) => (f.level == null || f.spike == null ? null : score(f.level, f.spike, cond)));
  const ok = vals.filter((v) => v != null);
  const max = Math.max(...ok, 1);
  const bars = rows
    .map((f, i) => {
      const h = Number(f.t.slice(11, 13));
      if (vals[i] == null) return `<span class="na" title="${h}시 정보 부족"></span>`;
      const cls = h === bestHour ? "best" : h >= h0 && h < h1 ? "in" : "";
      const pct = Math.max(6, (100 * vals[i]) / max);
      return `<span class="${cls}" style="height:${pct}%" title="${h}시 부담 점수 ${vals[i].toFixed(1)}"></span>`;
    })
    .join("");
  return `<div class="dayline"><span class="k">${fmtDay(cond.day)} 하루 시간대별 부담 점수(예측, 낮을수록 무던함 · 진한 막대=고른 시간대, 테두리=추천 시각)</span>
    <div class="bars" aria-hidden="true">${bars}</div><div class="bars-x"><span>0시</span><span>6시</span><span>12시</span><span>18시</span></div></div>`;
}

function render() {
  const cond = readCond();
  const recs = recommend(cond);
  const lv = { low: "낮음", mid: "보통", high: "높음" };
  $("#r-sub").textContent = `${fmtDay(cond.day)} ${cond.slot.replace("-", "~")}시 · 소리 크기 민감도 ${lv[cond.level]} · 갑작스러운 소리 민감도 ${lv[cond.spike]}`;
  if (!recs.length) {
    $("#recs").innerHTML = `<li class="rec"><span class="tag na">정보 부족</span> 고른 시간대에 예측값이 있는 장소가 없습니다.</li>`;
    return;
  }
  $("#recs").innerHTML = recs
    .map((r, i) => {
      const s = r.site, f = r.f;
      const confL = s.confidence.level, confS = s.confidence.spike;
      return `<li class="rec">
        <div class="rec-head"><span class="rank">${i + 1}</span><span class="rec-title">${s.name}</span>
          <span class="rec-time">${fmtTime(f.t)}</span><span class="tag pred">예측</span></div>
        <div class="facts">
          ${fact("pred", "예측 소음 수준", `${n1(f.level)} dB`, `모델 오차(MAE) ±${s.mae.level.ai.toFixed(1)} dB · 신뢰도 <span class="tag conf-${confL}">${confL}</span>`)}
          ${fact("pred", "예측 돌발 소음 지수", `${n1(f.spike)} dB`, `모델 오차(MAE) ±${s.mae.spike.ai.toFixed(1)} dB · 신뢰도 <span class="tag conf-${confS}">${confS}</span>`)}
          ${fact("obs", "최근 실측 <span class='tag obs'>실측</span>", `평균 ${s.recent.level} · 최대 ${s.recent.max} dB`, `${fmtTime(s.recent.t)} 측정`)}
          ${fact("obs", "최근 1주 평균 <span class='tag obs'>실측</span>", `${n1(s.recent_week.level)} dB · 돌발 ${n1(s.recent_week.spike)} dB`, `${DATA.data_period[1].slice(5, 10).replace("-", "/")}까지 7일`)}
          ${fact("", "센서 위치", `공원 경계에서 ${Math.round(s.dist_m)}m`, `${s.sensor_address} (${s.sensor_serial})`)}
          ${fact("", "부담 점수", r.score.toFixed(1), `= ${WEIGHT[cond.level]}×(${n1(f.level)}−35) + ${WEIGHT[cond.spike]}×${n1(f.spike)}`)}
        </div>
        ${dayBars(s, cond, r.hour)}
        <p class="why">단순 요일·시간 평균으로는 같은 시각 ${f.level_baseline == null ? "<span class='tag na'>정보 부족</span>" : `${n1(f.level_baseline)} dB`}입니다.
        이 장소의 테스트 오차는 AI ${s.mae.level.ai.toFixed(2)} dB, 단순 평균 ${s.mae.level.baseline.toFixed(2)} dB였습니다.</p>
      </li>`;
    })
    .join("");
}

function renderPerf() {
  const m = DATA.model.service_mae;
  const row = (k, label) => {
    const x = m[k];
    return `<tr><td>${label}</td><td class="num better">${x.ai_recursive_mae.toFixed(2)}</td><td class="num">${x.baseline_mae.toFixed(2)}</td>
      <td class="num">${x.ref_last_week_naive_mae.toFixed(2)}</td><td class="num">${x.ai_improvement_pct.toFixed(1)}%</td></tr>`;
  };
  $("#perf").innerHTML = `
    <p class="muted">테스트 기간(${DATA.test_period[0].slice(0, 10)} ~ ${DATA.test_period[1].slice(0, 10)}) 5개 센서, 화면과 같은 조건(데이터 마지막 시각으로부터 ${DATA.horizon_hours[0]}~${DATA.horizon_hours[1]}시간 뒤 예측)의 평균 절대 오차(MAE, dB, 낮을수록 정확). 실측 ${m.level.n}시간 기준.</p>
    <table><thead><tr><th>대상</th><th>AI</th><th>요일·시간<br>평균</th><th>지난주<br>그대로</th><th>평균 대비<br>개선</th></tr></thead>
    <tbody>${row("level", "소음 수준")}${row("spike", "돌발 소음 지수")}</tbody></table>
    <p class="muted">학습: ${DATA.train_period[0].slice(0, 10)} ~ ${DATA.train_period[1].slice(0, 10)} 실측(약 10.5주) · 테스트: 그 뒤 약 3.5주 · 결측 시간은 보간하지 않고 평가에서 뺐습니다.
    7~8월 주간 평균 소음이 9월 이후보다 최대 15dB 높아 전체 기간 평균은 불리한 비교 대상이라 "지난주 그대로"도 함께 적었습니다.
    소음 수준에서 AI와 "지난주 그대로"의 차이는 ${Math.abs(m.level.ref_last_week_naive_mae - m.level.ai_recursive_mae).toFixed(2)}dB로 작습니다. 자세한 수치: <a href="https://github.com/neogeo0701/calm-map-sdot/blob/main/model/metrics.json">model/metrics.json</a></p>`;
}

function renderMap() {
  map = L.map("map", { scrollWheelZoom: false });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  const css = getComputedStyle(document.documentElement);
  const pts = [];
  for (const s of DATA.sites) {
    const p = [s.place_lat, s.place_lon], q = [s.sensor_lat, s.sensor_lon];
    pts.push(p, q);
    L.polyline([p, q], { color: css.getPropertyValue("--muted"), weight: 1.5, dashArray: "4 4" }).addTo(map);
    L.circleMarker(p, { radius: 9, color: "#fff", weight: 2, fillColor: css.getPropertyValue("--accent"), fillOpacity: 1 })
      .bindPopup(`<b>${s.name}</b><br>센서까지 경계 거리 ${Math.round(s.dist_m)}m`).addTo(map);
    L.marker(q, { icon: L.divIcon({ className: "", html: `<div style="width:12px;height:12px;background:${css.getPropertyValue("--pred")};border:2px solid #fff;border-radius:2px"></div>`, iconSize: [14, 14] }) })
      .bindPopup(`S-DoT 센서 ${s.sensor_serial}<br>${s.sensor_address}`).addTo(map);
  }
  map.fitBounds(pts, { padding: [24, 24] });
}

function initBigText() {
  const btn = $("#bigText");
  let on = false;
  try { on = localStorage.getItem("calm-big") === "1"; } catch (e) {}
  const apply = () => {
    document.documentElement.classList.toggle("big", on);
    btn.setAttribute("aria-pressed", String(on));
    if (map) setTimeout(() => map.invalidateSize(), 50);
  };
  btn.addEventListener("click", () => {
    on = !on;
    try { localStorage.setItem("calm-big", on ? "1" : "0"); } catch (e) {}
    apply();
  });
  apply();
}

fetch("predictions.json")
  .then((r) => r.json())
  .then((d) => {
    DATA = d;
    buildDays();
    render();
    renderPerf();
    renderMap();
    $("#cond").addEventListener("change", render);
    $("#gen").textContent = `실측 데이터 ${d.data_period[0].slice(0, 10)} ~ ${d.data_period[1].slice(0, 10)} · 예측 기간 ${d.forecast_period[0].slice(0, 10)} ~ ${d.forecast_period[1].slice(0, 10)} · 생성 ${d.generated_at.replace("T", " ")}`;
  })
  .catch(() => {
    $("#recs").innerHTML = `<li class="rec"><span class="tag na">정보 부족</span> 예측 파일을 불러오지 못했습니다.</li>`;
  });
initBigText();
