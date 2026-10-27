// 무던한 지도 — 정적 웹앱. 예측값은 pipeline/daily_update.py가 매일 미리 계산한 predictions.json만 읽는다(서버·키 없음).
"use strict";

const WEIGHT = { low: 0.5, mid: 1, high: 2 };
const FLOOR_DB = 35; // 센서 측정 하한(학습 데이터 최솟값)
const DOW = ["일", "월", "화", "수", "목", "금", "토"];
const $ = (s) => document.querySelector(s);

let DATA = null;
let map = null;

// 한국시간 현재 "YYYY-MM-DDTHH" (지난 시각을 추천에서 빼는 기준)
const nowKST = () => new Date(Date.now() + 9 * 3600e3).toISOString().slice(0, 13);
const todayKST = () => nowKST().slice(0, 10);
const isPast = (t) => t.slice(0, 13) < nowKST();
const expired = () => (DATA.valid_until || DATA.forecast_period[1]).slice(0, 13) < nowKST();

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
      if (isPast(f.t)) continue; // 지난 시각은 추천하지 않음
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

// 소음 수준 값의 출처 날짜: 같은 요일이면서 마지막 실측 주(level_source_week) 안에 드는 날
function sourceDay(day) {
  const [a, b] = DATA.level_source_week;
  const dt = new Date(day + "T00:00:00");
  while (true) {
    dt.setDate(dt.getDate() - 7);
    const y = `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
    if (y <= b && y >= a) return y;
    if (y < a) return y;
  }
}

function buildDays() {
  const days = [...new Set(DATA.sites[0].forecast.map((f) => f.t.slice(0, 10)))];
  const today = todayKST();
  const slotOpen = (d) => !isPast(`${d}T17`); // 기본 시간대(15~18시)에 아직 남은 시각이 있는 날
  // 기본 조건: 유효기간 안 첫 토요일 15~18시(지났으면 15~18시가 남은 첫 날)
  const def = days.find((d) => new Date(d + "T00:00:00").getDay() === 6 && slotOpen(d)) || days.find(slotOpen) || days[days.length - 1];
  $("#days").innerHTML = days
    .map((d) => {
      const past = d < today;
      return `<label class="${past ? "past" : ""}"><input type="radio" name="day" value="${d}" ${d === def ? "checked" : ""} ${past ? "disabled" : ""}>${fmtDay(d)}${past ? " 지남" : ""}</label>`;
    })
    .join("");
}

function renderValidity() {
  const f = (t) => `${Number(t.slice(5, 7))}/${Number(t.slice(8, 10))}`;
  const v = DATA.valid_until || DATA.forecast_period[1];
  const el = $("#validity");
  el.innerHTML = `예측 생성 <b>${f(DATA.generated_at)} ${DATA.generated_at.slice(11, 16)}</b> · 마지막 실측 <b>${f(DATA.data_period[1])} ${Number(DATA.data_period[1].slice(11, 13))}시</b> · 유효기간 <b>${f(DATA.forecast_period[0])} ~ ${f(v)}</b>
    <br><small>서울시 공개 자료가 1주 단위로 늦게 올라와, 마지막 실측 뒤 며칠~2주를 예측합니다. 매일 아침 자동으로 다시 계산하고, 지난 날짜·시각은 추천하지 않습니다.</small>`;
  el.classList.toggle("expired", expired());
  if (expired()) el.innerHTML += `<br><b>유효기간이 지났습니다. 다음 자동 갱신 뒤 다시 열어 주세요.</b>`;
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
  return `<div class="dayline"><span class="k">${fmtDay(cond.day)} 하루 시간대별 부담 점수(낮을수록 무던함 · 진한 막대=고른 시간대, 테두리=추천 시각)</span>
    <div class="bars" aria-hidden="true">${bars}</div><div class="bars-x"><span>0시</span><span>6시</span><span>12시</span><span>18시</span></div></div>`;
}

function render() {
  const cond = readCond();
  const recs = expired() ? [] : recommend(cond);
  const lv = { low: "낮음", mid: "보통", high: "높음" };
  $("#r-sub").textContent = `${fmtDay(cond.day)} ${cond.slot.replace("-", "~")}시 · 소리 크기 민감도 ${lv[cond.level]} · 갑작스러운 소리 민감도 ${lv[cond.spike]}`;
  renderAlt(cond, recs);
  $("#r-sum").innerHTML = recs.map((r, i) => `<span><b>${i + 1}</b> ${r.site.name} ${r.hour}시</span>`).join("");
  if (!recs.length) {
    $("#recs").innerHTML = `<li class="rec"><span class="tag na">정보 부족</span> 고른 시간대에 예측값이 있는 장소가 없습니다.</li>`;
    return;
  }
  const lwk = DATA.level_source_week.map((d) => d.slice(5).replace("-", "/")).join("~");
  const M = DATA.live_methods;
  // 값의 출처 설명(화면 방법별)
  const source = (t, f) => M[t] === "lastweek"
    ? `마지막 관측 주(${lwk})의 같은 요일·시간 값: ${fmtDay(sourceDay(f.t.slice(0, 10)))} ${Number(f.t.slice(11, 13))}시 실측`
    : M[t] === "baseline" ? `${DATA.spike_history_start.slice(0, 4)}년 이후 같은 요일·시간 실측 평균` : M[t] === "recent4w" ? "마지막 관측 4주의 같은 요일·시간 평균" : "AI 예측";
  const tagCls = (t) => (M[t] === "lastweek" ? "obs" : "pred");
  $("#recs").innerHTML = recs
    .map((r, i) => {
      const s = r.site, f = r.f;
      const confL = s.confidence.level, confS = s.confidence.spike;
      const spikeName = i === 0 ? "돌발 소음 지수 <small>(갑자기 커지는 소리 정도)</small>" : "돌발 소음 지수";
      return `<li class="rec">
        <p class="summary">${summaryLine(f)}</p>
        <div class="rec-head"><span class="rank">${i + 1}</span><span class="rec-title">${s.name}</span>
          <span class="rec-time">${fmtTime(f.t)}</span></div>
        <div class="facts">
          ${fact(tagCls("spike"), `${spikeName} <span class='tag ${tagCls("spike")}'>${METHOD_LABEL[M.spike]}</span>`, `${n1(f.spike)} dB · ${grade("spike", f.spike)}`, `${source("spike", f)} · 신뢰도 <span class="tag conf-${confS}">${confS}</span>`)}
          ${fact(tagCls("level"), `소음 수준 <span class='tag ${tagCls("level")}'>${METHOD_LABEL[M.level]}</span>`, `${n1(f.level)} dB · ${grade("level", f.level)}`, `${source("level", f)} · 신뢰도 <span class="tag conf-${confL}">${confL}</span>`)}
          ${fact("obs", "최근 실측 <span class='tag obs'>실측</span>", `평균 ${s.recent.level} · 최대 ${s.recent.max} dB`, `${fmtTime(s.recent.t)} 측정`)}
          ${fact("obs", "최근 1주 평균 <span class='tag obs'>실측</span>", `${n1(s.recent_week.level)} dB · 돌발 ${n1(s.recent_week.spike)} dB`, `${lwk} 7일`)}
          ${fact("", "센서 위치", `공원 경계에서 ${Math.round(s.dist_m)}m`, "")}
        </div>
        ${dayBars(s, cond, r.hour)}
        <details class="more">
          <summary>자세히 보기</summary>
          <div class="facts">
            ${["spike", "level"].map((t) => fact("", `${t === "spike" ? "돌발 소음 지수" : "소음 수준"} 평균 오차`, `${METHOD_LABEL[M[t]]} ±${s.mae[t].used.toFixed(2)} dB`,
              `${Object.keys(METHOD_LABEL).filter((m) => m !== M[t]).map((m) => `${METHOD_LABEL[m]} ±${s.mae[t][m].toFixed(2)}`).join(" · ")} dB (테스트 기간, 이 장소)`)).join("")}
            ${M.spike === "baseline" && M.level === "baseline"
              ? fact("", "요일·시간 평균", "화면 값과 같음", "같은 요일·시각의 2024년 이후 실측 평균을 그대로 씁니다")
              : fact("", "단순 평균이었다면", `돌발 ${f.spike_baseline == null ? "정보 부족" : n1(f.spike_baseline) + " dB"} · 소음 ${f.level_baseline == null ? "정보 부족" : n1(f.level_baseline) + " dB"}`, "같은 요일·시각의 2024년 이후 평균")}
            ${fact("", "부담 점수", r.score.toFixed(1), `= ${WEIGHT[cond.level]}×(${n1(f.level)}−35) + ${WEIGHT[cond.spike]}×${n1(f.spike)}`)}
            ${fact("", "센서", s.sensor_serial, s.sensor_address)}
          </div>
        </details>
      </li>`;
    })
    .join("");
}

const METHOD_LABEL = { baseline: "요일·시간 평균", ai: "AI 예측", recent4w: "직전 4주 평균", lastweek: "마지막 관측 주 값" };
const SLOTS = { "9-12": "오전 9~12시", "12-15": "낮 12~15시", "15-18": "오후 15~18시", "18-21": "저녁 18~21시" };
const avgScore = (recs) => recs.reduce((a, r) => a + r.score, 0) / recs.length;

// 시간대 대안 추천: 추천 3곳이 모두 "많은 편"(갑자기 커지는 소리 또는 소리 크기)일 때,
// 같은 날짜·앞뒤 하루의 다른 시간대 중 추천 3곳 부담 점수 평균이 가장 낮은 곳(지금보다 낮을 때만)
function altSlot(cond, recs) {
  if (recs.length < 3) return null;
  const allMany = (t) => recs.every((r) => grade(t, r.f[t]) === "많은 편");
  if (!allMany("spike") && !allMany("level")) return null;
  const days = [...new Set(DATA.sites[0].forecast.map((f) => f.t.slice(0, 10)))];
  const di = days.indexOf(cond.day);
  let best = null;
  for (const day of days.slice(Math.max(0, di - 1), di + 2)) {
    for (const slot of Object.keys(SLOTS)) {
      if (day === cond.day && slot === cond.slot) continue;
      const rr = recommend({ ...cond, day, slot });
      if (rr.length < 3) continue;
      const sc = avgScore(rr);
      if (!best || sc < best.score) best = { day, slot, score: sc };
    }
  }
  return best && best.score < avgScore(recs) ? { ...best, now: avgScore(recs) } : { none: true };
}

function renderAlt(cond, recs) {
  const box = $("#r-alt");
  const a = altSlot(cond, recs);
  if (!a) { box.hidden = true; box.innerHTML = ""; return; }
  const dt = new Date(a.day ? a.day + "T00:00:00" : 0);
  box.hidden = false;
  box.innerHTML = a.none
    ? `이 시간대는 전반적으로 소리가 많은 편이에요.`
    : `이 시간대는 전반적으로 소리가 많은 편이에요. <button type="button" class="alt-btn" data-day="${a.day}" data-slot="${a.slot}">${DOW[dt.getDay()]}요일(${dt.getMonth() + 1}/${dt.getDate()}) ${SLOTS[a.slot]}가 더 조용해요 →</button>
       <small>부담 점수 평균 ${a.now.toFixed(1)} → ${a.score.toFixed(1)}</small>`;
  const btn = box.querySelector(".alt-btn");
  if (btn) btn.addEventListener("click", () => {
    $(`input[name="day"][value="${btn.dataset.day}"]`).checked = true;
    $(`input[name="slot"][value="${btn.dataset.slot}"]`).checked = true;
    render();
  });
}

// 보호자용 등급: 테스트 기간 실측 분포의 1/3·2/3 분위값 기준(grade_cuts)
function grade(t, v) {
  if (v == null) return "정보 부족";
  const [a, b] = DATA.grade_cuts[t];
  return v < a ? "적은 편" : v < b ? "보통" : "많은 편";
}

function summaryLine(f) {
  const lv = grade("level", f.level), sp = grade("spike", f.spike);
  return `소리 크기 <b>${lv}</b>, 갑자기 커지는 소리 <b>${sp}</b>`;
}

function renderPerf() {
  const T = DATA.test, B = DATA.backtest;
  const p1 = (v) => `${(100 * v).toFixed(1)}%`;
  const c = (v, best) => `<td class="num${best ? " better" : ""}">${v.toFixed(2)}</td>`;
  const cp = (v, best) => `<td class="num${best ? " better" : ""}">${p1(v)}</td>`;
  const sp = T.spike, lv = T.level, P = B.point, CI = B.ci_screen_minus;
  const ci = (x) => `${x.diff >= 0 ? "+" : ""}${(100 * x.diff).toFixed(1)}%p [${(100 * x.lo).toFixed(1)}, ${(100 * x.hi).toFixed(1)}]`;
  const gh = "https://github.com/neogeo0701/calm-map-sdot/blob/main/";
  const rows = [["screen", "이전 화면(돌발 AI + 소음 마지막 관측 주)"], ["ai", "AI 예측"], ["baseline", "요일·시간 평균 ← 지금 화면"], ["recent4w", "직전 4주 평균"], ["lastweek", "마지막 관측 주 값"]];
  $("#perf").innerHTML = `
    <p><b>추천이 실제로 맞았는지</b>를 과거로 되돌아가 확인했습니다(순차 백테스트). ${B.period[0]}부터 ${B.period[1]}까지 기준일 ${B.origins}개마다 그날까지의 자료만으로 9~15일 뒤를 예측해 추천을 만들고, 실제 측정과 비교했습니다.</p>
    <table><thead><tr><th>방법</th><th>가장 조용한 시간대<br>맞힌 비율 ↑</th><th>"많은 편"을 놓친<br>비율 ↓</th></tr></thead><tbody>
    ${rows.map(([m, l]) => `<tr><td>${l}</td>${cp(P[m].M1, m === "baseline")}${cp(P[m].M4_spike, m === "baseline")}</tr>`).join("")}
    </tbody></table>
    <p class="muted">"많은 편을 놓친 비율" = 추천한 곳·시각이 실제로는 갑자기 커지는 소리 "많은 편"이었는데 화면은 "적은 편/보통"이라고 한 비율. 이전 화면 − 요일·시간 평균 차이(95% 신뢰구간): 맞힌 비율 ${ci(CI.baseline.M1)}, 놓친 비율 ${ci(CI.baseline.M4_spike)}.</p>
    <p><b>AI가 단순한 요일·시간 평균보다 추천을 더 잘한다는 근거를 찾지 못했습니다.</b> 그래서 지금 화면은 두 값 모두 ${DATA.spike_history_start.slice(0, 4)}년 이후 같은 요일·시간 실측 평균을 씁니다. AI는 "최근 몇 주만 보는 방법"보다는 나았지만(놓친 비율 ${ci(CI.lastweek.M4_spike)}), 긴 기간 평균은 넘지 못했습니다.
    "놓친 비율"은 지나친 경고에는 벌점을 주지 않아 평균값에 유리할 수 있습니다(자세한 해석: <a href="${gh}BACKTEST_REPORT.md">BACKTEST_REPORT.md</a>).</p>
    <details class="more"><summary>값 자체의 평균 오차(테스트 ${DATA.split.test[0].slice(5, 10)}~${DATA.split.test[1].slice(5, 10)}, 1회 평가)</summary>
    <table><thead><tr><th>대상</th><th>AI</th><th>요일·시간<br>평균</th><th>직전 4주<br>평균</th><th>마지막<br>관측 주 값</th></tr></thead>
    <tbody>
    <tr><td>돌발 소음 지수</td>${c(sp.ai)}${c(sp.baseline, true)}${c(sp.ref_recent4w_mean)}${c(sp.ref_last_week_naive)}</tr>
    <tr><td>소음 수준</td>${c(lv.ai)}${c(lv.baseline, true)}${c(lv.ref_recent4w_mean)}${c(lv.ref_last_week_naive)}</tr>
    </tbody></table>
    <p class="muted">화면과 같은 조건(데이터 마지막 시각에서 ${DATA.horizon_hours[0]}~${DATA.horizon_hours[1]}시간 뒤)의 평균 오차(dB, 낮을수록 정확). 모든 방법이 같은 학습 기간(2024-01 이후). 5개 센서, 실측 ${sp.n}시간. 결측 시간은 보간하지 않고 뺐습니다. 수치: <a href="${gh}model/attempts/long_history/test_metrics.json">test_metrics.json</a> · <a href="${gh}model/attempts/backtest/backtest_results.json">backtest_results.json</a></p>
    </details>`;
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
    renderValidity();
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
