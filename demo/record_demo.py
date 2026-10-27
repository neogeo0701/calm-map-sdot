"""데모 영상 녹화: Playwright로 웹앱을 자동 조작하며 1920x1080 프레임을 찍고, ffmpeg로 자막을 입혀 mp4를 만든다.

사용법: python demo/record_demo.py [앱 URL]   (기본 http://localhost:8765/ — `python -m http.server 8765 -d public`)
설명 화면(slides.html)은 `python demo/build_slides.py`로 먼저 만든다.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import http.server
import functools

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

APP = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/"
DEMO = pathlib.Path("demo").resolve()
FRAMES = DEMO / "frames"
FPS = 15

shutil.rmtree(FRAMES, ignore_errors=True)
FRAMES.mkdir(parents=True)

# 설명 화면용 로컬 서버(demo 폴더)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DEMO)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
SLIDES = "http://127.0.0.1:8766/slides.html"


class Rec:
    def __init__(self, page):
        self.page, self.n, self.t, self.list, self.subs = page, 0, 0.0, [], []

    def snap(self, dur):
        p = FRAMES / f"f{self.n:05d}.png"
        self.page.screenshot(path=str(p))
        self.list.append((p.name, dur))
        self.n += 1
        self.t += dur

    def hold(self, secs):
        self.snap(secs)

    def sub(self, text, secs, fn=None):
        """자막 한 줄을 secs초 동안 보여주며 그동안 fn(장면 동작)을 실행한다."""
        start = self.t
        if fn:
            fn()
        rest = start + secs - self.t
        if rest > 0.05:
            self.hold(rest)
        self.subs.append((start, self.t, text))

    def scroll_to(self, selector, secs=1.2, offset=70):
        y0 = self.page.evaluate("window.scrollY")
        y1 = self.page.evaluate(f"Math.max(0, document.querySelector('{selector}').getBoundingClientRect().top + window.scrollY - {offset})")
        steps = max(1, int(secs * FPS))
        for i in range(1, steps + 1):
            k = i / steps
            e = k * k * (3 - 2 * k)
            self.page.evaluate(f"window.scrollTo(0, {y0 + (y1 - y0) * e})")
            self.snap(1 / FPS)

    def click(self, selector, pause=0.7):
        loc = self.page.locator(selector)
        loc.evaluate("e => { e.style.outline = '4px solid #c4651c'; e.style.outlineOffset = '2px'; }")
        self.snap(pause)
        loc.click()
        loc.evaluate("e => { e.style.outline = ''; }")
        self.page.wait_for_timeout(150)
        self.snap(pause)

    def slide(self, sid):
        self.page = slides_page
        self.page.goto(f"{SLIDES}#{sid}")
        self.page.wait_for_load_state("networkidle")
        self.page.evaluate(f"location.hash = '{sid}'")
        self.page.wait_for_timeout(200)


def radio(name, value):
    return f'label:has(input[name="{name}"][value="{value}"])'


with sync_playwright() as p:
    b = p.chromium.launch()
    slides_page = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
    app_page = b.new_page(viewport={"width": 960, "height": 540}, device_scale_factor=2)
    page = app_page
    r = Rec(slides_page)

    # 0:00 문제 제기
    r.slide("problem")
    r.sub("가상 이용 시나리오: 소리에 예민한 아이와 공원에 가려는 보호자", 8)
    r.sub("갑자기 커지는 소리가 언제 적을지 미리 알 방법이 없습니다", 8)
    r.slide("idea")
    r.sub("무던한 지도: AI가 '돌발 소음 지수'를 예측해 공원 방문 시간을 추천합니다", 9)

    # 0:25 센서 위치와 실측 원본
    r.slide("raw")
    r.sub("서울 열린데이터광장 S-DoT 원본: 1시간마다 소음 최대·평균·최소", 8)
    r.sub("평균만으로는 갑작스러운 소리를 놓치므로 '최대 − 평균'을 돌발 소음 지수로 정했습니다", 7)
    r.page = app_page
    page.goto(APP)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)
    r.sub("송파구 공원·놀이터 5곳, 공원 경계 0~26m 안의 센서 5대만 골랐습니다", 8,
          lambda: r.scroll_to("#m-title", 1.5))
    r.sub("초록 = 공원, 주황 = 센서. 지도에는 선정 장소와 센서만 표시합니다", 7,
          lambda: (page.locator(".leaflet-marker-icon").first.click(), page.wait_for_timeout(400), r.hold(0.1)))

    # 0:55 예측 결과와 기준 모델 대비 평균 오차(MAE)
    r.slide("test")
    r.sub("2024년부터 약 34개월 실측으로 학습, 검증 2.1주 · 테스트 3.5주로 나눠 확인했습니다", 9)
    r.sub("돌발 소음 지수: 검정 = 실측, 주황 = AI 예측, 파란 점선 = 요일·시간 평균", 7)
    r.slide("mae")
    r.sub("돌발 소음 지수(갑자기 커지는 소리 정도) 평균 오차: AI 1.48dB, 지난주 그대로 1.86, 긴 기간 평균 1.49", 8)
    r.sub("소음 수준은 AI(1.24)가 '지난주 그대로'(1.18dB)를 넘지 못해 지난주 값을 씁니다", 9)
    r.slide("importance")
    r.sub("돌발 지수 AI가 주로 본 것: 1주 전 같은 시각, 1주 전 주간 평균, 시간대", 7)

    # 1:35 조건을 바꾸면 추천이 달라지는 시연 (2회)
    r.page = app_page
    page.goto(APP)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1200)
    r.sub("조건 3개: 소리 크기 민감도, 갑작스러운 소리 민감도, 요일·시간대", 6)
    r.sub("10월 31일(토) 오후 15~18시, 둘 다 '보통'으로 추천 3곳", 6, lambda: r.scroll_to("#r-title", 1.2, 20))
    r.sub("카드 맨 위 한 줄 요약: '소리 크기 적은 편, 갑자기 커지는 소리 많은 편'", 7, lambda: r.scroll_to("#recs li:first-child .summary", 1.2, 110))
    r.sub("그 아래 값의 출처(AI 예측 / 지난주 실측), 신뢰도, 최근 실측, 센서 거리", 6, lambda: r.scroll_to("#recs li:first-child .facts", 1.0, 150))
    r.sub("심사 포인트: '자세히 보기'를 펼치면 평균 오차, 단순 평균 비교, 점수 계산식, 센서 번호", 10,
          lambda: (r.scroll_to("#recs li:first-child details", 1.0, 160), r.click("#recs li:first-child details summary"),
                   r.scroll_to("#recs li:first-child details", 0.8, 110)))
    r.sub("보호자에게는 접어 두고 요약만 먼저 보여 줍니다", 3,
          lambda: (r.click("#recs li:first-child details summary", 0.5), r.scroll_to("#recs li:first-child", 0.8, 110)))
    r.sub("3곳 모두 '갑자기 커지는 소리 많은 편'이라 목록 위에 더 조용한 시간대를 안내합니다", 7,
          lambda: r.scroll_to("#r-alt", 1.2, 110))
    r.sub("시간대 대안 추천: 같은 날·앞뒤 하루 중 부담 점수가 가장 낮은 '일요일 오전 9~12시'", 6)
    r.sub("누르면 그 조건으로 바로 바뀝니다. 삼전공원은 갑자기 커지는 소리 '보통'", 8,
          lambda: (r.click(".alt-btn"), r.scroll_to("#r-sub", 1.0, 110), r.hold(1.5), r.scroll_to("#recs li:nth-child(3)", 1.2, 110)))
    r.sub("시연 1: 다시 토요일 오후, 갑작스러운 소리에 더 예민한 아이라면", 5,
          lambda: (r.scroll_to("#cond", 1.0), r.click(radio("day", "2026-10-31"), 0.4), r.click(radio("slot", "15-18"), 0.4)))
    r.sub("소리 크기 '낮음', 갑작스러운 소리 '높음'으로 바꿉니다", 6,
          lambda: (r.click(radio("level", "low")), r.click(radio("spike", "high"))))
    r.sub("3위가 거여공원(16시·갑자기 커지는 소리 많은 편) → 삼전공원(15시·보통)으로 바뀌었습니다", 9,
          lambda: (r.scroll_to("#r-title", 1.2, 20), r.hold(2.5), r.scroll_to("#recs li:nth-child(3)", 1.2, 110)))
    r.sub("시연 2: 수요일 오전, 계속되는 소리에 더 예민한 아이라면", 4, lambda: r.scroll_to("#cond", 1.0))
    r.sub("10/28(수), 오전 9~12시, 소리 크기 '높음', 갑작스러운 소리 '낮음'", 7,
          lambda: (r.click(radio("day", "2026-10-28")), r.click(radio("slot", "9-12")),
                   r.click(radio("level", "high"), 0.5), r.click(radio("spike", "low"), 0.5)))
    r.sub("1위가 거여공원 10시로 바뀌었습니다. 카드의 하루 막대로 다른 시간도 비교할 수 있습니다", 8, lambda: r.scroll_to("#r-title", 1.2, 20))

    # 2:30 한계와 기대효과
    r.sub("이 서비스가 말해주지 못하는 것: 실내 소음, 순간 경적, 소리 종류, 아이별 반응. 진단·보장이 아닙니다", 9,
          lambda: r.scroll_to(".limits", 1.5))
    r.sub("큰 글씨 모드도 버튼 하나로 바꿀 수 있습니다", 5,
          lambda: (r.scroll_to("body", 1.0, 0), r.click("#bigText")))
    r.slide("effect")
    r.sub("같은 방법을 서울 센서 1,100여 대 주변 공원으로 넓히고, 실제 방문 피드백으로 검증하는 것이 다음 단계입니다", 10)
    r.sub("무던한 지도 — 검증된 돌발 소음 예측으로 외출 시간을 고르도록 돕습니다", 6)
    b.close()

srv.shutdown()

# ffmpeg: 프레임 목록 → 영상, 자막 입히기
with open(FRAMES / "list.txt", "w", encoding="utf-8") as f:
    for name, dur in r.list:
        f.write(f"file '{name}'\nduration {dur:.4f}\n")
    f.write(f"file '{r.list[-1][0]}'\n")


def ts(s):
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h:02d}:{m:02d}:{s % 60:06.3f}".replace(".", ",")


with open(DEMO / "subs.srt", "w", encoding="utf-8") as f:
    for i, (a, z, text) in enumerate(r.subs, 1):
        f.write(f"{i}\n{ts(a)} --> {ts(z)}\n{text}\n\n")

ff = imageio_ffmpeg.get_ffmpeg_exe()
style = "FontName=Malgun Gothic,FontSize=15,PrimaryColour=&H00FFFFFF,BackColour=&H99000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=22,Bold=1"
subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "frames/list.txt",
                "-vf", f"fps=30,scale=1920:1080:flags=lanczos,format=yuv420p,subtitles=subs.srt:force_style='{style}'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-movflags", "+faststart", "calm-map-demo.mp4"],
               cwd=DEMO, check=True)
print(f"duration {r.t:.1f}s, frames {r.n}, subs {len(r.subs)}")
