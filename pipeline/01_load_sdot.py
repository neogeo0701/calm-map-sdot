"""S-DoT 주간 CSV(cp949)를 읽어 소음·기온·습도 시간 단위 테이블로 합친다. 결측은 보간하지 않는다."""
import glob
import pandas as pd

COLS = {
    "시리얼": "serial", "측정시간": "measured_at", "지역": "area_type", "자치구": "district", "행정동": "dong",
    "온도 평균(℃)": "temp", "습도 평균(%)": "humid",
    "소음 최대(dB)": "noise_max", "소음 평균(dB)": "noise_avg", "소음 최소(dB)": "noise_min",
}

frames = []
for f in sorted(glob.glob("data/raw/S-DoT_NATURE_*.csv")):
    df = pd.read_csv(f, encoding="cp949", usecols=list(COLS), dtype=str, index_col=False)
    df = df.rename(columns=COLS)
    frames.append(df)
    print(f, len(df))

df = pd.concat(frames, ignore_index=True)
df["measured_at"] = pd.to_datetime(df["measured_at"].str.replace("_", " "), errors="coerce").dt.floor("h")
for c in ["temp", "humid", "noise_max", "noise_avg", "noise_min"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["measured_at", "serial"]).drop_duplicates(["serial", "measured_at"], keep="last")
df.to_pickle("data/processed/sdot_hourly_all.pkl")
print(df.shape, df["measured_at"].min(), df["measured_at"].max(), df["serial"].nunique())
print(df["area_type"].value_counts())
