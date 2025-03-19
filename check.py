import os

train_dir = "C:/Users/조성찬/OneDrive - UOS/바탕 화면/ICSV31-AI-Challenge-main/DRONE DATA/train"

# ✅ 경로 존재 여부 확인
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"❌ 경로가 존재하지 않습니다: {train_dir}")

# ✅ .wav 파일이 있는지 확인
wav_files = [file for file in os.listdir(train_dir) if file.endswith(".wav")]

if not wav_files:
    raise RuntimeError(f"❌ {train_dir} 내에 .wav 파일이 없습니다! 경로를 다시 확인하세요.")

print(f"✅ Training directory 확인됨: {train_dir}")
print(f"✅ 발견된 WAV 파일 목록: {wav_files[:10]}")  # 처음 10개만 출력
