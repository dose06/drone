import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import utils
import torchaudio


class BaselineDataLoader(Dataset):
    def __init__(
        self,
        file_list: list[str],
        sr: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
    ) -> None:
        self.file_list = file_list
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        """ 데이터 로드 및 전처리 """
        wav_path = self.file_list[idx]

        # ✅ 디버깅 출력
        print(f"🔹 Loading file: {wav_path}")

        # ✅ 파일 존재 여부 확인
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {wav_path}")

        # ✅ .wav 파일인지 확인
        if not wav_path.endswith(".wav"):
            raise RuntimeError(f"❌ 잘못된 파일 형식: {wav_path}")

        # 🔹 WAV 파일을 로드하여 MEL 스펙트로그램 변환
        log_mel_spec = wav_to_log_mel(
            wav_path,
            self.sr,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
            self.power,
        )

        # 🔹 추가적인 레이블 로딩 (필요하면 utils.py에서 정의)
        anomaly_label = utils.get_anomaly_label(wav_path)
        drone_label = utils.get_drone_label(wav_path)
        direction_label = utils.get_direction_label(wav_path)

        return log_mel_spec, anomaly_label, drone_label, direction_label



def wav_to_log_mel(
    wav_path: str,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    power: float,
) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    # Windows 경로 문제를 방지하기 위해 경로 정리
    wav_path = os.path.normpath(wav_path)
    print(f"🔹 Loading WAV file: {wav_path}")

    # 형식을 명시적으로 지정합니다.
    try:
        wav_data, sr_actual = torchaudio.load(wav_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"Failed to load wav file {wav_path}. Error: {e}")

    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spec = mel_transform(wav_data)
    log_mel_spec = amp_to_db(mel_spec)
    return log_mel_spec


def get_train_loader(args) -> DataLoader:
    train_dir = args.train_dir

    # ✅ .wav 파일만 포함하도록 필터링
    file_list = [
        os.path.join(train_dir, file)
        for file in os.listdir(train_dir)
        if file.endswith(".wav") and os.path.isfile(os.path.join(train_dir, file))  # 폴더 제외
    ]

    # ✅ 디버깅용 출력
    print(f"🔹 get_train_loader()에서 확인된 train_dir: {train_dir}")
    print(f"🔹 get_train_loader()에서 발견된 WAV 파일 목록: {file_list[:10]}")  # 처음 10개만 출력

    if not file_list:
        raise RuntimeError(f"❌ {train_dir} 내에 .wav 파일이 없습니다! 경로를 다시 확인하세요.")

    file_list.sort()

    train_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.power
    )

    train_loader = DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    return train_loader



def get_eval_loader(args) -> Tuple[DataLoader, List[str]]:
    eval_dir = args.eval_dir

    # ✅ `.wav` 파일만 필터링
    file_list = [os.path.join(eval_dir, file) for file in os.listdir(eval_dir) if file.endswith(".wav")]

    # ✅ 파일이 비어 있는지 확인
    if len(file_list) == 0:
        raise RuntimeError(f"{eval_dir} 내에 .wav 파일이 없습니다!")

    file_list.sort()
    eval_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.power
    )

    eval_loader = DataLoader(
        eval_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return eval_loader, file_list


def get_test_loader(args) -> Tuple[DataLoader, List[str]]:
    test_dir = args.test_dir

    # ✅ `.wav` 파일만 필터링
    file_list = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(".wav")]

    # ✅ 파일이 비어 있는지 확인
    if len(file_list) == 0:
        raise RuntimeError(f"{test_dir} 내에 .wav 파일이 없습니다!")

    file_list.sort()
    test_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length, args.hop_length, args.n_mels, args.power
    )

    test_loader = DataLoader(
        test_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return test_loader, file_list
