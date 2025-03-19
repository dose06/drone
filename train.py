import argparse
import os

import torch
import yaml
from tqdm import tqdm

import dataset
import net
import utils

import yaml

# ✅ config.yaml 확인
with open("param.yaml", "r", encoding="utf-8") as f:
    param = yaml.safe_load(f)

train_dir = param["train_dir"]

print(f"🔹 train.py에서 확인된 train_dir: {train_dir}")

# ✅ train_dir 내부의 파일 목록 출력
import os
if not os.path.exists(train_dir):
    print(f"❌ {train_dir} 경로가 존재하지 않습니다!")
else:
    wav_files = [file for file in os.listdir(train_dir) if file.endswith(".wav")]
    print(f"✅ train.py에서 발견된 WAV 파일 목록: {wav_files[:10]}")  # 처음 10개만 출력

print("CUDA Available:", torch.cuda.is_available())  # CUDA 사용 가능 여부
print("Device Count:", torch.cuda.device_count())   # 사용 가능한 GPU 개수
print("Current Device:", torch.cuda.current_device())  # 현재 사용 중인 GPU 번호
print("Device Name:", torch.cuda.get_device_name(0))  # GPU 이름 출력

def get_args() -> argparse.Namespace:
    param_path = "./param.yaml"
    with open("param.yaml", "r", encoding="utf-8") as f:  # UTF-8로 읽기
        param = yaml.safe_load(f)
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", default=param["train_dir"], type=str)
    parser.add_argument("--eval-dir", default=param["eval_dir"], type=str)
    parser.add_argument("--test-dir", default=param["test_dir"], type=str)

    parser.add_argument("--result-dir", default=param["result_dir"], type=str)
    parser.add_argument("--model-dir", default=param["model_dir"], type=str)

    parser.add_argument("--model-path", default=param["model_path"], type=str)

    parser.add_argument("--epochs", default=param["epochs"], type=int)
    parser.add_argument("--batch-size", default=param["batch_size"], type=int)
    parser.add_argument("--lr", default=param["lr"], type=float)

    parser.add_argument("--gpu", default=param["gpu"], type=int)
    parser.add_argument("--n-workers", default=param["n_workers"], type=int)

    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n-fft", default=param["n_fft"], type=int)
    parser.add_argument("--win-length", default=param["win_length"], type=int)
    parser.add_argument("--hop-length", default=param["hop_length"], type=int)
    parser.add_argument("--n-mels", default=param["n_mels"], type=int)
    parser.add_argument("--power", default=param["power"], type=float)

    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    print("Training started...")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    model = net.WaveNetModel().cuda()

    dataloader = dataset.get_train_loader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()

        p_bar = tqdm(dataloader, total=len(dataloader), desc="Training", ncols=100)
        for data in p_bar:
            log_mel = data[0].cuda()

            recon_log_mel = model(log_mel)

            loss = criterion(recon_log_mel, log_mel[..., model.get_receptive_field() :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    utils.save_model(model, os.path.join(args.model_dir, args.model_path))


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(2025)

    train(args)
