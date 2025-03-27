import argparse
import os

import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt  # 추가

import dataset
import net
import utils

from model.baseline.transformerchatgpt import TransformerModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ✅ config.yaml 확인
with open("param.yaml", "r", encoding="utf-8") as f:
    param = yaml.safe_load(f)

train_dir = param["train_dir"]

print(f"🔹 train.py에서 확인된 train_dir: {train_dir}")

# ✅ train_dir 내부의 파일 목록 출력
if not os.path.exists(train_dir):
    print(f"❌ {train_dir} 경로가 존재하지 않습니다!")
else:
    wav_files = [file for file in os.listdir(train_dir) if file.endswith(".wav")]
    print(f"✅ train.py에서 발견된 WAV 파일 목록: {wav_files[:10]}")  # 처음 10개만 출력

print("CUDA Available:", torch.cuda.is_available())  # CUDA 사용 가능 여부
try:
    print("Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))
except:
    print("❗ CUDA 정보 출력 불가 (아마도 CPU 환경)")
def get_args() -> argparse.Namespace:
    param_path = "./param.yaml"
    with open("param.yaml", "r", encoding="utf-8") as f:
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

    # 예시: 로그 멜 스펙트로그램의 feature 차원이 128이라고 가정
    input_dim = args.n_mels  # 또는 데이터에 맞는 차원
    model = TransformerModel(input_dim=input_dim, d_model=256, nhead=8, num_layers=4).to(device)

    dataloader = dataset.get_train_loader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = torch.nn.MSELoss()

    # 에포크별 평균 loss 기록을 위한 리스트
    epoch_losses = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss_total = 0.0
        batch_count = 0

        p_bar = tqdm(dataloader, total=len(dataloader), desc="Training", ncols=100)
        for data in p_bar:
            # 예시: data[0]의 shape가 [batch, 1, 128, 63]인 경우
            log_mel = data[0].squeeze(1)  # 결과: [batch, 128, 63]
            # time(63)이 시퀀스 길이, 128이 feature dimension이 되어야 하므로 transpose 수행
            log_mel = log_mel.transpose(1, 2).to(device)  # 결과: [batch, 63, 128]

            # 모델을 통해 재구성된 로그 멜 스펙트로그램 출력
            recon_log_mel = model(log_mel)
            loss = criterion(recon_log_mel, log_mel)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_total += loss.item()
            batch_count += 1

            p_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss_total / batch_count if batch_count != 0 else 0
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        scheduler.step()

    # 학습 후 모델 저장
    utils.save_model(model, os.path.join(args.model_dir, args.model_path))

    # 에포크별 평균 loss 그래프 그리기 및 저장
    plt.figure()
    plt.plot(range(1, args.epochs+1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    graph_path = os.path.join(args.result_dir, 'training_loss.png')
    plt.savefig(graph_path)
    plt.show()
    print(f"Training loss graph saved to {graph_path}")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(2025)
    train(args)
