import argparse
import os

import torch

import dataset
import net
import train
import utils
from train import get_args  
from model.baseline.transformerchatgpt import TransformerModel


def get_unique_filename(base_path, base_name, extension):
    """
    base_path: 파일이 저장될 디렉터리
    base_name: 파일 기본 이름 (예: "test_score")
    extension: 파일 확장자 (예: ".csv")
    """
    idx = 0
    filename = os.path.join(base_path, f"{base_name}{extension}")
    # 만약 파일이 존재하면, 숫자 인덱스를 붙여서 새로운 이름을 만듭니다.
    while os.path.exists(filename):
        idx += 1
        filename = os.path.join(base_path, f"{base_name}_{idx}{extension}")
    return filename


def test(args: argparse.Namespace) -> None:
    print("Test started...")

    input_dim = args.n_mels  # 예: 128
    model = TransformerModel(input_dim=input_dim, d_model=256, nhead=8, num_layers=4).cuda()

    model_path = os.path.join(args.model_dir, args.model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    dataloader, file_list = dataset.get_test_loader(args)

    criterion = torch.nn.MSELoss()

    model.eval()

    score_list = [["File", "Score"]]

    for idx, data in enumerate(dataloader):
        # 만약 log_mel의 shape가 [batch, 1, seq_len, n_mels]라면, squeeze로 차원 제거
        log_mel = data[0].squeeze(1).cuda()  # 결과: [batch, seq_len, n_mels]
        recon_log_mel = model(log_mel)
        loss = criterion(recon_log_mel, log_mel)
        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]
        score_list.append([file_name, loss.item()])

    # 고유한 파일 이름 생성 후 CSV 저장
    result_path = get_unique_filename(args.result_dir, "test_score", ".csv")
    utils.save_csv(score_list, result_path)
    print(f"결과가 {result_path}에 저장되었습니다.")


if __name__ == "__main__":
    args = train.get_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    test(args)
