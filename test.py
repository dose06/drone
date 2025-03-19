import argparse
import os

import torch

import dataset
import net
import train
import utils


def test(args: argparse.Namespace) -> None:
    print("Test started...")

    model = net.WaveNetModel().cuda()

    model_path = os.path.join(args.model_dir, args.model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    dataloader, file_list = dataset.get_test_loader(args)

    criterion = torch.nn.MSELoss()

    model.eval()

    score_list = [["File", "Score"]]

    for idx, data in enumerate(dataloader):
        log_mel = data[0].cuda()

        recon_log_mel = model(log_mel)

        loss = criterion(recon_log_mel, log_mel[..., model.get_receptive_field() :])

        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]

        score_list.append([file_name, loss.item()])

    utils.save_csv(score_list, os.path.join(args.result_dir, "test_score.csv"))


if __name__ == "__main__":
    args = train.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test(args)
