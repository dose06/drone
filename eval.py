import argparse
import os

import sklearn.metrics as metrics
import torch

import dataset
import net
import train
import utils


def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")

    model = net.WaveNetModel().cuda()
    model_path = os.path.join(args.model_dir, args.model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    dataloader, file_list = dataset.get_eval_loader(args)
    criterion = torch.nn.MSELoss()

    score_list = [["File", "Score"]]

    drone_label_list = []
    y_true, y_pred = [], []

    for idx, data in enumerate(dataloader):
        log_mel = data[0].cuda()
        anomaly_label = data[1]
        drone_label = data[2]

        recon_log_mel = model(log_mel)

        loss = criterion(recon_log_mel, log_mel[..., model.get_receptive_field() :])

        drone_label_list.append(drone_label.item())

        y_true.append(1 if anomaly_label.item() > 0 else 0)
        y_pred.append(loss.item())

        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]

        score_list.append([file_name, loss.item()])

    auc = metrics.roc_auc_score(y_true, y_pred)
    print("AUC: ", auc)
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_score.csv"))

    drone_type_list = ["A", "B", "C"]
    for drone_type in drone_type_list:
        indices = [
            i
            for i, label in enumerate(drone_label_list)
            if label == drone_type_list.index(drone_type)
        ]
        pred_labels = [y_pred[i] for i in indices]
        true_labels = [y_true[i] for i in indices]
        fault_auc = metrics.roc_auc_score(true_labels, pred_labels)
        print(f"Drone {drone_type} AUC: {fault_auc}")


if __name__ == "__main__":
    args = train.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eval(args)
