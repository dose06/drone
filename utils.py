import csv
from typing import Any, List, Tuple

import torch


def read_csv(file_path: str) -> List:
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        return list(reader)


def save_csv(save_data: List[Any], save_file_path: str) -> None:
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)


def get_anomaly_label(file_path: str) -> int:
    file_name = file_path.split("/")[-1]
    train_mode = file_name.split("_")[0]

    if train_mode == "test":
        return -1
    elif "normal" in file_name:
        return 0
    else:
        return 1


def get_drone_label(file_path: str) -> int:
    file_name = file_path.split("/")[-1]
    drone_mode = file_name.split("_")[1]

    if drone_mode == "A":
        return 0
    elif drone_mode == "B":
        return 1
    elif drone_mode == "C":
        return 2
    else:
        return -1


def get_direction_label(file_path: str) -> int:
    file_name = file_path.split("/")[-1]
    direction_mode = file_name.split("_")[2]

    if direction_mode == "Back":
        return 0
    elif direction_mode == "Front":
        return 1
    elif direction_mode == "Left":
        return 2
    elif direction_mode == "Right":
        return 3
    elif direction_mode == "Clockwise":
        return 4
    elif direction_mode == "CounterClockwise":
        return 5
    else:
        return -1


def save_model(model: torch.nn.Module, save_path: str) -> None:
    torch.save(model.state_dict(), save_path)
