import torch
import os
import json
import datetime
from PIL import Image
from sconf import Config
from src.donut_lib import train
from donut import DonutModel


def run_model_on_file(model_path, file_path):

    with open(os.path.join(model_path, 'special_tokens_map.json'), "r") as file:
        special_tokens = json.load(file)
        prompt = special_tokens["additional_special_tokens"][0]

    model = DonutModel.from_pretrained(model_path, ignore_mismatched_sizes=True)

    if torch.cuda.is_available():
        model.half()
        device = torch.device("cuda")
        model.to(device)
    else:
        model.to("cpu")

    model.eval()

    image = Image.open(file_path).convert("RGB")
    with torch.no_grad():
        output = model.inference(image=image, prompt=prompt)
        return output["predictions"][0]


def train_from_config_file(config_path, exp_name):

    torch.cuda.empty_cache()
    config = Config(config_path)
    config.exp_name = exp_name
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train(config)


if __name__ == "__main__":

    # training model
    # config_path = "data/synthetic_forms/divers/config.yaml"
    # exp_name = "cerfa_3_types"
    # train_from_config_file(config_path, exp_name)

    # run inference
    model_path = os.path.join("models/donut_trained/divers/20231002_095949")
    file_path = "data/synthetic_forms/divers/test/cerfa_12485_03_fake32.jpg"
    pred = run_model_on_file(model_path, file_path)
    for k, v in pred.items():
        print(f"{k}: {v}")
