import os
import math
import random
import re
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from nltk import edit_distance
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from donut import DonutDataset, DonutConfig, DonutModel

"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.get("pretrained_model_name_or_path", False):
            self.model = DonutModel.from_pretrained(
                self.config.pretrained_model_name_or_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = DonutModel(
                config=DonutConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    # with DonutConfig, the architecture customization is available, e.g.,
                    # encoder_layer=[2,2,14,2], decoder_layer=4, ...
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, decoder_labels = list(), list(), list()
        for batch_data in batch:
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1][:, :-1])
            decoder_labels.append(batch_data[2][:, 1:])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        decoder_labels = torch.cat(decoder_labels)
        loss = self.model(image_tensors, decoder_input_ids, decoder_labels)[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        image_tensors, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
        )["predictions"]

        scores = list()
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                self.print(f"Prediction: {pred}")
                self.print(f"    Answer: {answer}")
                self.print(f" Normed ED: {scores[0]}")

        return scores

    def validation_epoch_end(self, validation_step_outputs):
        num_of_loaders = len(self.config.dataset_name_or_paths)
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)

    def configure_optimizers(self):

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class DonutDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    pl.utilities.seed.seed_everything(config.get("seed", 42), workers=True)

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "validation": []}
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...

        # add categorical special tokens (optional)
        if task_name == "rvlcdip":
            model_module.model.decoder.add_special_tokens([
                "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>",
                "<form/>", "<handwritten/>", "<invoice/>", "<letter/>",
                "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>",
                "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
            ])
        if task_name == "docvqa":
            model_module.model.decoder.add_special_tokens(["<yes/>", "<no/>"])

        for split in ["train", "validation"]:
            datasets[split].append(
                DonutDataset(
                    dataset_name_or_path=dataset_name_or_path,
                    donut_model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"<s_{task_name}>",
                    prompt_end_token="<s_answer>" if "docvqa" in dataset_name_or_path else f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key,
                )
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        resume_from_checkpoint=config.get("resume_from_checkpoint_path", None),
        num_nodes=config.get("num_nodes", 1),
        gpus=torch.cuda.device_count(),
        # strategy="ddp",
        accelerator="gpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)
