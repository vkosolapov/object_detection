import torch

from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm
import copy

from dataloader import DataLoader
from detectors.centernet.dataset import decode_bbox, postprocess


class TrainLoop:
    def __init__(
        self,
        experiment_name,
        device,
        workers,
        datasets,
        image_size,
        batch_size,
        model,
        optimizer,
        num_epochs,
        criterion,
        criterion_weights,
        metrics,
        main_metric,
        scheduler=None,
        early_stopping=None,
        checkpoint_file=None,
    ):
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(f"./runs/{experiment_name}")
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_loaders = {
            "train": DataLoader(
                dataset=datasets["train"],
                phase="train",
                batch_size=batch_size,
                augmentations=None,
                workers=workers,
            ),
            "val": DataLoader(
                dataset=datasets["val"],
                phase="val",
                batch_size=batch_size,
                augmentations=None,
                workers=workers,
            ),
        }
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.criterion_weights = criterion_weights
        self.losses = {}
        self.running_losses = {}
        self.epoch_losses = {}
        self.metrics = metrics
        self.metrics_values = {}
        self.main_metric = main_metric
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_file = checkpoint_file

    def evaluate_minibatch(self):
        self.running_loss += self.loss.item() * self.batch_size
        for key in self.criterion.keys():
            self.running_losses[key] += self.losses[key].item() * self.batch_size

        preds = []
        labels = []
        outputs = decode_bbox(
            self.pred["cls"],
            self.pred["size"],
            self.pred["offset"],
            confidence=0.3,
            device=self.device,
        )
        outputs = postprocess(
            outputs,
            need_nms=False,
            image_shape=self.image_size,
            input_shape=4,
            letterbox_image=False,
            nms_thres=0.4,
        )

        self.labels = self.labels.cpu()
        for i in range(len(outputs)):
            labels_count = self.labels_count[i]
            if not outputs[i] is None:
                pred = {
                    "boxes": torch.Tensor(outputs[i][:, :4]),
                    "scores": torch.Tensor(outputs[i][:, 4]),
                    "labels": torch.Tensor(outputs[i][:, 5]),
                }
            else:
                pred = {
                    "boxes": torch.Tensor(),
                    "scores": torch.Tensor(),
                    "labels": torch.Tensor(),
                }
            preds.append(pred)
            if not self.labels[i] is None:
                label = {
                    "boxes": torch.Tensor(self.labels[i, :labels_count, :4]).view(
                        labels_count, 4
                    ),
                    "labels": torch.Tensor(self.labels[i, :labels_count, 4]).view(
                        labels_count
                    ),
                }
            else:
                label = {
                    "boxes": torch.Tensor(),
                    "labels": torch.Tensor(),
                }
            labels.append(label)
        self.metrics_values = {}
        for key in self.metrics.keys():
            self.metrics_values[key] = self.metrics[key](preds, labels)["map"]

    def log_minibatch(self, phase, epoch, minibatch):
        dataset_size = self.data_loaders[phase].dataset_size
        self.writer.add_scalar(
            f"batch_loss/{phase}",
            self.loss.item(),
            dataset_size // self.batch_size * epoch + minibatch,
        )
        for key in self.criterion.keys():
            self.writer.add_scalar(
                f"batch_loss_{key}/{phase}",
                self.losses[key].item(),
                dataset_size // self.batch_size * epoch + minibatch,
            )
        for key in self.metrics.keys():
            self.writer.add_scalar(
                f"batch_{key}/{phase}",
                self.metrics_values[key],
                dataset_size // self.batch_size * epoch + minibatch,
            )
        self.writer.close()

    def evaluate_epoch(self, phase):
        dataset_size = self.data_loaders[phase].dataset_size
        self.epoch_loss = self.running_loss / dataset_size
        for key in self.criterion.keys():
            self.epoch_losses[key] = self.running_losses[key] / dataset_size
        for key in self.metrics.keys():
            self.metrics_values[key] = self.metrics[key].compute()["map"]
            self.metrics[key].reset()

    def log_epoch(self, phase, epoch):
        self.writer.add_scalar(f"epoch_loss/{phase}", self.epoch_loss, epoch)
        for key in self.criterion.keys():
            self.writer.add_scalar(
                f"epoch_loss_{key}/{phase}", self.epoch_losses[key], epoch
            )
        for key in self.metrics.keys():
            self.writer.add_scalar(
                f"epoch_{key}/{phase}", self.metrics_values[key], epoch
            )
        self.writer.close()

    def log_norm(self, phase, epoch, minibatch):
        dataset_size = self.data_loaders[phase].dataset_size
        total_param_norm = 0
        total_grad_norm = 0
        for p in self.model.parameters():
            param_norm = p.detach().data.norm(2)
            total_param_norm += param_norm.item() ** 2
            grad_norm = p.grad.detach().data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
        total_param_norm = total_param_norm ** (0.5)
        total_grad_norm = total_grad_norm ** (0.5)
        self.writer.add_scalar(
            f"norm/param",
            total_param_norm,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.add_scalar(
            f"norm/grad",
            total_grad_norm,
            dataset_size // self.batch_size * epoch + minibatch,
        )
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        self.running_loss = 0.0
        for key in self.criterion.keys():
            self.running_losses[key] = 0.0
        for (
            i,
            (
                inputs,
                cls_labels,
                size_labels,
                offset_labels,
                mask_labels,
                labels,
                labels_count,
            ),
        ) in tqdm(enumerate(self.data_loaders["train"].data_loader)):
            self.inputs = inputs.to(self.device)
            self.cls_labels = cls_labels.to(self.device)
            self.size_labels = size_labels.to(self.device)
            self.offset_labels = offset_labels.to(self.device)
            self.mask_labels = mask_labels.to(self.device)
            self.labels = labels.to(self.device)
            self.labels_count = labels_count
            with torch.set_grad_enabled(True):
                self.pred = self.model(self.inputs)
                self.losses["cls"] = self.criterion["cls"](
                    self.pred["cls"], self.cls_labels
                )
                self.losses["size"] = self.criterion["size"](
                    self.pred["size"], self.size_labels, self.mask_labels
                )
                self.losses["offset"] = self.criterion["offset"](
                    self.pred["offset"], self.offset_labels, self.mask_labels
                )
                self.loss = 0
                for key in self.criterion.keys():
                    self.loss += self.losses[key] * self.criterion_weights[key]
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.evaluate_minibatch()
            self.log_minibatch("train", epoch, i)
            self.log_norm("train", epoch, i)
        if self.scheduler:
            self.scheduler.step()
        self.evaluate_epoch("train")
        self.log_epoch("train", epoch)
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        if self.scheduler:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pth")

    def test_epoch(self, epoch):
        self.model.eval()
        self.running_loss = 0.0
        for key in self.criterion.keys():
            self.running_losses[key] = 0.0
        for (
            i,
            (
                inputs,
                cls_labels,
                size_labels,
                offset_labels,
                mask_labels,
                labels,
                labels_count,
            ),
        ) in tqdm(enumerate(self.data_loaders["val"].data_loader)):
            self.inputs = inputs.to(self.device)
            self.cls_labels = cls_labels.to(self.device)
            self.size_labels = size_labels.to(self.device)
            self.offset_labels = offset_labels.to(self.device)
            self.mask_labels = mask_labels.to(self.device)
            self.labels = labels.to(self.device)
            self.labels_count = labels_count
            with torch.set_grad_enabled(False):
                self.pred = self.model(self.inputs)
                self.losses["cls"] = self.criterion["cls"](
                    self.pred["cls"], self.cls_labels
                )
                self.losses["size"] = self.criterion["size"](
                    self.pred["size"], self.size_labels, self.mask_labels
                )
                self.losses["offset"] = self.criterion["offset"](
                    self.pred["offset"], self.offset_labels, self.mask_labels
                )
                self.loss = 0
                for key in self.criterion.keys():
                    self.loss += self.losses[key] * self.criterion_weights[key]
            self.evaluate_minibatch()
            self.log_minibatch("val", epoch, i)
        self.evaluate_epoch("val")
        self.log_epoch("val", epoch)
        return self.loss

    def train_model(self):
        since = time.time()
        early_stopping_counter = 0
        if self.checkpoint_file:
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optim_state"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch, self.num_epochs - 1))
            print("-" * 10)
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            if self.metrics_values[self.main_metric] > best_metric:
                best_metric = self.metrics_values[self.main_metric]
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if (
                    self.early_stopping
                    and early_stopping_counter >= self.early_stopping
                ):
                    break
            print()
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val metric: {:4f}".format(best_metric))
        self.model.load_state_dict(best_model_wts)
        torch.save(
            self.model.state_dict(), f"checkpoints/final_{self.experiment_name}.pt"
        )
        return self.model
