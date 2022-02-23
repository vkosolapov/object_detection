from contextlib import ExitStack
import time
from tqdm import tqdm
import copy
import torch
from torch.cuda.amp import GradScaler, autocast
import torchvision
from dataloader import DataLoader
from scheduler import CyclicCosineDecayLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchmetrics.detection.map import MeanAveragePrecision
from torchmetrics import Precision, Recall

from torch.utils.tensorboard import SummaryWriter


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
        losses_computer,
        predictions_postprocessor,
        metrics,
        main_metric,
        scheduler=None,
        swa=False,
        early_stopping=None,
        mixed_precision=False,
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
                workers=workers,
                device=device,
            ),
            "val": DataLoader(
                dataset=datasets["val"],
                phase="val",
                batch_size=batch_size,
                workers=workers,
                device=device,
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
        self.losses_computer = losses_computer
        self.predictions_postprocessor = predictions_postprocessor
        self.metrics = metrics
        self.metrics_values = {}
        self.main_metric = main_metric
        self.scheduler = scheduler
        self.swa = swa
        if swa:
            self.swa_model = AveragedModel(model)
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = GradScaler()
            self.swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        self.checkpoint_file = checkpoint_file
        self.draw_example = True

    def evaluate_minibatch(self, phase):
        self.running_loss += self.loss.item() * self.batch_size
        for key in self.criterion.keys():
            if self.criterion[key] is None:
                continue
            self.running_losses[key] += self.losses[key].item() * self.batch_size

        if not phase == "train":
            preds = []
            labels = []
            outputs = self.predictions_postprocessor(
                self.pred, self.image_size, self.device
            )
            if self.draw_example:
                reverted_labels = self.predictions_postprocessor(
                    self.targets, self.image_size, self.device
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
                if self.draw_example:
                    # image = torchvision.utils.draw_bounding_boxes(
                    #     self.original_inputs[i]
                    #     .permute(2, 0, 1)
                    #     .type(torch.uint8)
                    #     .cpu(),
                    #     self.labels[i, :labels_count, :4],
                    # )
                    # self.writer.add_image(f"preprocessed_image_{self.epoch}", image)
                    print(self.labels[i, :labels_count, :4].size())
                    if not reverted_labels[i] is None:
                        #     image = torchvision.utils.draw_bounding_boxes(
                        #         self.original_inputs[i]
                        #         .permute(2, 0, 1)
                        #         .type(torch.uint8)
                        #         .cpu(),
                        #         torch.from_numpy(reverted_labels[i][:, :4]),
                        #     )
                        #     self.writer.add_image(
                        #         f"postprocessed_image_{self.epoch}", image
                        #     )
                        print(torch.from_numpy(reverted_labels[i][:, :4]).size())
                    # else:
                    #     print("Empty reverted labels")
                    if not outputs[i] is None:
                        image = torchvision.utils.draw_bounding_boxes(
                            self.original_inputs[i]
                            .permute(2, 0, 1)
                            .type(torch.uint8)
                            .cpu(),
                            torch.from_numpy(outputs[i][:, :4]),
                        )
                        self.writer.add_image(f"predictions_{self.epoch}", image)
                        self.writer.add_scalar(
                            "BBoxes_count", outputs[i][:, :4].shape[0], self.epoch
                        )
                        # print(outputs[i][:, :4].shape)
                    else:
                        print("Empty predictions")
                    self.draw_example = False

            targets_cls = (
                self.targets[0].type(torch.int32).permute(0, 2, 3, 1).reshape([-1, 6])
            )
            _, targets_cls = torch.max(targets_cls, dim=-1)
            preds_cls = torch.sigmoid(self.pred[0]).permute(0, 2, 3, 1).reshape([-1, 6])
            self.metrics_values = {}
            for key in self.metrics.keys():
                if isinstance(self.metrics[key], MeanAveragePrecision):
                    self.metrics_values[key] = self.metrics[key](preds, labels)["map"]
                elif isinstance(self.metrics[key], Precision) or isinstance(
                    self.metrics[key], Recall
                ):
                    self.metrics_values[key] = self.metrics[key](
                        preds_cls.cpu(), targets_cls.cpu()
                    )

    def log_minibatch(self, phase, epoch, minibatch):
        dataset_size = self.data_loaders[phase].dataset_size
        self.writer.add_scalar(
            f"batch_loss/{phase}",
            self.loss.item(),
            dataset_size // self.batch_size * epoch + minibatch,
        )
        for key in self.criterion.keys():
            if self.criterion[key] is None:
                continue
            self.writer.add_scalar(
                f"batch_loss_{key}/{phase}",
                self.losses[key].item(),
                dataset_size // self.batch_size * epoch + minibatch,
            )
        if not phase == "train":
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
            if self.criterion[key] is None:
                continue
            self.epoch_losses[key] = self.running_losses[key] / dataset_size
        if not phase == "train":
            for key in self.metrics.keys():
                if isinstance(self.metrics[key], MeanAveragePrecision):
                    self.metrics_values[key] = self.metrics[key].compute()["map"]
                else:
                    self.metrics_values[key] = self.metrics[key].compute()
                self.metrics[key].reset()

    def log_epoch(self, phase, epoch):
        self.writer.add_scalar(f"epoch_loss/{phase}", self.epoch_loss, epoch)
        for key in self.criterion.keys():
            if self.criterion[key] is None:
                continue
            self.writer.add_scalar(
                f"epoch_loss_{key}/{phase}", self.epoch_losses[key], epoch
            )
        if not phase == "train":
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
        self.epoch = epoch
        self.draw_example = True
        self.model.train()
        self.running_loss = 0.0
        for key in self.criterion.keys():
            if self.criterion[key] is None:
                continue
            self.running_losses[key] = 0.0
        for (i, batch) in tqdm(enumerate(self.data_loaders["train"].data_loader)):
            with ExitStack() as stack:
                if self.mixed_precision:
                    stack.enter_context(autocast())
                self.original_inputs = batch[0]  # .to(self.device)
                self.inputs = batch[1].to(self.device)
                self.labels_count = batch[2]
                self.labels = batch[3]  # .to(self.device)
                self.targets = batch[4:]
                for j in range(len(self.targets)):
                    self.targets[j] = self.targets[j].to(self.device)
                with torch.set_grad_enabled(True):
                    self.pred = self.model(self.inputs)
                    self.losses_computer(
                        self.pred, self.targets, self.criterion, self.losses
                    )
                    self.loss = 0
                    for key in self.criterion.keys():
                        if self.criterion[key] is None:
                            continue
                        self.loss += self.losses[key] * self.criterion_weights[key]
                    self.optimizer.zero_grad()
                    if self.mixed_precision:
                        self.scaler.scale(self.loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.loss.backward()
                        self.optimizer.step()
            self.evaluate_minibatch("train")
            self.log_minibatch("train", epoch, i)
            self.log_norm("train", epoch, i)
        if self.scheduler:
            self.scheduler.step()
        if self.swa and (
            not isinstance(self.scheduler, CyclicCosineDecayLR)
            or self.scheduler._restart_flag == True
        ):
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            update_bn(self.data_loaders["train"].data_loader, self.swa_model)
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
            if self.criterion[key] is None:
                continue
            self.running_losses[key] = 0.0
        for (i, batch) in tqdm(enumerate(self.data_loaders["val"].data_loader)):
            with ExitStack() as stack:
                if self.mixed_precision:
                    stack.enter_context(autocast())
                self.original_inputs = batch[0]  # .to(self.device)
                self.inputs = batch[1].to(self.device)
                self.labels_count = batch[2]
                self.labels = batch[3]  # .to(self.device)
                self.targets = batch[4:]
                for j in range(len(self.targets)):
                    self.targets[j] = self.targets[j].to(self.device)
                with torch.set_grad_enabled(False):
                    if self.swa and (
                        not isinstance(self.scheduler, CyclicCosineDecayLR)
                        or self.scheduler._restart_flag == True
                    ):
                        self.pred = self.swa_model(self.inputs)
                    else:
                        self.pred = self.model(self.inputs)
                    self.losses_computer(
                        self.pred, self.targets, self.criterion, self.losses
                    )
                    self.loss = 0
                    for key in self.criterion.keys():
                        if self.criterion[key] is None:
                            continue
                        self.loss += self.losses[key] * self.criterion_weights[key]
            self.evaluate_minibatch("val")
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
