# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from
https://github.com/pytorch/examples/blob/master/imagenet/main.py."""
import logging
from argparse import ArgumentParser
from argparse import Namespace
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from main import get_dataset
from main import init_benchmarking
from misc.logging_configuration import initialize_logging
from misc.time_helper import stopwatch
from pytorch_lightning.core import LightningModule
from pytorch_lightning.profiler import AdvancedProfiler
from torch_overrides.dataloader import DataLoader
from torch_overrides.worker import _worker_loop
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning import loggers as pl_loggers

class ImageNetLightningModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """

    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        data_path: str,
        arch: str = "resnet18",
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)

    @stopwatch(trace_name="(6)-training_step", trace_level=6)
    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    @stopwatch(trace_name="(7)-validation_step", trace_level=7)
    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("val_loss", loss_val, on_step=True, on_epoch=True)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified
        values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._val_dataloader

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace("val", "test"): v for k, v in out.items()}

        outputs = {
            "test_loss": outputs["val_loss"],
            "progress_bar": substitute_val_keys(outputs["progress_bar"]),
            "log": substitute_val_keys(outputs["log"]),
        }
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("ImageNetLightningModel")
        parser.add_argument(
            "-a", "--arch", metavar="ARCH", default="resnet18",
        )
        parser.add_argument(
            "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 4)"
        )
        # parser.add_argument(
        #     "-b",
        #     "--batch-size",
        #     default=4,
        #     type=int,
        #     metavar="N",
        #     help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node"
        #     " when using Data Parallel or Distributed Data Parallel",
        # )
        parser.add_argument(
            "--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
        return parent_parser


def main(args: Namespace) -> None:
    # create datasets
    val_dataset = get_dataset(args.dataset, dataset_type="val", limit=args.dataset_limit)
    train_dataset = get_dataset(args.dataset, dataset_type="train", limit=args.dataset_limit)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    val_dataset.set_transform(transform)
    train_dataset.set_transform(transform)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        prefetch_factor=args.prefetch_factor,
        num_fetch_workers=args.num_fetch_workers,
        fetch_impl=args.fetch_impl,
        batch_pool=args.batch_pool,
        pin_memory=args.pin_memory,
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        prefetch_factor=args.prefetch_factor,
        num_fetch_workers=args.num_fetch_workers,
        fetch_impl=args.fetch_impl,
        batch_pool=args.batch_pool,
        pin_memory=args.pin_memory,
    )

    output_base_folder = init_benchmarking(
        args=args,
        action="_".join(
            [
                "benchmark_e2e",
                str(args.dataset),
                str(args.batch_size),
                str(args.num_workers),
                str(args.num_fetch_workers),
                "sync",
            ]
        ),
    )

    torch.utils.data._utils.worker._worker_loop = partial(
        _worker_loop,
        initializer=partial(
            initialize_logging, loglevel=logging.getLogger().getEffectiveLevel(), output_base_folder=output_base_folder,
        ),
    )

    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModel(train_dataloader=train_data_loader, val_dataloader=None, **vars(args))
    # model = ImageNetLightningModel(train_dataloader=train_data_loader, **vars(args))
    tb_logger = pl_loggers.TensorBoardLogger(f"{output_base_folder}/lightning/")
    trainer = pl.Trainer.from_argparse_args(args,
                                            profiler="simple",
                                            logger=tb_logger,
                                            log_every_n_steps=5,
                                            callbacks=[GPUStatsMonitor()])

    start_train(args, model, trainer)


@stopwatch(trace_name="(8)-start_train", trace_level=8)
def start_train(args, model, trainer):
    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument("--data-path", metavar="DIR", type=str, help="path to dataset")
    parent_parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set"
    )
    parent_parser.add_argument("--seed", type=int, default=42, help="seed for initializing training.")
    parent_parser.add_argument("--fetch-impl", type=str, default="asyncio", help="vanilla | asyncio | threaded")
    parent_parser.add_argument("--dataset-limit", type=int, default=None)
    parent_parser.add_argument("--batch-pool", type=int, default=10)
    parent_parser.add_argument("--num-fetch-workers", type=int, default=10)
    parent_parser.add_argument("--num-workers", type=int, default=1)
    parent_parser.add_argument("--prefetch-factor", type=int, default=2)
    parent_parser.add_argument("--dataset", type=str, default="s3", help="s3 | scratch")
    parent_parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parent_parser.add_argument("--batch-size", type=int, default=4)
    parent_parser.add_argument("--pin-memory", type=int, default=0)

    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    # parser.set_defaults(deterministic=True, max_epochs=5)
    parser.set_defaults(deterministic=True, max_epochs=5, gpus=[2])
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
