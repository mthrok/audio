import time
from collections import namedtuple

import torch
import torch.distributed as dist

from utils import dist_utils, metrics

_LG = dist_utils.getLogger(__name__)

TrainMetric = namedtuple("TrainMetric", ["si_snr"])
TrainMetric.__str__ = (
    lambda self: f"SI-SNR: {self.si_snr:10.3e}"
)

EvalMetric = namedtuple("EvalMetric", ["si_snri", "sdri"])
EvalMetric.__str__ = (
    lambda self: f"SI-SNRi: {self.si_snri:10.3e}, SDRi: {self.sdri: 10.3e}"
)


class OccasionalLogger:
    """Simple helper class to log once in a while or when progress is quick enough"""

    def __init__(self, time_interval=180, progress_interval=0.1):
        self.time_interval = time_interval
        self.progress_interval = progress_interval

        self.last_time = 0.0
        self.last_progress = 0.0

    def log(self, metric, progress, force=False):
        now = time.monotonic()
        if (
            force
            or now > self.last_time + self.time_interval
            or progress > self.last_progress + self.progress_interval
        ):
            self.last_time = now
            self.last_progress = progress
            _LG.info_on_master("train: %s [%3d%%]", metric, 100 * progress)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        valid_loader,
        eval_loader,
        grad_clip,
        device,
        *,
        debug,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader
        self.grad_clip = grad_clip
        self.device = device
        self.debug = debug

    def train_one_epoch(self):
        self.model.train()
        logger = OccasionalLogger()

        num_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader, start=1):
            mix = batch.mix.to(self.device)
            src = batch.src.to(self.device)
            mask = batch.mask.to(self.device)

            estimate = self.model(mix)
            si_sdr = metrics.si_sdr_pit(estimate, src, mask).mean()

            loss = -si_sdr
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, norm_type=2.0
            )
            self.optimizer.step()

            metric = TrainMetric(si_sdr.item())
            logger.log(metric, progress=i / num_batches, force=i == num_batches)

            if self.debug:
                break

    def evaluate(self):
        with torch.no_grad():
            return self._test(self.eval_loader)

    def validate(self):
        with torch.no_grad():
            return self._test(self.valid_loader)

    def _test(self, loader):
        self.model.eval()

        total_si_snri = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_sdri = torch.zeros(1, dtype=torch.float32, device=self.device)

        for batch in loader:
            mix = batch.mix.to(self.device)
            src = batch.src.to(self.device)
            mask = batch.mask.to(self.device)

            estimate = self.model(mix)
            si_snri = metrics.si_sdri(estimate, src, mix, mask)
            sdri = metrics.sdri(estimate, src, mix, mask)

            total_si_snri += si_snri.sum()
            total_sdri += sdri.sum()

            if self.debug:
                break

        dist.all_reduce(total_si_snri, dist.ReduceOp.SUM)
        dist.all_reduce(total_sdri, dist.ReduceOp.SUM)

        num_samples = len(loader.dataset)
        metric = EvalMetric(total_si_snri.item() / num_samples, total_sdri.item() / num_samples)
        return metric
