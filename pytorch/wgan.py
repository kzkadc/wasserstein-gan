import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import MNIST
from ignite.engine.engine import Events, Engine

import numpy as np
import cv2
import matplotlib.pyplot as plt

import math
from pathlib import Path
import pprint

from model import get_critic, get_generator


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="WGAN(-GP)")
    parser.add_argument("-b", "--batchsize", type=int, default=100)
    parser.add_argument("-i", "--iteration", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--n_cri", type=int, nargs=2, default=[5, 100])
    parser.add_argument("--gp_lam", type=float, default=10.0)
    parser.add_argument("-g", type=int, default=0, help="GPU ID (negative value indicates CPU mode)")
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    main(args)


Z_DIM = 30


def main(args):
    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    result_dir = Path(args.result_dir)

    # MNISTデータ取得
    mnist_train = MNIST(root=".", download=True, train=True,
                        transform=lambda x: np.expand_dims(np.asarray(x, dtype=np.float32), 0) / 255)
    mnist_loader = DataLoader(mnist_train, args.batchsize)
    mnist_loader = InfiniteDataLoader(mnist_loader)

    generator = get_generator(Z_DIM).to(device)
    critic = get_critic().to(device)

    opt_g = Adam(generator.parameters(), args.alpha, (args.beta1, args.beta2))
    opt_c = Adam(critic.parameters(), args.alpha, (args.beta1, args.beta2))

    trainer = Engine(WGANTrainer(mnist_loader, generator, critic, opt_g, opt_c,
                                 args.n_cri, args.gp_lam, device))

    log_dict = {}
    accumulator = MetricsAccumulator(["generator_loss", "critic_loss"])
    trainer.add_event_handler(Events.ITERATION_COMPLETED, accumulator)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=500), record_metrics(log_dict, accumulator))
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=500), print_metrics(log_dict, accumulator.keys))
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=500),
                              plot_metrics(log_dict, "iteration", accumulator.keys, result_dir / "metrics.pdf"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=500), save_img(generator, result_dir / "generated_samples", device))

    # 指定されたイテレーション数で終了させる
    trainer.add_event_handler(Events.ITERATION_COMPLETED(once=args.iteration), lambda engine: engine.terminate())

    trainer.run(mnist_loader, max_epochs=10**10)


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    @property
    def batch_size(self):
        return self.data_loader.batch_size

    def __iter__(self):
        self.iterator = iter(self.data_loader)
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            batch = next(self.iterator)

        return batch


class WGANTrainer:
    def __init__(self, data_loader, gen: nn.Module, cri: nn.Module,
                 opt_g, opt_c, n_cri, gp_lam: float, device):

        assert len(n_cri) == 2

        self.data_loader = data_loader
        self.gen = gen
        self.cri = cri
        self.opt_g = opt_g
        self.opt_c = opt_c
        self.n_cri = n_cri
        self.gp_lam = gp_lam
        self.device = device

    def __call__(self, engine, batch):
        # 複数回サンプリングする必要があるため渡されたbatchは使わない
        # データはself.data_loaderから取得

        self.gen.train()
        self.cri.train()

        # update critic
        iter_num = engine.state.iteration
        upd_num = self.n_cri[1] if iter_num <= 25 or iter_num % 500 == 0 else self.n_cri[0]
        critic_mean_loss = 0
        i = 0
        for batch in self.data_loader:
            x_real, _ = batch
            x_real = x_real.to(self.device)

            z = np.random.uniform(size=(self.data_loader.batch_size, Z_DIM)).astype(np.float32)
            z = torch.from_numpy(z).to(self.device)
            x_fake = self.gen(z).detach()
            x_fake.requires_grad = True

            cri_loss = (self.cri(x_fake) - self.cri(x_real)).mean()  # Wasserstein距離の逆符号

            # gradient penalty
            eps = np.random.uniform(size=(self.data_loader.batch_size, 1, 1, 1)).astype(np.float32)
            eps = torch.from_numpy(eps).to(self.device)
            x_fusion = eps * x_real + (1 - eps) * x_fake  # (N,1,H,W)
            g_critic = torch.autograd.grad(torch.unbind(self.cri(x_fusion)), x_fusion, create_graph=True)[0]  # (N,1,H,W)
            gp = g_critic.square().sum(dim=(1, 2, 3)).sqrt()
            gp = (gp - 1).square().mean()

            total_loss = cri_loss + self.gp_lam * gp

            self.opt_c.zero_grad()
            total_loss.backward()
            self.opt_c.step()

            critic_mean_loss += total_loss.item()

            i += 1
            if i >= upd_num:
                break

        critic_mean_loss /= upd_num

        # update generator
        z = np.random.uniform(size=(self.data_loader.batch_size, Z_DIM)).astype(np.float32)
        z = torch.from_numpy(z).to(self.device)
        x_fake = self.gen(z)
        gen_loss = -self.cri(x_fake).mean()

        self.opt_c.zero_grad()
        self.opt_g.zero_grad()
        gen_loss.backward()
        self.opt_g.step()

        return {
            "critic_loss": critic_mean_loss,
            "generator_loss": gen_loss.item()
        }


class MetricsAccumulator:
    def __init__(self, keys: list):
        self.keys = keys
        self.logs = {k: 0.0 for k in keys}
        self.num = 0

    def __call__(self, engine):
        for k in self.keys:
            self.logs[k] += engine.state.output[k]
        self.num += 1

    def get_mean(self, key):
        return self.logs[key] / self.num

    def reset(self):
        self.logs = {k: 0.0 for k in self.keys}
        self.num = 0


def record_metrics(log_dict: dict, accumulator: MetricsAccumulator, keys: list = None):
    if keys is None:
        keys = accumulator.keys

    def _record(engine):
        _add("iteration", engine.state.iteration)

        for k in keys:
            _add(k, accumulator.get_mean(k))

        accumulator.reset()

    def _add(key, value):
        if key in log_dict:
            log_dict[key].append(value)
        else:
            log_dict[key] = [value]

    return _record


def plot_metrics(log_dict: dict, x_key: str, y_keys: list, out_path: Path):
    def _plot(engine):
        plt.figure()
        for k in y_keys:
            plt.plot(log_dict[x_key], log_dict[k], label=k)
        plt.legend()
        plt.xlabel(x_key)
        plt.savefig(str(out_path))
        plt.close()

    return _plot


def print_metrics(log_dict: dict, keys: list):
    return lambda engine: print(", ".join(f"{k}: {log_dict[k][-1]}" for k in keys))


def save_img(generator: nn.Module, out_dir_path: Path, device):
    try:
        out_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save(engine):
        z = np.random.uniform(size=(1, Z_DIM)).astype(np.float32)
        z = torch.from_numpy(z).to(device)

        generator.eval()
        with torch.no_grad():
            x = generator(z).detach().cpu().numpy().squeeze() * 255

        p = out_dir_path / f"out_iter_{engine.state.iteration:05d}.png"
        cv2.imwrite(str(p), x.astype(np.uint8))

    return _save


if __name__ == "__main__":
    parse_args()
