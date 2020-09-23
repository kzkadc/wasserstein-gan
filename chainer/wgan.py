# coding: utf-8

from model import Critic, Generator
from pathlib import Path
import pprint
import numpy as np
import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions

import matplotlib
matplotlib.use("Agg")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="WGAN(-GP)")
    parser.add_argument("-b", "--batchsize", type=int, default=100)
    parser.add_argument("-e", "--epoch", type=int, default=100)
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


def main(args):
    chainer.config.user_gpu = args.g
    if args.g >= 0:
        chainer.backends.cuda.get_device_from_id(args.g).use()
        print("GPU mode")

    mnist_train = chainer.datasets.get_mnist()[0]  # MNISTデータ取得
    mnist_train = chainer.dataset.concat_examples(mnist_train)[0]  # 画像だけ（ラベルは不要）
    mnist_train = mnist_train.reshape((-1, 1, 28, 28))  # 画像形式(N,C,H,W)に整形
    mnist_iter = iterators.SerialIterator(mnist_train, args.batchsize, shuffle=True, repeat=True)  # iteratorを作成

    generator = Generator(Z_DIM)
    critic = Critic()
    if args.g >= 0:
        generator.to_gpu()
        critic.to_gpu()

    opt_g = optimizers.Adam(args.alpha, args.beta1, args.beta2)
    opt_g.setup(generator)
    opt_c = optimizers.Adam(args.alpha, args.beta1, args.beta2)
    opt_c.setup(critic)

    updater = WGANUpdater(mnist_iter, opt_g, opt_c, args.n_cri, args.gp_lam)
    trainer = Trainer(updater, (args.epoch, "epoch"), out=args.result_dir)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(["epoch", "generator/loss", "critic/loss"]))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(("generator/loss", "main/wdist"), "epoch", file_name="loss_plot.eps"))
    trainer.extend(extensions.snapshot_object(generator, "model_epoch_{.updater.epoch}.model"), trigger=(10, "epoch"))
    trainer.extend(ext_save_img(generator, args.result_dir + "/out_images"))

    trainer.run()


# 生成画像を保存するextension
def ext_save_img(generator, out):
    out_path = Path(out)
    try:
        out_path.mkdir(parents=True)
    except FileExistsError:
        pass

    @chainer.training.make_extension(trigger=(1, "epoch"))
    def _ext_save_img(trainer):
        z = generator.xp.random.uniform(size=(1, Z_DIM)).astype(np.float32)
        with chainer.using_config("train", False):
            img = generator(Variable(z)).array * 255
        img = img.reshape((28, 28)).astype(np.uint8)
        if chainer.config.user_gpu >= 0:
            img = generator.xp.asnumpy(img)

        p = out_path / "out_epoch_{:03d}.png".format(trainer.updater.epoch)
        cv2.imwrite(str(p), img)

    return _ext_save_img


Z_DIM = 30


class WGANUpdater(updaters.StandardUpdater):
    def __init__(self, iterator, gen_opt, cri_opt, n_cri, gp_lam, **kwds):
        opts = {
            "gen": gen_opt,
            "cri": cri_opt
        }
        self.n_cri = n_cri
        self.gp_lam = gp_lam
        super().__init__(iterator, opts, **kwds)

    def update_core(self):
        gen_opt = self.get_optimizer("gen")
        cri_opt = self.get_optimizer("cri")
        generator = gen_opt.target
        critic = cri_opt.target
        batch_size = self.get_iterator("main").batch_size

        # バッチ（本物）を取得
        x_real = self.get_iterator("main").next()
        x_real = Variable(np.stack(x_real))
        if chainer.config.user_gpu >= 0:
            x_real.to_gpu()

        xp = x_real.xp

        # update critic
        upd_num = self.n_cri[1] if self.iteration <= 25 or self.iteration % 500 == 0 else self.n_cri[0]
        for i in range(upd_num):
            z = xp.random.uniform(size=(batch_size, Z_DIM)).astype(np.float32)
            x_fake = generator(Variable(z))

            cri_loss = F.average(critic(x_fake) - critic(x_real))  # Wasserstein距離の逆符号

            # gradient penalty
            eps = xp.random.uniform(size=(batch_size, 1, 1, 1)).astype(np.float32)
            x_fusion = eps * x_real + (1 - eps) * x_fake  # (N,1,H,W)
            g_critic = chainer.grad([critic(x_fusion)], [x_fusion], enable_double_backprop=True)[0]  # (N,1,H,W)
            gp = F.batch_l2_norm_squared(g_critic)
            gp = F.average((F.sqrt(gp) - 1)**2)
            total_loss = cri_loss + self.gp_lam * gp

            critic.cleargrads()
            total_loss.backward()
            cri_opt.update()

        # update generator
        z = xp.random.uniform(size=(batch_size, Z_DIM)).astype(np.float32)
        x_fake = generator(Variable(z))
        gen_loss = -F.average(critic(x_fake))

        generator.cleargrads()
        critic.cleargrads()
        gen_loss.backward()
        gen_opt.update()

        chainer.report({
            "generator/loss": gen_loss,
            "critic/loss": cri_loss,
            "main/wdist": -cri_loss
        })


if __name__ == "__main__":
    parse_args()
