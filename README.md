# Wasserstein GAN(-GP)
[Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." International Conference on Machine Learning. 2017.](http://proceedings.mlr.press/v70/arjovsky17a.html)  
[Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in Neural Information Processing Systems. 2017.](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans)
 
Implemented with [Chainer](https://chainer.org)

## Requirements
Chainer, OpenCV

```bash
$ pip install chainer opencv-python
```

## How to run
```bash
$ python wgan.py [options]
```

You can read help with `-h` option.

```bash
$ python wgan.py -h
usage: wgan.py [-h] [-b BATCHSIZE] [-e EPOCH] [--alpha ALPHA] [--beta1 BETA1]
               [--beta2 BETA2] [--n_cri N_CRI N_CRI] [--gp_lam GP_LAM] [-g G]
               [--result_dir RESULT_DIR]

WGAN(-GP)

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchsize BATCHSIZE
  -e EPOCH, --epoch EPOCH
  --alpha ALPHA
  --beta1 BETA1
  --beta2 BETA2
  --n_cri N_CRI N_CRI
  --gp_lam GP_LAM
  -g G                  GPU ID (negative value indicates CPU mode)
  --result_dir RESULT_DIR
```
