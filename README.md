# Convolutional Neural Tangent Kernel (CNTK)

This repository contains the code for Convolutional Neural Tangent Kernel (CNTK) in the following paper

[On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/abs/1904.11955) (NeurIPS 2019)

### Citation

```bib
@inproceedings{arora2019exact,
  title={On exact computation with an infinitely wide neural net},
  author={Arora, Sanjeev and Du, Simon S. and Hu, Wei and Li, Zhiyuan and Salakhutdinov, Ruslan and Wang, Ruosong},
  booktitle={Thirty-third Conference on Neural Information Processing Systems},
  year={2019}
}
```

## Usage

Requires [Python](https://www.python.org/downloads/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

1. Install [CuPy](https://docs.cupy.dev/en/stable/install.html), [SciPy](https://scipy.org/install/) and [tqdm](https://github.com/tqdm/tqdm#installation)
2. Download CIFAR-10

```sh
wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xzvf -
```

3. Parallelize [Line 122-124](/blob/master/CNTK.py#L122) in [CNTK.py](/blob/master/CNTK.py) according to your specific computing environment to utilize multiple GPUs

### Reproduce Results (Paper Table 1)

For column CNTK-V:

```sh
python CNTK.py --gap no --fix no --depth <depth>
```

> *where \<depth\> is 3, 4, 6, 11 or 21*

For column CNTK-GAP:

```sh
python CNTK.py --gap yes --fix yes --depth <depth>
```

> *where \<depth\> is 3, 4, 6, 11 or 21*
