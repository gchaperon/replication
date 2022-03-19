# Pointer Networks
My replication code for the paper [Pointer Networks](https://arxiv.org/abs/1506.03134).

# The data
The official data is hosted at http://goo.gl/NDcOIG. It can be downloaded using `gdown`
```bash
$ gdown --folder --output data https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
```
or manually using the web interface.

Alternatively, since gdown has sometimes failed me, I'm hosting it myself (hopefully it's not illegal).
You can download it like so
```bash
$ wget -qO- https://users.dcc.uchile.cl/~gchapero/datasets/ptr-nets-data.tar.gz | tar -C data -xzv
```

In any case, the data dir should look like this
```
data
├── convex_hull_10_test.txt
├── convex_hull_200_test.txt
├── convex_hull_500_test.txt.zip
├── convex_hull_50_test.txt
├── convex_hull_50_train.txt
├── convex_hull_5-50_train.txt.zip
├── ...
└── tsp_5_train.zip
```

# Notes on data
## Train
Training splits with data files correspondance

| train split         | file                         |
|---------------------|------------------------------|
| `n=5 (optimal)`     | `tsp_5_train.zip/tsp5.txt`   |
| `n=10 (optimal)`    | `tsp_10_train_exact.txt`     |
| `n=50 (a1 trained)` | `tsp_50_train.zip`           |
| `n=50 (a3 trained)` | `???`                        |
| `n=5-20 (optimal)`  | `tsp_5-20_train.zip/*`       |

Some files were determined by inspecting the average tour distance. In particular, there are 2 `tsp_10_*` train/test pairs and only one `tsp_50_*` train/test pair.

## Test
Data files correspondance with Table 2 of the paper

| n                 | optimal                         | a1                                | a2    | a3    |
|-------------------|---------------------------------|-----------------------------------|-------|-------|
| 5                 | `tsp_5_train.zip/tsp5_test.txt` | `???`                             | `???` | `???` |
| 10                | `tsp_10_test_exact.txt`         | `tsp_10_train.zip/tsp10_test.txt` | `???` | `???` |
| 50 (a1 trained)   | `na`                            | `tsp_50_test.txt.zip`             | `???` | `???` |
| 50 (a3 trained)   | `na`                            | `same`                            | `???` | `???` |
| 5 (5-20 trained)  | `same`                          | `???`                             | `???` | `???` |
| 10 (5-20 trained) | `same`                          | `???`                             | `???` | `???` |
| 20 (5-20 trained) | `???`                           | `tsp_20_test.txt`                 | `???` | `???` |
| 25 (5-20 trained) | `na`                            | `???`                             | `???` | `???` |
| 30 (5-20 trained) | `na`                            | `???`                             | `???` | `???` |
| 40 (5-20 trained) | `na`                            | `tsp_40_test.txt`                 | `???` | `???` |
| 50 (5-20 trained) | `na`                            | `same`                            | `???` | `???` |

`na` means it shouldn't exist, `???` means it should exist but isn't found in `./data` and `same` means its the same file from a previous row.

Again, some were determined by inspecting the average tour distance.

# Changes
Here is the list of changes I did to the original paper description

* Optimizer: SGD to Adam. I obtained similar results with both but Adam converged faster.

# Differences with https://github.com/devsisters/pointer-network-tensorflow
1. el input (la secuencia de puntos) lo pasa por una capa lineal, ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L104
2. el estado inicial del encoder es entrenable parece, ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L116
3. el primer vector sobre el que se hace atencion (bos token o <= en el paper, `first_decoder_input` en el codigo) tambien es entrenable parece, ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L124
4. la entrada al decoder es una permutacion de los **estados escondidos** del encoder (incluyendo el `first_decoder_input`) en vez de una permutacion de los **inputs del encoder** (vectores de largo 2 representando puntos), ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L133
5. parece que tanto el bos token (=> en el paper) y el eos token (<= en el paper) son el mismo vector `first_decoder_input`, ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L141
6. por alguna razon cuando reordena los estados escondidos del encoder para obtener el input al decoder usa la funcion `tf.stop_gradient` (que entiendo que funciona como `torch.Tensor.detach` en pytorch), ver https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/model.py#L133 y https://github.com/devsisters/pointer-network-tensorflow/blob/f3956d6786c9d401d100e4c09a248b5d185d09fb/layers.py#L71
