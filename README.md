# Neural-Network-and-Deep-Learning-pj2

Course project for CIFAR-10 classification and Batch Normalization analysis.

## Links

- GitHub repository: <https://github.com/jiab666/Neural-Network-and-Deep-Learning-pj2>
- Model weights: <https://drive.google.com/drive/folders/1x8BPofWcaTG0FGQ4PiEKDlDMwgU1n_vf?usp=drive_link>
- Dataset: <https://www.cs.toronto.edu/~kriz/cifar.html>

## Project Summary

- CIFAR-10 classification with VGG-style models
- Comparison of `VGG_A`, `VGG_A_BatchNorm`, and `VGG_A_Dropout`
- Optimizer comparison with `SGD`, `SGD with momentum`, and `Adam`
- Loss-landscape and gradient-stability analysis for BN
- Final report written in standalone LaTeX under `report_tex/`

## Main Files

- `report_tex/project_2_2026.tex`: main report source
- `report_tex/project_2_2026.pdf`: compiled report
- `report_tex/bib.bib`: bibliography
- `codes/VGG_BatchNorm/models/vgg.py`: model definitions
- `codes/VGG_BatchNorm/VGG_Loss_Landscape.py`: experiment runner

## Quick Start

Run a small verification experiment:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --epochs 1 --train-subset 64 --val-subset 64 --batch-size 32 --landscape-lrs 0.001 0.0005 --landscape-models vgg_a vgg_a_bn --num-workers 0
```

Run both BN landscape analysis and optimizer comparison:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --mode landscape optimizers --epochs 5 --train-subset 2000 --val-subset 1000 --optimizer-models vgg_a vgg_a_bn vgg_a_dropout
```

Run the final full-data training with resumable checkpoints:

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --mode optimizers --epochs 40 --batch-size 128 --train-subset -1 --val-subset -1 --optimizer-models vgg_a vgg_a_bn --optimizers adam --optimizer-lr 0.0001 --resume
```

Compile the LaTeX report:

```bash
cd report_tex
latexmk -pdf -interaction=nonstopmode project_2_2026.tex
```

Generated figures, metrics, and checkpoints are written to `codes/VGG_BatchNorm/outputs/`.
