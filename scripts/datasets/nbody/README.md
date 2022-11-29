# N-body MNIST
The underlying dynamics in the N-body MNIST dataset is governed by the Newton's law of universal gravitation:

$\frac{d^2\boldsymbol{x}\_{i}}{dt^2} = - \sum\_{j\neq i}\frac{G m\_j (\boldsymbol{x}\_{i}-\boldsymbol{x}\_{j})}{(\|\boldsymbol{x}\_i-\boldsymbol{x}\_j\|+d\_{\text{soft}})^r}$

where $\boldsymbol{x}\_{i}$ is the spatial coordinates of the $i$-th digit, $G$ is the gravitational constant, $m\_j$ is the mass of the $j$-th digit, $r$ is a constant representing the power scale in the gravitational law, $d\_{\text{soft}}$ is a small softening distance that ensures numerical stability.

To download the N-body MNIST dataset used in our paper from AWS S3 run:
```bash
cd ROOT_DIR/earth-forecasting-transformer
python ./scripts/datasets/nbody/download_nbody_paper.py
```

Alternatively, run the following commands to generate N-body MNIST dataset.
```bash
cd ROOT_DIR/earth-forecasting-transformer
python ./scripts/datasets/nbody/generate_nbody_dataset.py --cfg ./scripts/datasets/nbody/cfg.yaml
```
