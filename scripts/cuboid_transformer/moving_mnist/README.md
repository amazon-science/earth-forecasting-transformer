# Earthformer Training on Moving MNIST
Run the following command to train Earthformer on Moving MNIST dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/moving_mnist/train_cuboid_mnist.py --gpus 2 --cfg ./scripts/cuboid_transformer/moving_mnist/cfg.yaml --ckpt_name last.ckpt --save tmp_mnist
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_mnist/lightning_logs --name 'tmp_mnist'
```
