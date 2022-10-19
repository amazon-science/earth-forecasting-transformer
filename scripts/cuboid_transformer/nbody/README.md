# Earthformer Training on N-body MNIST
Run the following command to train Earthformer on N-body MNIST dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/nbody/train_cuboid_nbody.py --gpus 2 --cfg ./scripts/cuboid_transformer/nbody/cfg.yaml --ckpt_name last.ckpt --save tmp_nbody
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_nbody/lightning_logs --name 'tmp_nbody'
```
