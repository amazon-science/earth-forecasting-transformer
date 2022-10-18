# Earthformer Training on EarthNet2021 with auxiliary meso scale data
Run the following command to train Earthformer on EarthNet2021 dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/earthnet_w_meso/train_cuboid_earthnet.py --gpus 2 --cfg ./scripts/cuboid_transformer/earthnet_w_meso/cfg.yaml --ckpt_name last.ckpt --save tmp_earthnet_w_meso
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/earthnet_w_meso/train_cuboid_earthnet.py --gpus 2 --cfg ./scripts/cuboid_transformer/earthnet_w_meso/cfg.yaml --pretrained --save tmp_earthnet_w_meso
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_earthnet_w_meso/lightning_logs --name 'tmp_earthnet_w_meso'
```
