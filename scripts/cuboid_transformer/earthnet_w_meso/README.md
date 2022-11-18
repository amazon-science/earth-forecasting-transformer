# Earthformer Training on EarthNet2021 with auxiliary meso scale data
Run the following command to train Earthformer on EarthNet2021 dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_earthnet.py --gpus 2 --cfg cfg.yaml --ckpt_name last.ckpt --save tmp_earthnet_w_meso
```

Or run the following command to directly load pretrained checkpoint for test.
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_earthnet.py --gpus 2 --pretrained --save tmp_earthnet_w_meso
```
Run the tensorboard command to upload experiment records
```bash
tensorboard dev upload --logdir ./experiments/tmp_earthnet_w_meso/lightning_logs --name 'tmp_earthnet_w_meso'
```
