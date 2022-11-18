# Earthformer Training on ICAR-ENSO dataset 
Run the following command to train Earthformer on ICAR-ENSO dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_enso.py --gpus 2 --cfg cfg.yaml --ckpt_name last.ckpt --save tmp_enso
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_enso.py --gpus 2 --pretrained --save tmp_enso
```
Run the tensorboard command to upload experiment records
```bash
tensorboard dev upload --logdir ./experiments/tmp_enso/lightning_logs --name 'tmp_enso'
```
