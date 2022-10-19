# Earthformer Training on ICAR-ENSO dataset 
Run the following command to train Earthformer on ICAR-ENSO dataset. 
Change the configurations in [cfg.yaml](./cfg.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/enso/train_cuboid_enso.py --gpus 2 --cfg ./scripts/cuboid_transformer/enso/cfg.yaml --ckpt_name last.ckpt --save tmp_enso
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/enso/train_cuboid_enso.py --gpus 2 --cfg ./scripts/cuboid_transformer/enso/cfg.yaml --pretrained --save tmp_enso
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_enso/lightning_logs --name 'tmp_enso'
```
