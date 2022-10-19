# Earthformer Training on SEVIR&SEVIR-LR
## SEVIR
Run the following command to train Earthformer on SEVIR dataset. 
Change the configurations in [cfg_sevir.yaml](./cfg_sevir.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 2 --cfg ./scripts/cuboid_transformer/sevir/cfg_sevir.yaml --ckpt_name last.ckpt --save tmp_sevir
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 2 --cfg ./scripts/cuboid_transformer/sevir/cfg_sevir.yaml --pretrained --save tmp_sevir
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_sevir/lightning_logs --name 'tmp_sevir'
```
## SEVIR-LR
Run the following command to train Earthformer on SEVIR-LR dataset. 
Change the configurations in [cfg_sevirlr.yaml](./cfg_sevirlr.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 2 --cfg ./scripts/cuboid_transformer/sevir/cfg_sevirlr.yaml --ckpt_name last.ckpt --save tmp_sevirlr
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_sevirlr/lightning_logs --name 'tmp_sevirlr'
```
