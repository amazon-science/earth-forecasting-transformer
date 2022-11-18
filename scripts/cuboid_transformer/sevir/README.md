# Earthformer Training on SEVIR&SEVIR-LR
## SEVIR
Run the following command to train Earthformer on SEVIR dataset. 
Change the configurations in [cfg_sevir.yaml](./cfg_sevir.yaml)
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_sevir.py --gpus 2 --cfg cfg_sevir.yaml --ckpt_name last.ckpt --save tmp_sevir
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_sevir.py --gpus 2 --pretrained --save tmp_sevir
```
Run the tensorboard command to upload experiment records
```bash
tensorboard dev upload --logdir ./experiments/tmp_sevir/lightning_logs --name 'tmp_sevir'
```
## SEVIR-LR
Run the following command to train Earthformer on SEVIR-LR dataset. 
Change the configurations in [cfg_sevirlr.yaml](./cfg_sevirlr.yaml)
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_sevir.py --gpus 2 --cfg cfg_sevirlr.yaml --ckpt_name last.ckpt --save tmp_sevirlr
```
Run the tensorboard command to upload experiment records
```bash
tensorboard dev upload --logdir ./experiments/tmp_sevirlr/lightning_logs --name 'tmp_sevirlr'
```
