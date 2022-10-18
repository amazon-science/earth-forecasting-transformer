# Test Persistence on EarthNet2021
Run the following command to test Persistence on EarthNet2021 dataset. 
Change the configurations in [corresponding cfg.yaml](./cfg.yaml)
```bash
cd ROOT_DIR/earthformer
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/baselines/persistence/earthnet/test_persistence_earthnet.py --gpus 2 --cfg ./scripts/baselines/persistence/earthnet/cfg.yaml --save tmp_earthnet_persistence
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/earthformer
tensorboard dev upload --logdir ./experiments/tmp_earthnet/lightning_logs --name 'tmp_earthnet'
