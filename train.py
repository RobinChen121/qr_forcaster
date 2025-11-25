import os
import nni
import torch
import pickle
from argparse import Namespace
import pytorch_lightning as pl

from config import quantiles
from model import ForecasterQR
from arguments import get_params
from DatasetHandler import DatasetHandler

import sys

data_name = "LD2011_2014.txt"
if sys.platform == 'win32':
    data_path = os.path.join('D:/chenzhen/data/', data_name)
else:
    data_path = os.path.join('/Users/zhenchen/Documents/machine learning data/', data_name)
TRAINED_MODEL_PATH = os.path.join("trained_models")
DATALOADERS_PATH = os.path.join("dataloaders")


def main(args):
    forking = args.use_forking_sequences
    forking_total_seq_length = 500 if forking else None
    train_el_dataloader, val_el_dataloader = (DatasetHandler(
        data_path,
        num_samples=args.dataset_num_samples,  # default is 1000
        hist_hours=args.max_sequence_len,  # default is 24*7 hours
        pred_horizon=args.forcast_horizons,  # default is next 24 hours
        batch_size=args.batch_size,  # with forking, use lower batch size!
        forking_total_seq_length=forking_total_seq_length)
                                              .load_dataset())

    # save dataloaders for predictions
    # exist_ok=True 表示如果目录已经存在则不报错
    os.makedirs(DATALOADERS_PATH, exist_ok=True)
    train_dl_path = os.path.join(DATALOADERS_PATH, "train_dl.pkl")
    test_dl_path = os.path.join(DATALOADERS_PATH, "test_dl.pkl")
    with open(train_dl_path, "wb") as fp:
        pickle.dump(train_el_dataloader, fp)
    with open(test_dl_path, "wb") as fp:
        pickle.dump(val_el_dataloader, fp)

    # quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    # quantiles = [.2, .4, .5, .6, .8] # quantile is in the config.yml

    model = ForecasterQR(
        x_dim=3,
        y_dim=4,
        input_max_squence_len=args.max_sequence_len, # 7*24 is the max time sequence
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_layer_count,
        decoder_context_dim=args.decoder_context_dim,
        quantiles=quantiles,
        horizons=args.forcast_horizons,
        device="gpu",
        init_learning_rate=args.learning_rate, # default is 5e-2
        init_weight_decay=args.weight_decay,
        sequence_forking=forking is not None
    )

    # model checkpoint callback
    # 保存最小损失的参数权重配置
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        # accelerator=args.gpus,  # the parameter in the old version is 'gpus'
        accelerator=accelerator,  # 优先使用 GPU
        devices=1,
        max_epochs=2, #args.epochs,
        # checkpoint_callback=checkpoint_cb,
        callbacks=[checkpoint_cb],  # 所有回调放在 callbacks 列表里
        num_sanity_val_steps=0)

    # 输出``Trainer.fit` stopped: `max_epochs=2` reached.`
    # 是 PyTorch Lightning 的 Trainer 自带日志，
    # 也就是 Trainer.fit() 方法运行结束时打印的提示信息。
    trainer.fit(model, train_el_dataloader, val_el_dataloader)
    val_loss = trainer.callback_metrics["val_loss"].item()
    # nni.report_final_result() 的作用是 单向上传结果给 NNI 调参平台，
    # 不是保存到本地、也不是给下次训练用的
    # nni.report_final_result({"default": val_loss})


if __name__ == '__main__':
    try:
        # get parameters from tuner
        namespace_params = get_params()
        if namespace_params.use_nni:
            print("nni activated.")
            tuner_params = nni.get_next_parameter()
            params = vars(namespace_params)
            print("TUNER PARAMS: " + str(tuner_params))
            print("params:" + str(params))
            params.update(tuner_params)
            namespace_params = Namespace(**params)
        main(namespace_params)
    except Exception as ex:
        torch.cuda.empty_cache()
        raise
