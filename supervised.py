import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import time

from models.architecture.assembled import AssembledModel
from models.train_loop import PatchTSTTrainer
from data.data_loader import TSDataLoader

random.seed(2021)
torch.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.deterministic = True

def main():
    print("Starting PatchTST training...")
    name = "illness"
    depth = 3   # Reduced from original
    lr = 1e-4
    dropout = 0.3
    if name == "illness":
        seq_len = 48
        pred_len = 12
        patch_length = 16
        batch_size = 32
        small = True
        csv = "./data/data_files/illness/national_illness.csv"
    elif name == "ETT":
        seq_len = 336
        pred_len = 96
        patch_length = 16
        batch_size = 64
        small = True
        csv = "./data/data_files/ETT-small/ETTh1.csv"
    elif name == "ETTm1":
        seq_len = 336
        pred_len = 96
        patch_length = 16
        batch_size = 64
        small = True
        csv = "./data/data_files/ETT-small/ETTm1.csv"
    elif name == "electricity":
        seq_len = 336
        pred_len = 96
        patch_length = 16
        batch_size = 64
        small = False
        csv = "./data/data_files/electricity/electricity.csv"
    elif name == "traffic":
        seq_len = 336
        pred_len = 96
        patch_length = 16
        batch_size = 32
        small = False
        csv = "./data/data_files/traffic/traffic.csv"

    if small:
        ff_dim = 64  # Reduced complexity
        num_heads = 4
        embed_dim = 16
    else:
        ff_dim=256
        num_heads=16
        embed_dim=128

    feature_map ={
        'ETT': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'ETTm1': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'electricity': [str(i) for i in range(320)] + ['OT'],
        'traffic': [str(i) for i in range(861)] + ['OT'],
        'illness': ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT'],
    }
    features = feature_map[name]
    size = [seq_len, pred_len]

    print("Creating data loader...")
    # dataloader = TSDataLoader("./data/data_files/ETT-small/ETTh1.csv", batch_size=batch_size, size=size)
    dataloader = TSDataLoader(
        csv, 
        features=features, 
        batch_size=batch_size, 
        size=size
    )
    train_dataloader, val_dataloader, test_dataloader = dataloader.get_data_loaders()

    patch_tst = AssembledModel(
        patch_length=patch_length, 
        depth=depth, 
        seq_len=seq_len, 
        pred_len=pred_len, 
        ff_dim=ff_dim, 
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        dropout=dropout,
    )

    print("PatchTST model created.")

    output_dir = "./models/results/"
    logs_output_dir = "./models/results/logs/"
    model_trainer = PatchTSTTrainer(patch_tst, output_dir, lr=lr)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_output_dir)

    trainer = L.Trainer(max_epochs=250,
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=0,
                        logger=tb_logger,
                        # accumulate_grad_batches=2,
                        # gradient_clip_val=1.0,
                        )

    start_time = time.time()
    print("Starting training...")
    trainer.fit(model_trainer, train_dataloader, val_dataloader)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Load best model for testing
    # model_trainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model_trainer, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()