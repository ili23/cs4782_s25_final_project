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

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

def main():
    print("Starting PatchTST training...")
    # Hyperparameters
    seq_len = 336
    pred_len = 96
    patch_length = 16
    depth = 3  # Reduced from original
    batch_size = 128  # Increased batch size
    dropout = 0.2  # Added dropout
    weight_decay = 1e-2  # Added L2 regularization
    learning_rate = 1e-3  # Reduced learning rate

    small = False
    if small:
        ff_dim = 64  # Reduced complexity
        num_heads = 4
        embed_dim = 16
    else:
        ff_dim = 128  # Reduced from 256
        num_heads = 8  # Reduced from 16
        embed_dim = 64  # Reduced from 128

    feature_map = {
        'ETT': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'electricity': [str(i) for i in range(322)] + ['OT'],
        'traffic': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'illness': ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT'],
    }
    features = feature_map['ETT']
    size = [seq_len, pred_len]

    print("Creating data loader...")
    dataloader = TSDataLoader(
        "./data/data_files/electricity/electricity.csv", 
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
        dropout=dropout,  # Added dropout
        dataset=train_dataloader.dataset
    )

    print("PatchTST model created.")

    output_dir = "./models/results"
    logs_output_dir = "./models/results/logs"
    
    # Initialize trainer with regularization
    model_trainer = PatchTSTTrainer(
        patch_tst, 
        output_dir, 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='patchtst-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_output_dir)

    trainer = L.Trainer(
        max_epochs=100,  # Increased max epochs since we have early stopping
        check_val_every_n_epoch=1,  # Increased validation frequency
        num_sanity_val_steps=2,  # Added sanity checks
        logger=tb_logger,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        gradient_clip_val=0.5,  # Added gradient clipping
        accumulate_grad_batches=2  # Added gradient accumulation
    )

    start_time = time.time()
    print("Starting training...")
    trainer.fit(model_trainer, train_dataloader, val_dataloader)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Load best model for testing
    model_trainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model_trainer, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()