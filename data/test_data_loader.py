import numpy as np
import pandas as pd
# Changed import from data_loader to test
from data_loader import TSDataLoader, TimeSeriesDataset
import matplotlib.pyplot as plt


def test_dataloader(data_path, seq_len=336, pred_len=96, target_column=None):
    """Test the TSDataLoader with temporal splitting."""
    print(f"Testing dataloader with seq_len={seq_len}, pred_len={pred_len}")

    # Initialize the data loader
    data_loader = TSDataLoader(
        data_csv_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=32,
        train_val_split=True,
        val_ratio=0.2,
        target_column=target_column
    )

    # Get the data loaders
    train_loader, val_loader = data_loader.get_data_loaders()

    # Print dataset sizes
    print(f"Train dataset size: {len(data_loader.train_dataset)}")
    print(f"Validation dataset size: {len(data_loader.val_dataset)}")

    # Check a few batches
    print("\nExamining train batches:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")

        # Check if time features are included in the batch
        if len(batch) == 2:
            x_batch, y_batch = batch
            print(f"  Input shape: {x_batch.shape}")
            print(f"  Target shape: {y_batch.shape}")
            # Changed to show first timestep, all features
            print(f"  Input sample: {x_batch[0, 0, :]}")
            # Changed to show first timestep, all features
            print(f"  Target sample: {y_batch[0, 0, :]}")
        else:
            x_batch, y_batch, x_mark, y_mark = batch
            print(f"  Input shape: {x_batch.shape}")
            print(f"  Target shape: {y_batch.shape}")
            print(f"  Input time features shape: {x_mark.shape}")
            print(f"  Target time features shape: {y_mark.shape}")
            # Changed to show first timestep, all features
            print(f"  Input sample: {x_batch[0, 0, :]}")
            # Changed to show first timestep, all features
            print(f"  Target sample: {y_batch[0, 0, :]}")
            # First timestep time features
            print(f"  Input time features sample: {x_mark[0, 0, :]}")

        # Break after a few batches
        if i >= 2:
            break

    return train_loader, val_loader


def visualize_batch(train_loader):
    """Visualize a batch from the data loader."""
    # Get a batch
    batch = next(iter(train_loader))

    if len(batch) == 2:
        x_batch, y_batch = batch
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot a few samples from the batch
        for i in range(min(3, x_batch.shape[0])):
            # Plot input sequence
            ax.plot(np.arange(x_batch.shape[1]), x_batch[i, :, 0].numpy(),
                    label=f'Input Seq {i}')

            # Plot target sequence with offset
            ax.plot(np.arange(x_batch.shape[1], x_batch.shape[1] + y_batch.shape[1]),
                    y_batch[i, :, 0].numpy(), '--', label=f'Target Seq {i}')

        ax.set_title('Sample Time Series Sequences')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig('sample_batch.png')
        print("Visualization saved as 'sample_batch.png'")
    else:
        x_batch, y_batch, x_mark, y_mark = batch

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot time series data
        for i in range(min(3, x_batch.shape[0])):
            # Plot input sequence
            ax1.plot(np.arange(x_batch.shape[1]), x_batch[i, :, 0].numpy(),
                     label=f'Input Seq {i}')

            # Plot target sequence with offset
            ax1.plot(np.arange(x_batch.shape[1], x_batch.shape[1] + y_batch.shape[1]),
                     y_batch[i, :, 0].numpy(), '--', label=f'Target Seq {i}')

        ax1.set_title('Sample Time Series Sequences')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.legend()

        # Plot time features
        time_feature_names = ['hour', 'day', 'month', 'year', 'dayofweek']
        colors = ['r', 'g', 'b', 'c', 'm']

        # Only plot first sample's time features
        for j, name in enumerate(time_feature_names):
            if j < x_mark.shape[2]:  # Check if this feature exists
                ax2.plot(np.arange(x_mark.shape[1]), x_mark[0, :, j].numpy(),
                         color=colors[j], label=name)

        ax2.set_title('Time Features for First Sample')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Feature Value')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('sample_batch_with_time_features.png')
        print("Visualization saved as 'sample_batch_with_time_features.png'")


if __name__ == "__main__":
    # Replace with your actual dataset path
    # Adjust this path as needed
    data_path = "./data/data_files/electricity/electricity.csv"

    # Test with different sequence lengths and prediction lengths
    train_loader, val_loader = test_dataloader(
        data_path, seq_len=336, pred_len=96)

    # Visualize a batch
    visualize_batch(train_loader)
