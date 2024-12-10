import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import pytorch_lightning.tuner as tuner
import matplotlib.pyplot as plt

from omsimulator.data.dataloader import BaseDataModule
from omsimulator.model.model import PMTNetwork

def main():
    # Model parameters
    model_dict = {
        "num_conv_layers": 3,
        "num_filters": 150,
        "num_lin_layers": 3,
        "linear_features": 150,
        "num_global_features": 5,
        "mse_weight": 0.0,
        "logs_path": "/scratch/tmp/fvaracar/train_omclassfier/test_train"
    }
    
    model = PMTNetwork(
        num_conv_layers=model_dict["num_conv_layers"],
        num_filters=model_dict["num_filters"],
        num_lin_layers=model_dict["num_lin_layers"],
        linear_features=model_dict["linear_features"],
        num_global_features=model_dict["num_global_features"],
        mse_weight=model_dict["mse_weight"]
    )

    # Load data
    my_dir = BaseDataModule(
        "/scratch/tmp/fvaracar/geant_h5_files/first_run/sim_1",
        batch_size=1000,
        val_batch_size=256,
        num_workers=8
    )

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Monitor 'train_loss' if that's what you're interested in
        mode='min',            # 'min' for loss, 'max' for accuracy
        save_top_k=1,
        verbose=True
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback]
    )

    # Initialize Tuner for learning rate finding
    tuner_instance = tuner.Tuner(trainer)

    # Run Learning Rate Finder
    lr_find_results = tuner_instance.lr_find(
    model,
    datamodule=my_dir,
    min_lr=1e-5,
    max_lr=1,
    early_stop_threshold=None,
    attr_name='learning_rate'  # Specify the attribute name
)

    # Plot learning rate finder results and save
    fig = lr_find_results.plot(suggest=True)
    plt.savefig("lr_fig.png")
    suggested_lr = lr_find_results.suggestion()
    print(f"Suggested learning rate: {suggested_lr}")

    # Update model learning rate
    model.hparams.lr = suggested_lr

    # Start training
    trainer.fit(model, my_dir)

if __name__ == "__main__":
    main()
