import argparse
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
import pytorch_lightning as pl
from omsimulator.data.dataloader import BaseDataModule
from omsimulator.model.model import PMTNetwork
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train PMTNetwork")
    parser.add_argument("--num_conv_layers", type=int, default=3, help="Number of convolutional layers")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters in convolutional layers")
    parser.add_argument("--num_lin_layers", type=int, default=2, help="Number of linear layers")
    parser.add_argument("--linear_features", type=int, default=50, help="Number of features in linear layers")
    parser.add_argument("--num_global_features", type=int, default=5, help="Number of global features")
    parser.add_argument("--mse_weight", type=float, default=0.01, help="Weight of MSE loss")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Path of ckpt file for retraining.")
    parser.add_argument("--add_name", type=str, default=None, help="Additional name for saving the files.")
    parser.add_argument("--wv_global", action="store_true", help="Define wavelength as a global value (default: False)")
    parser.add_argument("--QE", action="store_true", help="Include QE directly in the traninig (default: False)")
    parser.add_argument("--conv_map", action="store_true", help="Map output using a conv layer. If not, a linear layer with weight sharing will be used (default: False)")
    parser.add_argument("--global_features", action="store_true", help="Include global features in the training. Helpful when there are symmetry breaking elements like PAC (default: False)")
    
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Fill model_dict with parsed arguments or use defaults
    model_dict = {
        "num_conv_layers": args.num_conv_layers,
        "num_filters": args.num_filters,
        "num_lin_layers": args.num_lin_layers,
        "linear_features": args.linear_features,
        "num_global_features": args.num_global_features,
        "mse_weight": args.mse_weight,
        "wv_global": args.wv_global,
        "ckpt_file": args.ckpt_file,
        "QE": args.QE,
        "conv_map": args.conv_map,
        "global_features": args.global_features
    }
    if args.add_name is not None:
        model_dict["logs_path"]=f"/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_{args.mse_weight}_wv_global_{str(args.wv_global)}_{args.add_name}"
    else:
        model_dict["logs_path"]=f"/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_{args.mse_weight}_wv_global_{str(args.wv_global)}"
    # Saving model config
    with open(f"{model_dict['logs_path']}.yaml", "w") as file:
        yaml.dump(model_dict,file)
        
    print(f"Training with following settings: {model_dict}")
    dir_path = f"/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_{args.mse_weight}"
    os.makedirs(dir_path, exist_ok=True)
    # Initialize the model
    model = PMTNetwork(
        num_conv_layers=model_dict["num_conv_layers"],
        num_filters=model_dict["num_filters"],
        num_lin_layers=model_dict["num_lin_layers"],
        linear_features=model_dict["linear_features"],
        num_global_features=model_dict["num_global_features"],
        mse_weight=model_dict["mse_weight"],
        QE=model_dict["QE"],
        conv_map=model_dict["conv_map"],
        global_features=model_dict["global_features"],
        wv_global = model_dict["wv_global"]
    )
    
    # Restore model weigths for re-training
    if model_dict["ckpt_file"] is not None:
        state_dict = model_dict["ckpt_file"]
        ckpt = torch.load(state_dict)
        model.load_state_dict(ckpt["state_dict"])  

    # Load data
    my_dir = BaseDataModule(
        "/scratch/tmp/fvaracar/geant_h5_files/second_run/",
        batch_size=10000,
        val_batch_size=256,
        QE=True,#model_dict["QE"],
        num_workers=8
    )

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Monitor training loss
        mode='min',            # Minimize loss
        save_top_k=1,          # Save only the best model
        verbose=True
    )

    # Initialize Trainer with specified logging path and profiler
    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=model_dict["logs_path"],
        enable_progress_bar=False,
        profiler=AdvancedProfiler()
    )

    # Train the model
    trainer.fit(model, my_dir,ckpt_path=model_dict["ckpt_file"])

if __name__ == "__main__":
    main()
