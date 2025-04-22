# OMNNSimulator

This project uses machine learning to simulate the response of multi-PMT optical modules.
The Neural Network models are trained on Geant4 simulations which are very accurate but slow.


Information about scripts in OMClassifier:


# torch_trazing_model.py :

This script converts a model into a format that can be used and deployed in C++

# OMSimC++Test :

Test how to use and deploy a model in C++

# convert_hdf5.py :

Script that converts Geant4 output .txt files into hdf5 format suitable for training

# main.py :

Script that can be used for the training of Neural Network models

# eff_area.py

Script that can be used to calculate optical module's effective areas with the Neural Network models. It is usually an useful check.

# src/omsimulator/data/dataloader.py

Dataloader used for training

# src/omsimulator/model/model.py

Neural Network model

# src/omsimulator/model/model_torch.py

Neural Network model used by torch_trazing_model. It does not inherit from torch_ligthning
