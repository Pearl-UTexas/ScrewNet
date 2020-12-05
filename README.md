# ScrewNet
Code for the "**ScrewNet: Category-Independent Articulation Model Estimation From Depth Images Using Screw Theory**" 
paper. Full paper available [here](https://arxiv.org/abs/2008.10518). [[Project webpage]()]  

## Instruction to run the code

### Install prerequisites and environment
```commandline
cd /path/to/the/repository/
conda env create -f environment.yaml
conda activate screwNet
```

### Download datasets and pretrained model weights
* Evaluation datasets: [Link](https://drive.google.com/file/d/1ot5U2KW-gwarPX-qLiHdSJYau0KQNsVk/view?usp=sharing)
* Pretrained weights: [Link](https://drive.google.com/file/d/1rz07tlapadc2D65ro02RhgO2Aqn4wO6L/view?usp=sharing)

### Run the evaluation code
#### Test
```commandline
python evaluate_model.py --model-dir <pretrained-model-dir> --model-name <model-name> --test-dir <test-dir-name> --model-type <screw, l2, noLSTM, 2imgs> --output-dir <output-dir>
```

#### [Optional] Visualization on jupyter notebook
* run ```jupyter notebook```
* open visualize_results notebook
* update evaluation directories (same as the output directory used for the evaluate_model.py script)
* run corresponding cells

### [Optional] Training on custom dataset
* Generate dataset using our fork of the Synthetic articulated dataset generator from [here](https://github.com/jainajinkya/SyntheticArticulatedData)
* Run the following command to train ScrewNet on the generated datasets
```commandline
python train_model.py --name <model-name> --train-dir <training-dataset-dir> --test-dir <test-dataset-dir> --ntrain <no_of_training_samples> --ntest <no_of_validation_samples> --epochs <no_epochs> --cuda --batch <batch-size> --device 0 --fix-seed
```