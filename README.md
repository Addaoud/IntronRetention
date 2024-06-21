# The role of chromatin state in intron retention: a case study in leveraging large scale deep learning models

<img src="https://github.com/Addaoud/IntronRetention/blob/main/model.PNG" width="640">

## Introduction
This repository is a companion page for a research paper submitted to the 18th Machine Learning in Computational Biology conference [MLCB-2023](https://sites.google.com/cs.washington.edu/mlcb2023/). It provides code and experiments to demonstrate the value of foundation models of chromatin state for accurate prediction of human introns that are subject to intron retention (IR). We used two approaches for leveraging the [Sei](https://github.com/FunctionLab/sei-framework) architecture for predicting retained introns: The Fine-tuned Sei version uses the representation learned by the Sei convolutional layers, adding a convolutional block plus pooling and fully connected and output layers. Alternatively, the Sei-outputs version uses the chromatin targets learned by Sei (transcription factor binding, histone modifications, and chromatin accessibility) and uses these as input to a logistic regression or light-GBM classifier. Furthermore, this repository uses a novel approach to score the relevance of each transcription factor in the Jaspar Database in intron retention, using Integrated Gradients (IG). The original data was obtained from [here](https://github.com/fahadahaf/chromir).

## Manuscript
Ahmed Daoud, and Asa Ben-Hur. "Using large scale transfer learning to highlight the role of chromatin state in intron retention." bioRxiv (2024): 2024-01. [https://www.biorxiv.org/content/10.1101/2024.01.26.577402v1](https://www.biorxiv.org/content/10.1101/2024.01.26.577402v1)

## Installation
You can clone this repository using the command:
```
git clone https://github.com/Addaoud/IntronRetention.git
```

## Dependencies
You can create a python virtual environment or an anaconda environment to install the dependencies using pip or conda:
```
pip install -r requirements.txt
conda install -r requirements.txt
```
Additionally, you need to install a pytorch version compatible with your cuda version. You can follow the steps in [here](https://pytorch.org/) to install the latest pytorch version or you can refer to [previous versions](https://pytorch.org/get-started/previous-versions/) to install an old pytorch version. 

## Getting started
  * Follow the steps in [here](https://github.com/FunctionLab/sei-framework) to download the pre-trained Sei framework and the target.names file. Please put both files in the main directory.
  * [Preprocess_data_to_numpy.py](https://github.com/Addaoud/IntronRetention/blob/main/Preprocess_data_to_numpy.py) can be used to map the DNA sequences to Sei targets, and save the results in numpy files. This data can be used subsequently as input to the Logistic Regression or Light GBM models. You can use the following command line:
```
python3 Preprocess_data_to_numpy.py --data directory_which_contains_the_final_data_csv --result directory_where_to_save_the_numpy_files
```
  * You can train the Logistic Regression by building a new model or loading an existing one using the following command lines:
```
python3 LR.py --json path_to_LR_json -n -t -e
python3 LR.py --json path_to_LR_json -m existing_model_path -t -e
```
  * You can train the LGBM model (and optionally perform a Bayesian optimization to search for the best hyperparameters) using the following command lines:
```
python3 LGBM.py --json path_to_LGBM_json --optimize
```
  * You can train the Fine-tuned Sei by building a new model or loading an existing one using the following command lines:
```
python3 FSei.py --json path_to_FSei_json -n -t -e
python3 FSei.py --json path_to_FSei_json -m existing_model_path -t -e
```
  * You can also finetune the DNABert-2 model or train, from scratch, the Basset or ConvNet models using the same command lines by changing the appropriate python script and corresponding json file
  * You can run the attribution script on a fine-tuned FSei model using the following command line:
```
python3 Attribute.py -m existing_model_path -b -i -r name_of_IG_folder
```
## Contributing
Contributions to this repository are welcome! If you find any bugs, have suggestions for new features, or want to improve the existing code, please create an issue or submit a pull request. You can post in the Github issues or e-mail Ahmed Daoud (ahmed0daoudad@gmail.com)