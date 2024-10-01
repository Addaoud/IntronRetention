# The role of chromatin state in intron retention: a case study in leveraging large scale deep learning models

<img src="https://github.com/Addaoud/IntronRetention/blob/main/model.PNG" width="640">

## Introduction
This repository is a companion page for a research paper submitted to Biorxiv (https://www.biorxiv.org/content/10.1101/2024.01.26.577402v2.full). It provides code and experiments to demonstrate the value of foundation models of chromatin state for accurate prediction of human introns that are subject to intron retention (IR).

## Manuscript
Ahmed Daoud, and Asa Ben-Hur. "The role of chromatin state in intron retention: a case study in leveraging large scale deep learning models." bioRxiv (2024): 2024-01. [https://www.biorxiv.org/content/10.1101/2024.01.26.577402v2.full](https://www.biorxiv.org/content/10.1101/2024.01.26.577402v2.full)

## Installation
You can clone this repository using the command:
```bash
git clone https://github.com/Addaoud/IntronRetention.git
```

## Dependencies
1) You can create a python virtual environment or an anaconda environment to install the dependencies using pip or conda:
```bash
pip install -r requirements.txt
conda install -r requirements.txt
```
2) You need to install a pytorch version compatible with your cuda version. You can follow the steps in [here](https://pytorch.org/) to install the latest pytorch version or you can refer to [previous versions](https://pytorch.org/get-started/previous-versions/) to install an older pytorch version. 

3) Additionally, you need to install gpu-version of LightGBM using pip:
```bash
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```
If you don't intend to run LightGBM on a gpu, you can install the cpu-version using pip and modify the device in the LGBM.py script to "cpu".

## Datasets
The file [data/final_data.csv](https://github.com/Addaoud/IntronRetention/blob/main/data/final_data.csv) contains the data that was used to produce all the results reported in the paper. 

## Getting started
Refer to the [Sei repository](https://github.com/FunctionLab/sei-framework) to download the pre-trained [Sei framework](https://zenodo.org/records/4906997) and the [target.names](https://github.com/FunctionLab/sei-framework/blob/main/model/target.names) file. Please put both files in the main directory. You can simply use the following commands:
```bash
cd IntronRetention
wget https://zenodo.org/records/4906997/files/sei_model.tar.gz?download=1 .
tar -xvzf sei_model.tar.gz 
wget https://raw.githubusercontent.com/FunctionLab/sei-framework/main/model/target.names .
```

Follow the next sections to reproduce the results.

#### FSEI
  * Refer to "json/FSei.json", to update paths and hyperparameters if necessary.
  * You can follow the usage guide to train and evaluate the Fine-tuned Sei. You can either build a new model (the first python command line) or load an existing one (the second python command line):
```bash
usage: FSei.py [-h] [--json JSON] [-n] [-m MODEL] [-p] [-f] [-t] [-e]

Train and evaluate the FSei model

options:
  -h, --help            show this help message and exit
  --json JSON           Path to the json file
  -n, --new             Use this option to build a new FSei model
  -m MODEL, --model MODEL
                        Use this option to load an existing FSei model from model_path
  -p, --pretrain        Use this option to load Sei pretrained weights
  -f, --freeze          Use this option to freeze Sei pretrained weights. This option should be used with -p
  -t, --train           Use this option to train the model
  -e, --evaluate        Use this option to evaluate the model

python3 FSei.py --json "json/FSei.json" -n -p -t -e
python3 FSei.py --json "json/FSei.json" -m "FSei_model_path" -t -e
```
  
  * To obtain the Integrated Grandients using the FSei model, run the attribution script using the python command line:
```bash
usage: Attribute.py [-h] [-i] [-b] [-w WINDOW] [-t THRESHOLD] [-p PREDICTION] [-d DATABASE] [-m MODEL_PATH]

Run IG and compare hot regions with motifs in Jaspar database

options:
  -h, --help            show this help message and exit
  -i, --integrate       Run integrated gradients
  -b, --bind            Compare hot regions with TF binding sites in the database
  -w WINDOW, --window WINDOW
                        Use this option to set the IG window size. Sequences of length IG window size centered around hot spots will be compared with TFs motifs.
  -t THRESHOLD, --threshold THRESHOLD
                        Use this option to set the IG threshold. Only hot spots above this threshold will be selected to be compared with TFs motifs.
  -p PREDICTION, --prediction PREDICTION
                        Use this option to set the prediction threshold. Only sequences correctly predicted above this threshold using the model will be selected
  -d DATABASE, --database DATABASE
                        Jaspar database path
  -m MODEL_PATH, --model_path MODEL_PATH
                        FSei model path

python3 Attribute.py -m "FSei_model_path" -i -b
```
Results are saved in a subdirectory "IG" under the model directory path.

#### Basenji-like, AttentionConv, Basset-like model
  * Refer to "json/Bassenji.json", "json/AttentionConv.json" , or "json/Basset.json" to update paths and hyperparameters if necessary.
  * You can train from scratch, the Bassenji-like, AttentionConv, or Basset-like models using respectively the following command lines:
```bash
python3 Basenji.py --json "json/Basenji.json" -n -t -e
python3 AttnConv.py --json "json/AttentionConv.json" -n -t -e
python3 Basset.py --json "json/Basset.json" -n -t -e
```

#### DNABert2
  * Refer to "json/DNABert.json", to update paths and hyperparameters if necessary.
  * You can finetune the DNABert-2 model using the following command line:
```bash
python3 DNABert.py --json "json/DNABert.json" -n -t -e
```

#### Logistic Regression and LightGBM
  *If you want to run the Logistic Regression or LightGBM models, you need to process the data to numpy files, using [Preprocess_data_to_numpy.py](https://github.com/Addaoud/IntronRetention/blob/main/Preprocess_data_to_numpy.py). It is used to apply Sei to the DNA sequences, and save the targets in numpy files. The files are used subsequently as input to the Logistic Regression or LightGBM models. Use the following command line:
```bash
python3 Preprocess_data_to_numpy.py --data "data" --result "data/numpy"
``` 
  * Refer to "json/LR.json" or "json/LGBM.json" to update paths and hyperparameters if necessary.
  * You can train the Logistic Regression using the following command lines to build a new model or load an existing one:
```bash
python3 LR.py --json "json/LR.json" -n -t -e
python3 LR.py --json "json/LR.json" -m "existing_model_path" -t -e
```
  * You can optimize and train the LGBM model using the following command line to build a new model:
```bash
python3 LGBM.py --json "json/LGBM.json" --optimize
```

## Contributing
Contributions to this repository are welcome! If you find any bugs, have suggestions for new features, or want to improve the existing code, please create an issue or submit a pull request. You can post in the Github issues or e-mail Ahmed Daoud (ahmed0daoudad@gmail.com)