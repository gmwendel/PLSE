# *PLSE*

Photoelectron Light Signal Estimator.
A tool to generate neural networks that classify the number of photoelectrons in a PMT waveform. 


## Installation

It is highly recommended PLSE be installed in a new anaconda environment.  


Create a new anaconda environment:
```
conda deactivate
conda create -n plse_env python=3.12
conda activate plse_env
```

Next, install PLSE:
```
git clone https://github.com/gmwendel/PLSE.git
cd PLSE
pip install -e .
```
By default, this will install the CPU version of TensorFlow.

To install the GPU version of TensorFlow, use the following command:
```
pip install -e .[gpu]
```

Verify the scripts have been added to the anaconda environment:
```
plse_train -h
```

## Basic Usage

### Training
Train a PLSE network by running:

```
plse_train mywaveformfile.npz -o mynetwork
```

glob can be used to specify multiple waveform files e.g. mywaveformfiles*.npz

Note: a simple training example is included in test.py and can be run by uncommenting line 4
### Evaluating
Test PLSE network evaluation by running a pre-trained example network and generating a confusion matrix:

```
plse_evaluate mywaveformfile.npz -m mynetwork/model.keras
```

## Data Extraction

Install ratpac-two and source the proper environmental variables
run 
```
python3 tools/rat2npz.py -h
```
for more info
