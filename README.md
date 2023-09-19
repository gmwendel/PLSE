# *PLSE*

Photoelectron Light Signal Estimator
A tool to generate neural networks that classify the number of photoelectrons in a PMT waveform. 




## Installation

First we must set up an anaconda environment with tensorflow installed.  


Create a new anaconda environment and install tensorflow ~ 2.11 prerequisites:
```
conda deactivate
conda create -n plse python=3.9
conda activate plse 

```

Install tensorflow using the following [instructions](https://www.tensorflow.org/install/pip).

Once an anaconda environment with tensorflow is properly configured, install PLSE:
```
git clone https://github.com/gmwendel/PLSE.git
cd PLSE
pip install -e .
```
Verify the scripts have been added to the anaconda environment:
```
plse_train -h
```

## Basic Usage

### Training
Train a PLSE network by running:

```
plse_train mywaveformfile.npz -n mynetwork
```

glob can be used to specify multiple waveform files e.g. mywaveformfiles*.npz

Note: a simple training example is included in test.py and can be run by uncommenting line 4
### Evaluating
Test PLSE network evaluation by running a pre-trained example network and generating a confusion matrix:

```
python3 plse/example/test.py
```

## Data Extraction

Install ratpac-two and source the proper environmental variables
run 
```
python3 tools/rat2npz.py -h
```
for more info
