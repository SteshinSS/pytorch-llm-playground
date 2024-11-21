# LLM Example
## Description


## Installation
```
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
```


## Data Download
```
python download_data.py
```

## Training
```
python -m llm.training.train_vanilla
```


## Development
To add package:
```
vim requirements.in  # add the package
pip-compile
pip-sync
```