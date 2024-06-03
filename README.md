# MIMIC-CLIP
An implementation of CLIP  for generating radiology reports on MIMIC Dataset. It was a test implementation so some documentation is missing

## Text Processing
The folder `text_processing` contains sample code for processing the radiology reports and findings. The radiology reports are extracted and compiled in files `data_list_p10_*.txt`. The `p10` refers to the folder in MIMIC with the same name. 
- `word2vec_Cbow.py`: Trains a continuous bag of words model. 

## Training CLIP Model
The code contains several implementations I tried for training a CLIP model
- `retrain_bert_multigpu.py`: Contains code for training a CLIP model on multiple GPUS using PyTorch's Distributed Data Parallel scheme.

