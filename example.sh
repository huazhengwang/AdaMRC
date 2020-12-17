#!/usr/bin/env bash


#Adaptation from SQuAD to NewsQA with ground-truth questions
python train.py --data_dir ../data/multitask/ --train_datasets squad,newsqa --dev_datasets newsqa,squad --not_update_decoder newsqa  

#Semi supervised learning setting
python train_semi.py  --data_dir ../data/multitask/  --covec_path ../data/MT-LSTM.pt  --train_datasets newsqa,squad --dev_datasets squad,newsqa --not_update_decoder newsqa --semi_dataset newsqa --semi_ratio 0.5 

