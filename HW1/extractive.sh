#!/usr/bin/evn bash
python3.6 predict.py --predict extractive --load_model ./model/extractive.hdf5 --test_data_path $1 --output_path $2
