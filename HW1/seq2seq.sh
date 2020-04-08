#!/usr/bin/evn bash
python3.6 predict.py --predict abstractive_without_att --load_model ./model/abstractive_without_att.hdf5 --test_data_path $1 --output_path $2
