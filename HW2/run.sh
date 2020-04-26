#!/usr/bin/evn bash
python3.6 predict.py --mode test --load_model ./model/final.pth --test_data_path $1 --output_path $2