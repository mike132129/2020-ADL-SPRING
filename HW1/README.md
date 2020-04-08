

To predict the result of the test data.

1. download the model, pretrained embedding and tokenizer
``` 
bash download.sh
```
2. check the requirement:
```
bash install_packages.sh
```
3. predict the result
extractive prediction:
```
bash extractive.sh /path/to/test.jsonl /path/to/predict.jsonl
```
seq2seq prediction
```
bash seq2seq.sh /path/to/test.jsonl /path/to/predict.jsonl
```
seq2seq with attention
```
bash attention.sh /path/to/test.jsonl /path/to/predict.jsonl
```

To train the model:
1. Put the train.jsonl in ./data/
2. Train the model
Train extractive model:
```
python3.6 train.py --train extractive
```

Train seq2seq model:
```
python3.6 train.py --train abstractive_without_attention
```

Train seq2seq with attention model:
```
python3.6 train.py --train abstractive_with_attention
```

To plot the figure in report questions:
1. Put the valid.jsonl in ./data/
2. plot relative locations:
```
python3.6 plot.py --plot relative_locations
```

3. attention weight visualization:
```
python3.6 plot.py --plot attention
```






