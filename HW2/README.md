To predict the classification result:

1. Download the model:
```
bash download.sh
```

2. Predict result for the classification question
```
bash run.sh /path/to/test.json /path/to/output_result.json
```

To fine-tune the bert model:

0. Put train.json, dev.json in ./data/ 

1. Data preprocessing
```
python3.6 makedata.py --dataset train
python3.6 makedata.py --dataset dev
```
2. Start training
```
python3 train.py --mode train
```


To plot for question 5, 6:

0. Put train.json, dev.json in ./data/,and put the evaluate.py and eval model in ./script/
1. To plot the answer length figure
```
bash plot.sh --plot answer_length
```
2. To plot the answerable threshold
```
bash plot.sh --plot answerable_threshold
```

