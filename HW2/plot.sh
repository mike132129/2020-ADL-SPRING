#!/usr/bin/evn bash

if [ $1 == "answer_length" ]
then
    echo "plot answer length\n take about 13 mins to compile";
    python3.6 utils.py --test_data_path ./data/train.json --output_path result.json --mode test --load_model ./model/final.pth --plot answer_length
else
    echo "plot answerable threshold"
    for i in {1..10..2}
    do 
        ten=10
        threshold=`awk -v x=$i -v y=$ten 'BEGIN {threshold=x/y; print threshold}'`
        echo predict threshold = $threshold
        python3.6 predict.py --mode test --test_data_path ./data/dev.json --load_model ./model/final.pth --output_path result-$threshold.json --threshold $threshold
        python3 scripts/evaluate.py ./data/dev.json result-$threshold.json score-$threshold.json ./scripts/data/
    done

    python3.6 utils.py --plot answerable_threshold
fi
