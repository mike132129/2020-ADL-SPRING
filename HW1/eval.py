import os

for i in range(20, 90, 5):
    print(i, 'epoch used..')
    os.system('python3 predict.py --load_model ./model/seq2seq-att-save-model-' + str(i) + '.hdf5 --test_data_path ./data/valid.jsonl --predic abstractive_with_att --output_path ./valid_result.jsonl')
    os.system('python3 scripts/scorer.py valid_result.jsonl ./data/valid.jsonl')

