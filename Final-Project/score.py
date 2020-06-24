import csv
import argparse
import unicodedata
import re
import pdb
import sys
import codecs

def normalize_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag


def score(ref_file, pred_file):
    with codecs.open(ref_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        ref_data = list(reader)

    with codecs.open(pred_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        pred_data = list(reader)

    f_score = 0.0
    for ref_row, pred_row in zip(ref_data, pred_data):

        assert ref_row['ID'] == pred_row['ID']
        refs = set(ref_row["Prediction"].split())
        preds = set(pred_row["Prediction"].split())



        p = len(refs.intersection(preds)) / len(preds) if len(preds) > 0 else 0.0
        r = len(refs.intersection(preds)) / len(refs) if len(refs) > 0 else 0.0
        f = 2*p*r / (p+r) if p + r > 0 else 0
        f_score += f

        if preds != {'NONE'} or refs != {'NONE'}:
            print(preds, refs, f)

    return f_score / len(ref_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file")
    parser.add_argument("pred_file")

    args = parser.parse_args()

    s = score(args.ref_file, args.pred_file)
    print(s)
