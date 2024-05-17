"""This script analyzes the results of hallucination prediction to extract relevant information such as
true or false positives and negatives, accuracy..."""

import json
import argparse


def classify_predictions(pred_data: list[dict], pred_key: str, true_key: str) -> tuple[list, list, list, list]:
    tp = []; tn = []; fp = []; fn = []
    for row in pred_data:
        if row[pred_key] == row[true_key] == 1:
            tp.append(row)
        elif row[pred_key] == row[true_key] == 0:
            tn.append(row)
        elif row[pred_key] == 1 and row[true_key] == 0:
            fp.append(row)
        elif row[pred_key] == 0 and row[true_key] == 1:
            fn.append(row)
        else:
            print(f'Error on row {row}')
            raise ValueError('Hallucination predictions/ground truth must be 0 or 1')
    return tp, tn, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hallucination Detection Results")
    parser.add_argument('--data', default=None, help='JSON file with predictions and ground truth')
    parser.add_argument('--predkey', default='hallucination_pred', help='JSON key for the prediction')
    parser.add_argument('--truekey', default='hallucination', help='JSON key for ground truth')
    parser.add_argument('--plot', default='y', help='Plot results? (y/n)')
    args = parser.parse_args()

    if not args.data:
        raise ValueError('Please input a --data file')
    plot_results = (args.plot[0].lower() == 'y')

    with open(args.data, 'r') as f:
        pred_data = json.load(f)
        tp, tn, fp, fn = classify_predictions(pred_data, args.predkey, args.truekey)
        print(f'TP: {len(tp)}\nTN: {len(tn)}\nFP: {len(fp)}\nFN: {len(fn)}')


if __name__ == '__main__':
    main()
