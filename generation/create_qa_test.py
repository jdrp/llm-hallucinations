"""This script takes in a JSON list of objects where each object contains a question with (context), correct and
hallucinated answers. With the chosen accuracy, the script will choose a random answer for each question,
and add a 'hallucination' field to tell whether it is hallucinated."""

import csv
import json
import argparse
import random


def choose_one_answer(qa_pair: dict, accuracy: float, include_context: bool, include_samples: bool, dataset: str) -> dict | None:
    qa_test = {}
    hallucinate = 1 if random.random() > accuracy else 0
    if dataset == 'halueval':
        if 'knowledge' in qa_pair.keys() and include_context:
            qa_test['context'] = qa_pair['knowledge']
        qa_test['question'] = qa_pair['question']
        qa_test['answer'] = qa_pair['hallucinated_answer'] if hallucinate else qa_pair['right_answer']
    elif dataset == 'truthfulqa':
        qa_test['question'] = qa_pair['Question']
        good_answers = qa_pair['Correct Answers'].split(';')
        hallucinated_answers = qa_pair['Incorrect Answers'].split(';')
        qa_test['answer'] = random.choice(hallucinated_answers) if hallucinate else random.choice(good_answers)
        if include_samples:
            qa_test['samples'] = [sample for sample in (hallucinated_answers if hallucinate else good_answers)
                                  if sample != qa_test['answer']]
            if not qa_test['samples']:
                return None
    qa_test['hallucination'] = hallucinate
    return qa_test


def generate_test(qa_data: list[dict], accuracy: float, include_context: bool, include_samples: bool, dataset: str) -> list[dict]:
    test_data = []
    for row in qa_data:
        if new_row := choose_one_answer(row, accuracy, include_context, include_samples, dataset):
            test_data.append(new_row)
    print(len(test_data))
    return test_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None, help='Original JSON file with right and hallucinated answers')
    parser.add_argument('--outfile', default=None, help='Output JSON file')
    parser.add_argument('--accuracy', type=float, default=0.6, help='Proportion of right answers to be chosen (0 to 1)')
    parser.add_argument('--context', default=True, action=argparse.BooleanOptionalAction, help='Include context field? (y/n)')
    parser.add_argument('--samples', default=False, action=argparse.BooleanOptionalAction, help='Include extra sampled responses')
    args = parser.parse_args()

    if not args.data:
        raise ValueError('Please add the input --data argument')
    outfile = args.outfile if args.outfile else f"{args.data.rsplit('.',1)[0]}_test{'_samples' if args.samples else ''}.json"

    with open(args.data, 'r') as f:
        if args.data.endswith('.json'):
            qa_data = json.loads('[' + ','.join(f.read().splitlines()) + ']')  # read as a JSON array
            dataset = 'halueval'
        elif args.data.endswith('.csv'):
            qa_data = list(csv.DictReader(f))
            dataset = 'truthfulqa'
        else:
            raise ValueError('Please enter a valid --data file (json or csv)')

    with open(outfile, 'w') as f:
        json.dump(generate_test(qa_data, args.accuracy, args.context, args.samples, dataset), f)


if __name__ == '__main__':
    main()
