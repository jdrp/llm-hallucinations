import csv
import json
import argparse
import random


def choose_one_answer(qa_pair: dict, accuracy: float, include_context: bool, dataset: str) -> dict:
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
    qa_test['hallucination'] = hallucinate
    return qa_test


def generate_test(qa_data: list[dict], accuracy: float, include_context: bool, dataset: str) -> list[dict]:
    test_data = []
    for row in qa_data:
        test_data.append(choose_one_answer(row, accuracy, include_context, dataset))
    return test_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Random Answer Selection')
    parser.add_argument('--data', default=None, help='Original JSON file with right and hallucinated answers')
    parser.add_argument('--outfile', default=None, help='Output JSON file')
    parser.add_argument('--accuracy', type=float, default=0.6, help='Proportion of right answers to be chosen (0 to 1)')
    parser.add_argument('--context', default='y', help='Include context field? (y/n)')
    args = parser.parse_args()

    if not args.data:
        raise ValueError('Please add the input --data argument')
    outfile = args.outfile if args.outfile else '_test.'.join(args.data.rsplit('.', 1))
    include_context = (args.context[0].lower() == 'y')

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
        json.dump(generate_test(qa_data, args.accuracy, include_context, dataset), f)


if __name__ == '__main__':
    main()
