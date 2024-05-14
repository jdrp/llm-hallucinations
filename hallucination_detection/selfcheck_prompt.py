import json
import argparse
from rich.progress import track
import nltk
import numpy as np

from ..utils import *


# nltk.download('punkt')


def contrast_sentence(model: str, sentence: str, sample: str, use_cot: bool) -> int:
    delete_history()
    with open('prompts/self_check' + ('_cot' if use_cot else '') + '.txt', 'r') as f:
        template = f.read()
    prompts = [prompt.replace('[C]', sample).replace('[S]', sentence)
               for prompt in template.split('\n[next]\n')]
    response = ''
    for prompt in prompts:
        response = prompt_model(model, prompt)
        print(response)

    return 1 if response.startswith('Yes') else 0 if response.startswith('No') else 0.5


def contrast_response(model: str, response: str, samples: list[str], use_cot: bool) -> float:
    sentences = nltk.sent_tokenize(response)
    inconsistencies = []
    for sentence in sentences:
        inconsistencies.append(np.average([contrast_sentence(model, sentence, sample, use_cot) for sample in samples]))
    print(inconsistencies)
    return np.average(inconsistencies)


def evaluate_data(model: str, data: list[dict], sensitivity: float, use_cot: bool) -> list[dict]:
    evaluated_data = []
    for row in track(data):
        row_copy = row.copy()
        row_copy['hallucination_pred'] = 1 if contrast_response(model, row['answer'], row['samples'],
                                                                use_cot) >= sensitivity else 0
        evaluated_data.append(row_copy)
    return evaluated_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Hallucination Detection - SelfCheckPrompt')
    parser.add_argument('--data', default=None,
                        help='JSON file containing list of question, answer, samples')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of rows to be processed (for testing)', )
    parser.add_argument('--model', default='llama3',
                        help='Ollama model used for detecting hallucination')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Average evaluation threshold to classify as hallucination')
    parser.add_argument('--cot', default='y',
                        help='Choose whether to use chain-of-thought when prompting (y/n)')
    parser.add_argument('--outfile', default=None,
                        help='Output JSON file')
    args = parser.parse_args()
    if not args.data:
        raise ValueError('Please add the input --data argument')
    if args.model not in (available_models := get_available_models()):
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    outfile = args.outfile if args.outfile else f'_selfcheck_{args.model}.'.join(args.data.rsplit('.', 1))
    use_cot = (args.cot[0] == 'y')

    with open(args.data, 'r') as f:
        qa_data = json.load(f)
    with open(outfile, 'w') as f:
        json.dump(evaluate_data(args.model, qa_data[:args.limit], args.sensitivity, use_cot), f)


if __name__ == '__main__':
    main()
