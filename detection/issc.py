"""Iterative Sentence-Sample Comparison"""

import json
import argparse
import random
import time
import pprint
import re

from rich.progress import track
import nltk
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


# nltk.download('punkt')


def contrast_sentence(model: str, prompt_template: str, sentence: str, sample: str) -> float:
    delete_history()
    prompts = [prompt.replace('[C]', sample).replace('[S]', sentence)
               for prompt in prompt_template.split('\n[next]\n')]
    response = ''
    log('PROMPTS')
    for prompt in prompts:
        log('>> ' + prompt)
    log('RESPONSES')
    for prompt in prompts:
        response = prompt_model(model, prompt)
        log('>> ' + response)
        response = re.sub('[^a-zA-Z0-9]', '', response.split('\n')[-1])
    verdict = 0. if response.startswith('VerdictYes') or response.startswith('VerdictTrue') \
        else 1. if response.startswith('VerdictNo') or response.startswith('VerdictFalse') \
        else 0.5
    return verdict


def contrast_response(model: str, prompt_template: str, row: dict) -> float:
    sentences = nltk.sent_tokenize(row['answer'])
    log('--------------------')
    log(f'HALLUCINATION {row['hallucination']}')
    inconsistencies = []
    for sentence in sentences:
        inconsistencies.append(np.average([contrast_sentence(model, prompt_template, sentence, sample) for sample in row['samples']]))

    prediction = float(np.average(inconsistencies))
    log(f'HALLUCINATION {row['hallucination']}')
    log(f'PREDICTION {prediction}')
    return prediction


def evaluate_data(model: str, prompt_template: str, data: list[dict]) -> list[dict]:
    evaluated_data = []
    for row in track(data):
        row_copy = row.copy()
        row_copy['hallucination_pred'] = contrast_response(model, prompt_template, row_copy)
        evaluated_data.append(row_copy)
    return evaluated_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Hallucination Detection - ISSC')
    parser.add_argument('--data', default=None,
                        help='JSON file containing list of question, answer, samples')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of rows to be processed (for testing)', )
    parser.add_argument('--model', default='llama3',
                        help='Ollama model used for detecting hallucination')
    parser.add_argument('--prompt', default='prompts/detection/detect_samples_sentences_noq_agnostic_cot.prompt',
                        help='Prompt template file')
    parser.add_argument('--outfile', default=None,
                        help='Output JSON file')
    parser.add_argument('--logfile', default=None, help='Path to log file')
    args = parser.parse_args()
    if not args.data:
        raise ValueError('Please add the input --data argument')
    if args.model not in (available_models := get_available_models()):
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    outfile = args.outfile if args.outfile else f"_selfcheck_{args.model.replace(':', '-')}{'_cot' if args.cot else ''}.".join(args.data.rsplit('.', 1))

    with open(args.prompt, 'r') as f:
        prompt_template = f.read()

    set_logfile(args.logfile)
    log('=================================================')
    log('Contrast samples')
    start_time = time.time()
    log(f'Current time {start_time:.2f}')
    log(pprint.pformat(vars(args)))
    log('')

    with open(args.data, 'r') as f:
        qa_data = json.load(f)
    if args.limit:
        qa_data = random.sample(qa_data, args.limit)
    with open(outfile, 'w') as f:
        json.dump(evaluate_data(args.model, prompt_template, qa_data), f, indent=4)
    log(f'Finished after {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()
