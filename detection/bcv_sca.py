"""Basic Contextual Verification (with or without CoT) or Sentence-level Contextual Analysis"""

import json
import argparse
import random
import time
import pprint

from rich.progress import track
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


def generate_prompts(template: str, question: str, answer: str, context: str | None = '') -> list[str]:
    prompts = template.split('\n[next]\n')
    prompts = [(prompt.replace('[C]', context) if context else prompt)
               .replace('[Q]', question)
               .replace('[A]', answer) for prompt in prompts]
    return prompts


def evaluate_qa_pair(model: str, prompt_template: str, qa_pair: dict, times: int,
                     threshold: float) -> float:
    question = qa_pair.get('question')
    answer = qa_pair.get('answer')
    context = qa_pair.get('context')
    prompts = generate_prompts(prompt_template, question, answer, context)
    log('--------------------')
    log(f'HALLUCINATION {qa_pair['hallucination']}')
    log('PROMPTS')
    for prompt in prompts:
        log('>> ' + prompt)
    log('RESPONSES')
    responses = []
    for _ in range(times):
        delete_history()
        response = ''
        for prompt in prompts:
            response = prompt_model(model, prompt).strip()
            log('>> ' + response)
        log('---')
        response = re.sub('[^a-zA-Z0-9]', '', response.split('\n')[-1])
        verdict = 1. if response.startswith('VerdictYes') or response.startswith('VerdictTrue') \
            else 0. if response.startswith('VerdictNo') or response.startswith('VerdictFalse') \
            else 0.5
        responses.append(verdict)
        log(f'VERDICT {verdict}')

    prediction = sum(responses) / len(responses)
    log(f'HALLUCINATION {qa_pair['hallucination']}')
    log(f'PREDICTION {prediction}')
    return (1. if prediction >= threshold else 0.) if threshold != -1. else prediction


def evaluate_qa_data(model: str, prompt_template: str, qa_data: list[dict], times: int, threshold: float) -> list[dict]:
    evaluated_qa_data = []
    for qa_pair in track(qa_data):
        qa_copy = qa_pair.copy()
        qa_copy['hallucination_pred'] = evaluate_qa_pair(model, prompt_template, qa_copy, times, threshold)
        evaluated_qa_data.append(qa_copy)
    return evaluated_qa_data


def main() -> None:
    random.seed = 50
    available_models = get_available_models()
    parser = argparse.ArgumentParser(description='Hallucination Detection')
    parser.add_argument('--data', default=None,
                        help='JSON file containing list of (context), question, answer')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of rows to be processed (for testing)', )
    parser.add_argument('--random', default=False, action=argparse.BooleanOptionalAction, help='Sample random rows')
    parser.add_argument('--model', default='llama3',
                        help=f"Ollama model used for detecting hallucination. Options: {', '.join(available_models)}")
    parser.add_argument('--times', type=int, default=2, help='Number of times to evaluate each answer')
    parser.add_argument('--threshold', type=float, default=-1,
                        help='Average evaluation threshold to classify as hallucination')
    parser.add_argument('--prompt', default='prompts/detection/detect_context_sentences_q_agnostic_cot.prompt',
                        help='Prompt template file')
    parser.add_argument('--outfile', default=None,
                        help='Output JSON file')
    parser.add_argument('--logfile', default=None, help='JSON log file to track responses')
    args = parser.parse_args()
    if not args.data:
        raise ValueError('Please add the input --data argument')
    if args.model not in available_models:
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    outfile = args.outfile if args.outfile else f"_{args.model.replace(':', '-')}.".join(args.data.rsplit('.', 1))

    with open(args.prompt, 'r') as f:
        prompt_template = f.read()

    set_logfile(args.logfile)
    log('=================================================')
    log('Detect hallucinations')
    start_time = time.time()
    log(f'Current time {start_time:.2f}')
    log(pprint.pformat(vars(args)))
    log('')

    # start up ollama
    delete_history()
    prompt_model(args.model, 'Hello')
    delete_history()

    with open(args.data, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    qa_data = random.sample(qa_data, args.limit) if args.random else qa_data[:args.limit]
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(evaluate_qa_data(args.model, prompt_template, qa_data, args.times, args.threshold), f, indent=4)
    log(f'Finished after {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()
