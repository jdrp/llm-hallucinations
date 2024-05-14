import json
import argparse
from rich.progress import track

from utils import *


def generate_prompts(template: str, question: str, answer: str, context: str = '') -> list[str]:
    with open(template, 'r') as f:
        prompts = f.read().split('\n[next]\n')
    prompts = [prompt.replace('[C]', context).replace('[Q]', question).replace('[A]', answer) for prompt in prompts]
    return prompts


def evaluate_qa_pair(model: str, question: str, answer: str, context: str | None, times: int,
                     sensitivity: float, use_cot: bool) -> int:
    if context:
        prompts = generate_prompts('prompts/evaluate_qa_with_context' + ('_cot' if use_cot else '') + '.txt', question, answer, context)
    else:
        prompts = generate_prompts('prompts/evaluate_qa_without_context' + ('_cot' if use_cot else '') + '.txt', question, answer)
    print(f"\n\n---------------------\n\nPrompts: {prompts}\n")
    responses = []
    for _ in range(times):
        delete_history()
        response = ''
        for prompt in prompts:
            response = prompt_model(model, prompt)
            print(response)
        responses.append(1. if response.startswith('Yes') else 0. if response.startswith('No') else 0.5)
    evaluation = sum(responses) / len(responses)
    return 1 if evaluation >= sensitivity else 0


def evaluate_qa_data(model: str, qa_data: list[dict], times: int, sensitivity: float, use_cot: bool) -> list[dict]:
    evaluated_qa_data = []
    for qa_pair in track(qa_data):
        qa_copy = qa_pair.copy()
        qa_copy['hallucination_pred'] = evaluate_qa_pair(model,
                                                         qa_copy.get('question'),
                                                         qa_copy.get('answer'),
                                                         qa_copy.get('context'),
                                                         times, sensitivity, use_cot)
        evaluated_qa_data.append(qa_copy)
    return evaluated_qa_data


def main() -> None:
    available_models = get_available_models()
    parser = argparse.ArgumentParser(description='Hallucination Detection')
    parser.add_argument('--data', default=None,
                        help='JSON file containing list of (context), question, answer')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of rows to be processed (for testing)', )
    parser.add_argument('--model', default='llama3',
                        help=f"Ollama model used for detecting hallucination. Options: {', '.join(available_models)}")
    parser.add_argument('--times', type=int, default=2, help='Number of times to evaluate each answer')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Average evaluation threshold to classify as hallucination')
    parser.add_argument('--cot', default='y',
                        help='Choose whether to use chain-of-thought when prompting (y/n)')
    parser.add_argument('--outfile', default=None,
                        help='Output JSON file')
    args = parser.parse_args()
    if not args.data:
        raise ValueError('Please add the input --data argument')
    if args.model not in available_models:
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    outfile = args.outfile if args.outfile else f'_{args.model}.'.join(args.data.rsplit('.', 1))
    use_cot = (args.cot[0] == 'y')

    with open(args.data, 'r') as f:
        qa_data = json.load(f)
    with open(outfile, 'w') as f:
        json.dump(evaluate_qa_data(args.model, qa_data[:args.limit], args.times, args.sensitivity, use_cot), f)


if __name__ == '__main__':
    main()
