import json
import argparse
from rich.progress import track
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


def generate_prompt(report: dict) -> str:
    with open('prompts/medical_right_and_hallucinated.txt', 'r') as f:
        return f.read().replace('[R]', json.dumps(report))


def generate_qa(model, report_file: str) -> list[dict]:
    with open(report_file, 'r') as f:
        report = json.load(f)
    prompt = generate_prompt(report)
    response = prompt_model(model, prompt)
    response_clean = response[response.find('['):response.rfind(']') + 1]
    # parse response
    try:
        qa_list = json.loads(response_clean)
        print(qa_list)
        if any(not set(qa.keys()) == {'question', 'right_answer', 'hallucinated_answer'} for qa in qa_list):
            raise ValueError
        return qa_list
    except ValueError:
        print('Error decoding reponse:')
        print(response)


def main() -> None:
    available_models = get_available_models()
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', default=None, help='Directory containing JSON medical reports')
    parser.add_argument('--model', default='llama3',
                        help=f"Ollama model used for detecting hallucination. Options: {', '.join(available_models)}")
    parser.add_argument('--output', default=None, help='Directory to save output QA files')
    args = parser.parse_args()

    if not args.reports or not os.path.exists(args.reports):
        raise ValueError('Please input a valid --reports folder')
    if args.model not in available_models:
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    output_folder = args.output if args.output else 'data/question_correct_hallucinated'

    os.makedirs(output_folder, exist_ok=True)
    for report_file in track(os.listdir(args.reports)):
        if report_file.endswith('.json'):
            report_name = report_file[:-5]
            output_file = report_name + '_qch.json'
            report_path = os.path.join(args.reports, report_file)
            output_path = os.path.join(output_folder, output_file)
            with open(output_path, 'w') as f:
                json.dump(generate_qa(args.model, report_path), f)


if __name__ == '__main__':
    main()
