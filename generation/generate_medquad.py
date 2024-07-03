import argparse

import sys
import requests
from bs4 import BeautifulSoup, PageElement
import os
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import random
import json
from rich.progress import track
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


site_contents = {}


def clean_inline_elements(parent: PageElement) -> PageElement:
    inline_elements = ['a', 'abbr', 'acronym', 'button', 'br', 'big', 'bdo', 'b', 'cite', 'code', 'dfn', 'i', 'em',
                       'img', 'input', 'kbd', 'label', 'map', 'object', 'output', 'tt', 'time', 'samp', 'script',
                       'select', 'small', 'span', 'strong', 'sub', 'sup', 'textarea']
    for tag in inline_elements:
        for e in parent.select(tag):
            e.unwrap()
    return parent


def remove_from_soup(soup: BeautifulSoup, html_type: str, tags: dict, search_by_child: bool = False, remove_sibling: str | None = None) -> None:
    if 'text' in tags:
        remove_this = soup.find(html_type, string=tags['text'])
    else:
        remove_this = soup.find(html_type, tags)
    if remove_this:
        if search_by_child:
            remove_this = remove_this.parent
        if remove_sibling:
            if next_sibling := remove_this.find_next_sibling(remove_sibling):
                next_sibling.decompose()
        remove_this.decompose()


def get_site_content(url: str) -> str:
    content = None
    try:
        domain = urlparse(url).netloc
        path = urlparse(url).path
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        separator = '\n'

        # extract the main content
        if domain == 'www.cancer.gov':  # 1_CancerGov_QA
            remove_from_soup(soup, 'nav', {'class': 'on-this-page'})
            remove_from_soup(soup, 'div', {'class': 'pdq-hp-patient-toggle'})
            content = clean_inline_elements(soup.find('main', {'id': 'main-content'})).get_text(separator=separator, strip=True).rsplit('To Learn More About')[0]
        elif domain == 'ghr.nlm.nih.gov':  # 3_GHR_QA
            content = clean_inline_elements(soup.find('div', {'class': 'main'})).get_text(separator=separator, strip=True).rsplit('Additional Information & Resources')[0]
        elif domain == 'www.niddk.nih.gov':  # 5_NIDDK_QA
            remove_from_soup(soup, 'b', {'text': 'On this page:'}, search_by_child=True, remove_sibling='ul')
            remove_from_soup(soup, 'strong', {'text': 'On this page:'}, search_by_child=True, remove_sibling='ul')
            remove_from_soup(soup, 'strong', {'text': 'In this section:'}, search_by_child=True, remove_sibling='ul')
            content = clean_inline_elements(soup.find('article', {'class': 'dk-content'})).get_text(separator=separator, strip=True).rsplit('References')[0]
        elif domain == 'www.nhlbi.nih.gov':  # 8_NHLBI_QA_XML
            content = clean_inline_elements(soup.find('div', {'class': 'field__items'})).get_text(separator=separator, strip=True)
        elif domain == 'www.nlm.nih.gov':
            if path.startswith('/medlineplus/ency'):  # 10_MPlus_ADAM_QA
                content = clean_inline_elements(soup.find('div', {'class': 'main-single'}) or soup.find('div', {'class': 'main'})).get_text(separator=separator, strip=True)
            elif path.startswith('/medlineplus/druginfo/meds'):  # 11_MPlusDrugs_QA
                remove_from_soup(soup, 'div', {'class': 'page-info'}, remove_sibling='div')
                content = clean_inline_elements(soup.find('article')).get_text(separator=separator, strip=True).rsplit('Last Revised -')[0]
            elif path.startswith('/medlineplus/druginfo/natural'):  # 12_MPlusHerbsSupplements_QA
                remove_from_soup(soup, 'div', {'class': 'page-info'}, remove_sibling='div')
                content = clean_inline_elements(soup.find('article')).get_text(separator=separator, strip=True).rsplit('References')[0]
            else:  # 4_MPlus_Health_Topics_QA
                remove_from_soup(soup, 'section', {'id': 'toc-section'})
                remove_from_soup(soup, 'div', {'class': 'mp-refs'})
                remove_from_soup(soup, 'p', {'class': 'attribution'})
                content = clean_inline_elements(soup.find('div', {'class': 'main'})).get_text(separator=separator, strip=True).rsplit('Start Here')[0]

    except requests.exceptions.HTTPError or requests.exceptions.ReadTimeout:
        print(f'Cannot find {url}')
        pass
    except AttributeError as e:
        print(f'No main content in {url}: {e}')
        pass
    return content


def find_element(parent, *tag_names):
    for tag in tag_names:
        element = parent.find(tag)
        if element is not None:
            return element
    return None


def parse_xml(path: str, fetch_context: bool) -> list[dict]:
    qas_with_url = []
    root = ET.parse(path).getroot()
    context = None
    if fetch_context:
        url = root.attrib['url']
        if url in site_contents.keys():
            context = site_contents[url]
        else:
            context = get_site_content(url)
            site_contents[url] = context
        if not context:
            return qas_with_url
    qa_pairs = find_element(root, 'QAPairs', 'qaPairs')
    if qa_pairs:
        for qa_pair in qa_pairs.findall('QAPair') or qa_pairs.findall('pair'):
            question = find_element(qa_pair, 'Question', 'question')
            answer = find_element(qa_pair, 'Answer', 'answer')
            if question is None or answer is None or (answer.text == 'No information found.'):
                continue
            question = question.text.replace(' (are)', '')
            if question.endswith(' - resources ?') or question.startswith('Do you have information about '):
                continue
            qa_entry = {
                'question': question,
                'right_answer': answer.text
            }
            if fetch_context:
                qa_entry['context'] = context
            qas_with_url.append(qa_entry)
    return qas_with_url


def generate_prompt(template: str, qa: dict) -> str:
    with open(template, 'r') as f:
        return f.read().replace('[C]', qa['context']).replace('[Q]', qa['question']).replace('[A]', qa['right_answer'])


def remove_extra_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered_sentences = [sentence for sentence in sentences if 'hallucinat' not in sentence.lower()]
    filtered_sentences = [sentence for sentence in sentences if 'context' not in sentence.lower()]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def hallucinate_answer(model: str, qa: dict) -> str:
    delete_history()
    prompt = generate_prompt('prompts/generation/generate_medquad_hall_qa.prompt', qa)
    response = prompt_model(model, prompt)
    response = remove_extra_sentences(response)
    print('------------------')
    print(qa['question'])
    print('Hallucination')
    print(response)
    return response


def regenerate_answer(model: str, qa: dict) -> str:
    delete_history()
    prompt = generate_prompt('prompts/generation/generate_medquad_right_qa.prompt', qa)
    response = prompt_model(model, prompt)
    print('------------------')
    print(qa['question'])
    print('New answer')
    print(response)
    return response


def main() -> None:
    available_models = get_available_models()
    random.seed = 50
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/MedQuAD', help='Data directory')
    parser.add_argument('--from_json', default=None, help='JSON missing hallucinations')
    parser.add_argument('--questions', type=int, default=-1,
                        help='Number of questions to (randomly) pick from each source')
    parser.add_argument('--hallucination', default=False, action=argparse.BooleanOptionalAction,
                        help='Generate hallucinated answers')
    parser.add_argument('--model', default='llama3',
                        help=f"Ollama model used for generating hallucinations. Options: {', '.join(available_models)}")
    parser.add_argument('--context', default=True, action=argparse.BooleanOptionalAction,
                        help='Fetch website content')
    parser.add_argument('--regenerate', default=False, action=argparse.BooleanOptionalAction,
                        help='Regenerate correct answers')
    parser.add_argument('--outfile', default='data/medquad.json', help='Output JSON file')
    args = parser.parse_args()
    if args.model not in available_models:
        raise ValueError(f"Please select a --model from the following: {', '.join(available_models)}")
    missing_sources = [
        '2_GARD_QA', '6_NINDS_QA',  # 404 not found
        '7_SeniorHealth_QA',  # 11001 host not found
        '9_CDC_QA',  # website structure completely changed, each question from a different subsite
        '5_NIDDK_QA'  # mostly 404
        # '10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA'  # no answers --- scraped with https://github.com/glicerico/medquad-scraper/blob/main/src/scrape_Herbs.py
    ]
    final_qas: list[dict] = []
    if args.from_json:
        with open(args.from_json, 'r') as f:
            final_qas = json.load(f)
    else:
        sources = (entry for entry in os.listdir(args.data) if entry[0].isnumeric() and entry not in missing_sources)
        for source in sources:
            source_qas = []
            files = os.listdir(f'{args.data}/{source}')
            random.shuffle(files)
            for file in track(files, description=source):
                file_qas = parse_xml(f'{args.data}/{source}/{file}', args.context)
                if not file_qas:
                    continue
                source_qas.append(random.choice(file_qas))
                if len(source_qas) == args.questions:
                    break
            random_qas = random.sample(source_qas, args.questions if args.questions > 0 else len(source_qas))
            final_qas.extend(random_qas)
    if args.regenerate:
        for qa in track(final_qas, description='Regenerating answers'):
            regen_answer = regenerate_answer(args.model, qa)
            if 'Missing information' in regen_answer:
                del qa
            else:
                qa['right_answer'] = regen_answer
    if args.hallucination:
        for qa in track(final_qas, description='Generating hallucinations'):
            qa['hallucinated_answer'] = hallucinate_answer(args.model, qa)
    with open(args.outfile, 'w') as f:
        json.dump(final_qas, f, indent=4)


if __name__ == '__main__':
    main()
