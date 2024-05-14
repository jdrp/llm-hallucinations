import ollama


messages = []


def get_available_models(filename: str = 'available_models.txt') -> list[str]:
    """Returns list of available (downloaded) models"""
    with open(filename, 'r') as f:
        return f.read().splitlines()


def delete_history() -> None:
    """Starts a new chat"""
    global messages
    messages = []


def prompt_model(model: str, prompt: str) -> str:
    """Prompts the model, continuing the previous conversation"""
    messages.append(
        {
            'role': 'user',
            'content': prompt,
        },
    )
    response = ollama.chat(model=model, messages=messages)['message']['content']
    messages.append(
        {
            'role': 'assistant',
            'content': response,
        }
    )
    # rsp = response[-1]['message']['content']
    return response
