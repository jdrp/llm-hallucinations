import ollama
import concurrent.futures


messages = []


def get_available_models(filename: str = 'available_models.txt') -> list[str]:
    """Returns list of available (downloaded) models"""
    with open(filename, 'r') as f:
        return f.read().splitlines()


def delete_history() -> None:
    """Starts a new chat"""
    global messages
    messages = []


def prompt_model(model: str, prompt: str, timeout: int = 900) -> str:
    """Prompts the model, continuing the previous conversation. Requires running 'ollama serve' first"""
    messages.append(
        {
            'role': 'user',
            'content': prompt,
        },
    )

    def chat_with_model():
        return ollama.chat(model=model, messages=messages)['message']['content']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(chat_with_model)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"The chat model took longer than {timeout} seconds to respond")

    messages.append(
        {
            'role': 'assistant',
            'content': response,
        }
    )

    return response


logfile: str | None = None


def set_logfile(new_logfile: str):
    global logfile
    logfile = new_logfile


def log(text: str, end: str = '\n'):
    if logfile:
        with open(logfile, 'a+', encoding='utf-8') as f:
            f.write(text)
            f.write(end)
    else:
        try:
            print(text)
        except Exception:
            pass
