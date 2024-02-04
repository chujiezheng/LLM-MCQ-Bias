from openai import OpenAI

class IChatAPI:
    def __init__(self):
        self.client = OpenAI()

    def __call__(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)
