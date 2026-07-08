from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, api_key: str, tools: list, system_instruction: str):
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        self.chat = self.client.chats.create(model="gemini-2.5-flash", config=self.config)

    def send_message(self, message):
        return self.chat.send_message(message)
