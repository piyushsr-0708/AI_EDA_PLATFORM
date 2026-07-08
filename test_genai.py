import sys
from google import genai
from google.genai import types

try:
    print("Testing google.genai function response...")
    part = types.Part.from_function_response(name="my_func", response={"result": 42})
    print("Part created successfully!")
    print(part)
except Exception as e:
    print("Error:", e)
