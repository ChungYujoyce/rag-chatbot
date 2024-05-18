import requests
import json

a = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

request = {
    "model": "llama2-7b-chat",
    "prompt": [a],
    "max_tokens": 128,
    "temperature": 0.0,
    "top_p": 0.95,
    "stop": []
}

# Sending the PUT request
outputs = requests.post(
    url="http://172.18.0.2:5000/v1/completions",
    data=json.dumps(request),
    headers={"Content-Type": "application/json"},
).json()
print(outputs)