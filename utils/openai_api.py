import json
import httpx
import os

from utils.function_definitions_llm import function_definitions_objects_llm

# OpenAI API settings
openai_api_chat = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
openai_api_key = os.getenv("AIPROXY_TOKEN")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}


def extract_parameters(prompt: str, function_definitions_llm):
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                openai_api_chat,
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an intelligent assistant that extracts structured parameters from user queries."},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": function_definitions_llm.get("name", "default_function_name"),
                            **function_definitions_llm
                        }
                    }],
                    "tool_choice": "auto"
                },
            )
        response.raise_for_status()
        response_data = response.json()
        print("🧾 Extracted raw argument string:", response_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        if "choices" in response_data and "tool_calls" in response_data["choices"][0]["message"]:
            extracted_data = response_data["choices"][0]["message"]["tool_calls"][0]["function"]
            parsed = json.loads(extracted_data.get("arguments", "{}"))

            # Special handling if only 'text' is present
            if list(parsed.keys()) == ["text"]:
                print("✅ Parsed parameters (text value):", parsed["text"])
                return parsed["text"]

            print("✅ Parsed parameters:", parsed)
            return parsed

        else:
            print("No parameters detected")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None




# Example usage
queries = [
    "Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24f1002771@ds.study.iitm.ac.in",
    # "Run npx -y prettier@3.4.2 README.md | sha256sum.",
    # "Type this formula in Google Sheets: =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 15, 12), 1, 10))",
]

function_defs = [
    "make_http_requests_with_uv",
    # "run_command_with_npx",
    # "use_google_sheets",
]

# for i in range(len(queries)):
# result = extract_parameters(queries[i], function_definitions_objects_llm[function_defs[i]])
# print(function_definitions_objects_llm[function_defs[i]])
# print(result)