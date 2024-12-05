import requests
import json

def verify_ollama_llm(prompt: str, model: str, url: str = "http://127.0.0.1:11434/api/generate"):
    """
    Verify if the Ollama-hosted local LLM is running and responding to streamed output.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The name of the model to use.
        url (str): The URL of the Ollama LLM server.

    Returns:
        str: The aggregated response from the LLM.
    """
    try:
        print(f"Sending prompt to Ollama LLM at {url}...")
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt},
            stream=True,  # Enable streaming
            timeout=10  # Timeout for the request
        )

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return None

        # Process the streaming response
        result = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():  # Ignore empty lines
                try:
                    json_data = json.loads(line)
                    result += json_data.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}, Error: {e}")
        
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama LLM: {e}")
        return None


if __name__ == "__main__":
    # Test prompt
    test_prompt = "Hello, how are you?"
    model_name = "qwen2.5-coder:latest"  # Replace with your model name
    
    # Verify the LLM
    response = verify_ollama_llm(test_prompt, model_name)
    
    if response:
        print("\nLLM Response:")
        print(response)
    else:
        print("Failed to verify the Ollama LLM.")
