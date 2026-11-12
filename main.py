import json
import requests
import os
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

read_web_page_json = {
    "type": "function",
    "function": {
        "name": "read_web_page",
        "description": "Read a web page and return the text on the page",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL of the web page to request"},
            },
            "required": ["url"],
        },
    },
}

tools = [read_web_page_json]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openrouter/free"


def loadAPIKeys():
    global openrouter_api_key
    load_dotenv(override=True)
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key:
        print(f"Open Router API Key exists and begins {openrouter_api_key[:8]}")
    else:
        print("Open Router API Key not set")

def isOpenRouterKeyAccepted():
    response = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {openrouter_api_key}"})
    result = response.json()

    if response.ok:
        key_info = result["data"]
        print(f"Success! OpenRouter accepted your key {key_info['label']}")
    else:
        print(f"OpenRouter did not accept this key: {result['error']['message']} (error {response.status_code})")
        print("Please repeat Setup parts 3 and 4.")

def connectToOpenRouter():

    # Retry up to 5 times when a free model is busy
    openrouter = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL, 
                        timeout=12, max_retries=2)

    messages = [{"role": "user", "content": "Hi there!"}]

    response = openrouter.chat.completions.create(model=MODEL, messages=messages)
    print(f"Answered by {response.model}")
    print((response.choices[0].message.content))

def read_web_page(url):
    MAZE_URL = "https://maze.edwarddonner.com"
    """Read a web page in the maze and return the text on the page, without the HTML"""
    print(f"Reading web page: {url}")
    if not url.startswith(MAZE_URL):
        return f"Sorry, this tool can only read pages in the maze. Please ask for a URL that starts with {MAZE_URL}"
    response = requests.get(url, timeout=20)
    page = BeautifulSoup(response.text, "html.parser")
    return page.body.get_text(separator="\n", strip=True)

def handle_tool_call(tool_call):
    """Run the tool that the model asked for and package the result as a message"""
    try:
        arguments = json.loads(tool_call.function.arguments)
        result = read_web_page(**arguments)
    except Exception as error:
        result = f"The tool call failed: {error}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": result}

def runToolOnceForTest():

    # Retry up to 5 times when a free model is busy
    openrouter = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL, 
                        timeout=12, max_retries=2)
    
    messages = [{"role": "user", "content": "Please read https://maze.edwarddonner.com and tell me what it says. Just describe the page for now, without going into the maze."}]

    response = openrouter.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    message = response.choices[0].message

    print(f"Answered by {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(message.tool_calls)

    messages.append(message)

    for tool_call in message.tool_calls:
        messages.append(handle_tool_call(tool_call))

    response = openrouter.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    print(f"Answered by {response.model}")
    print((response.choices[0].message.content))

def run_agent(task, max_steps=25):

    SYSTEM_PROMPT = """
        You are an agent that completes tasks by using your tools, one step at a time.
        Your replies are displayed as Markdown, so format them in Markdown with headings, bold text and lists.
        Put any map or diagram in a code block so that it lines up. Do not use emojis.
    """

    """Call the model in a loop, running the tool whenever it asks, until it gives a final answer"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    openrouter = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL, 
                            timeout=12, max_retries=2)
    
    for step in range(1, max_steps + 1):
        response = openrouter.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        message = response.choices[0].message
        print(f"Step {step} answered by {response.model}")

        text = (message.content or "").strip()
        
        if text:
            print((text))

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                messages.append(handle_tool_call(tool_call))
        elif text:
            return
        else:
            print("The reply was empty, so we ask again")

    print("The agent ran out of steps")

if __name__ == "__main__":
    loadAPIKeys()
    # isOpenRouterKeyAccepted()
    # connectToOpenRouter()
    # print(read_web_page("https://maze.edwarddonner.com"))
    # runToolOnceForTest()
    run_agent("Have a look at https://maze.edwarddonner.com and do what it says.")
