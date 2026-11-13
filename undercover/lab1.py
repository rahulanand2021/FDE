import json
import os
import time
from pathlib import Path
from typing import Literal
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from openrouter import OpenRouter
from pydantic import BaseModel, Field
import time


DATA = Path("data")

def loadAPIKeys():
    global openrouter_api_key
    load_dotenv(override=True)
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key:
        print(f"Open Router API Key exists and begins {openrouter_api_key[:8]}")
    else:
        print("Open Router API Key not set")

def openRouterObject():
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    openrouter = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL, 
                            timeout=12, max_retries=2)
    return openrouter

def testOpenRouter():
    openrouter = openRouterObject()

    messages = [
            {"role": "system", "content": "You are a helpful assistant who answers in the style of a cowboy."},
            {"role": "user", "content": "What is the meaning of the universe?"}
        ]
    response = openrouter.chat.completions.create(model="openai/gpt-6-luna", messages=messages)
    # print(f"Answered by {response.model}")
    print((response.choices[0].message.content))

def load(nameOfFileAsString: str):
    """Read one of the JSON seed files."""
    return json.loads((DATA / nameOfFileAsString).read_text(encoding="utf-8"))


def get_sample_data(tiny=False, held_out=False):
    """Return a list of claims, and the handlers' decisions on them keyed by claim id.

    held_out=True gives the twenty-five held-out claims in place of the queue.
    tiny=True keeps the five claims marked for the tiny version of either set.
    """
    claims = load("evals/held_out_claims.json" if held_out else "claims.json")
    decisions = load("evals/held_out_decisions.json" if held_out else "evals/decisions.json")
    if tiny:
        chosen = {decision["claim_id"] for decision in decisions if decision["tiny"]}
        claims = [claim for claim in claims if claim["claim_id"] in chosen]
        decisions = [decision for decision in decisions if decision["tiny"]]
    labels = {decision["claim_id"]: decision for decision in decisions}
    return claims, labels

def peekSampleData():
    claims, labels = get_sample_data()
    print(f"{len(claims)} claims, {len(labels)} labels")
    # print("Sample claim:", claims[0])
    # print("Sample label:", labels[claims[0]["claim_id"]])
    return claims, labels

def showClaimInPandasDataFrame(claims, labels):
    rows = []
    for claim in claims:
        label = labels[claim["claim_id"]]
        rows.append({
            "claim_id": claim["claim_id"],
            "summary": claim["summary"],
            "claim_type": claim["claim_type"],
            "amount_claimed": claim["amount_claimed"],
            "difficulty": label["difficulty"],
            "decision": label["decision"],
            "payout": label["payout"],
        })

    data = pd.DataFrame(rows)
    print(data)

    home_alone = [claim for claim in claims if claim["claim_id"] == "CLM-1015"][0]
    decision = labels["CLM-1015"]


    print(home_alone)
    # print((f"""**{home_alone['summary']}**

    # {home_alone['story']}

    # **Handler:** {decision['decision']} ${decision['payout']:,.2f}. {decision['reason']}
    # """))

if __name__ == "__main__":
    loadAPIKeys()
    # testOpenRouter()
    claims, labels = peekSampleData()
    showClaimInPandasDataFrame(claims, labels)
