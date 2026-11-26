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

recorded = {}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DATA = Path("data")


def loadAPIKeys():
    global openrouter_api_key
    load_dotenv(override=True)
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key:
        print(f"Open Router API Key exists and begins {openrouter_api_key[:8]}")
    else:
        print("Open Router API Key not set")

def test_create_agent_from_langchain():
    llm = ChatOpenAI(model="openrouter/free", base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)

    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers in the style of a pirate."},
        {"role": "user", "content": "What is the meaning of the universe?"},
        ]

    response = llm.invoke(messages)
    print((response.content))

def load(nameOfFileAsString: str):
    """Read one of the JSON seed files."""
    return json.loads((DATA / nameOfFileAsString).read_text(encoding="utf-8"))

@tool
def get_policy(policy_id: str) -> str:
    """Look up a customer's policy record: the plan, the add-ons, the purchase date, the trip and the travelers."""

    policy = policies.get(policy_id)
    return json.dumps(policy, indent=2) if policy else f"There is no policy {policy_id}"

@tool
def record_decision(claim_id: str, decision: Literal["approve", "decline", "refer"], payout: float, rationale: str) -> str:
    """Record the decision on a claim in the claims system.

    payout is the amount to pay in US dollars, or 0 for a decline or a referral.
    rationale is 2-3 sentences to explain the decision along with the relevant section of the policy.
    """
    recorded[claim_id] = {"decision": decision, "payout": payout, "rationale": rationale}
    return f"Recorded {decision} on {claim_id}"

def call_agent_once():

    INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
    Decide the claim you are given, using the policy wording below and the customer's policy record.

    Choose one decision:
    - approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
    - decline: the wording does not cover the claim.
    - refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.
    """

    policy_wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")
    luna = ChatOpenAI(model="openai/gpt-6-luna", base_url=OPENROUTER_BASE_URL, 
                      api_key=openrouter_api_key, reasoning_effort="low", timeout=120, max_retries=5)

    AGENT_SYSTEM_PROMPT = f"""{INSTRUCTIONS}
    Use your tools to look up the customer's policy record and to record your decision.

    {policy_wording}
    """
    agent = create_agent(model=luna, tools=[get_policy, record_decision], 
                         system_prompt=AGENT_SYSTEM_PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": make_claim_prompt(claims[0])}]})

    for message in result["messages"][1:]:
        message.pretty_print()

def make_claim_prompt(claim):
    """The claim alone, as JSON. The agent looks up the policy record for itself."""
    return f"""# Claim

        {json.dumps(claim, indent=2)}
        """

def is_right(answer, label):
    """True when the decision matches the handler's and the payout is within a dollar."""
    
    return answer["decision"] == label["decision"] and abs(answer["payout"] - label["payout"]) <= 1

def decide_with_agent(claim):
    """Run the agent on one claim and check what it recorded against the handler's decision."""
    
    INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
    Decide the claim you are given, using the policy wording below and the customer's policy record.

    Choose one decision:
    - approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
    - decline: the wording does not cover the claim.
    - refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.
    """

    policy_wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")
    luna = ChatOpenAI(model="openai/gpt-6-luna", base_url=OPENROUTER_BASE_URL, 
                      api_key=openrouter_api_key, reasoning_effort="low", timeout=120, max_retries=5)

    AGENT_SYSTEM_PROMPT = f"""{INSTRUCTIONS}
        Use your tools to look up the customer's policy record and to record your decision.

        {policy_wording}
        """
    agent = create_agent(model=luna, tools=[get_policy, record_decision], 
                         system_prompt=AGENT_SYSTEM_PROMPT)
    start = time.time()
    result = agent.invoke({"messages": [{"role": "user", "content": make_claim_prompt(claim)}]})
    answer = recorded.get(claim["claim_id"], {"decision": "nothing recorded", "payout": 0.0, "rationale": ""})
    label = labels[claim["claim_id"]]
    cost = sum(message.response_metadata["token_usage"]["cost"] for message in result["messages"] if message.type == "ai")
    return {
        "model": "openai/gpt-6-luna",
        "effort": "low",
        "technique": "Agent",
        "claim_id": claim["claim_id"],
        **answer,
        "difficulty": label["difficulty"],
        "right": is_right(answer, label),
        "cost": cost,
        "seconds": round(time.time() - start, 1),
    }

def call_agent():
    recorded.clear()
    agent_results = []
    for claim in claims:
        agent_results.append(decide_with_agent(claim))
    print(f"{len(recorded)} of {len(claims)} claims have a recorded decision")
    pd.DataFrame(agent_results).groupby(["model", "effort", "technique"], sort=False)[["right", "cost", "seconds"]].sum()

    
if __name__ == "__main__":
    loadAPIKeys()
    # test_create_agent_from_langchain()
    policies = {policy["policy_id"]: policy for policy in load("policies.json")}
    claims = load("claims.json")
    decisions = load("evals/decisions.json")
    labels = {decision["claim_id"]: decision for decision in decisions}
    # call_agent_once()
    call_agent()
