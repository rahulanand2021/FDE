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

class ClaimDecision(BaseModel):
    """A claims handler's decision on one claim."""

    rationale: str = Field(description="2-3 sentences to explain the decision along with the relevant section of the policy")
    decision: Literal["approve", "decline", "refer"]
    payout: float = Field(description="The amount to pay in US dollars, or 0 for a decline or a referral")


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

    fig = px.histogram(data, x="claim_type", color="decision", title="Claims by type")
    fig.show()
    
    fig = px.histogram(
        data,
        x="difficulty",
        color="decision",
        category_orders={"difficulty": ["routine", "hard", "manual"]},
        title="Claims by difficulty"
    )
    fig.show()

def loadPolicies(claims, labels):
    policies = {policy["policy_id"]: policy for policy in load("policies.json")}
    policy_wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")

    print(f"{len(claims)} claims, {len(labels)} labels, {len(policies)} policies")
    print(f"The policy wording is {len(policy_wording.split()):,} words")
    return policies

def make_user_prompt(claim, policies):
    """The customer's policy record and the claim, as JSON."""
    policy = policies[claim["policy_id"]]

    return f"""# Policy record

        {json.dumps(policy, indent=2)}

        # Claim

        {json.dumps(claim, indent=2)}
        """

def make_messages(claim, policies):
    """Build the messages for one claim: the task and the wording, then the policy record and the claim."""


    policy_wording = (DATA / "policy_wording.md").read_text(encoding="utf-8")

    INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
    Decide the claim you are given, using the policy wording below and the customer's policy record.

    Choose one decision:
    - approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
    - decline: the wording does not cover the claim.
    - refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.
    """
    SYSTEM_PROMPT = f"""{INSTRUCTIONS}
    {policy_wording}
    """


    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(claim, policies)},
    ]

def is_right(answer, label):
    """True when the decision matches the handler's and the payout is within a dollar."""
    return answer["decision"] == label["decision"] and abs(answer["payout"] - label["payout"]) <= 1

## Call the LLM
def decide(claim, policies, model, effort):
    """Ask one model to decide one claim."""
    openrouter = openRouterObject()
    start = time.time()
    response = openrouter.chat.completions.parse(
        model=model,
        messages=make_messages(claim, policies),
        response_format=ClaimDecision,
        reasoning_effort=effort,
        extra_body={"provider": {"require_parameters": True}},
    )
    answer = response.choices[0].message.parsed
    return {
        "model": model,
        "effort": effort,
        "technique": "LLM",
        "claim_id": claim["claim_id"],
        "decision": answer.decision,
        "payout": answer.payout,
        "rationale": answer.rationale,
        "cost": response.usage.cost,
        "seconds": round(time.time() - start, 1),
    }

def run(model, policies, effort, claims, labels):
    """Run the model on the claims and compare against the labels."""
    results = []
    for claim in claims:
        answer = decide(claim, policies, model, effort)
        label = labels[claim["claim_id"]]
        results.append({**answer, "difficulty": label["difficulty"], "right": is_right(answer, label)})
    right = sum(result["right"] for result in results)
    cost = sum(result["cost"] for result in results)
    seconds = sum(result["seconds"] for result in results)
    print(f"{model} at {effort}: {right} of {len(results)} right for ${cost:.4f} in {seconds:.0f} seconds")
    return results

def show_misses(results, labels):
    """Show each claim in a set of results that the model got wrong, with its rationale and the handler's reason."""
    for result in results:
        if not result["right"]:
            label = labels[result["claim_id"]]
            print((f"""**{result['claim_id']}, {label['difficulty']}**: {result['decision']} ${result['payout']:,.2f}, where the handler said {label['decision']} ${label['payout']:,.2f}

        - Model: {result['rationale']}
        - Handler: {label['reason']}
        - Tests: {label['tests']}
        """))
            
if __name__ == "__main__":
    loadAPIKeys()
    # testOpenRouter()
    claims, labels = peekSampleData()
    policies = loadPolicies(claims, labels)
    # pol1 = make_user_prompt(claims[0],policies)
    # print(pol1)
    # message = make_messages(claims[0], policies)
    # print(message)
    # showClaimInPandasDataFrame(claims, labels)

    # first = decide(claims[0], policies, "openai/gpt-6-luna", "low")
    # print(first)

    # luna_low  = run("openai/gpt-6-luna", policies, "low", claims, labels)
    # luna_medium  = run("openai/gpt-6-luna", policies, "medium", claims, labels)
    # scored = pd.DataFrame(luna_low + luna_medium)
    # print(scored.pivot_table(index=["model", "effort"], columns="difficulty", values="right", aggfunc="sum", sort=False))
    # show_misses(luna_low, labels)
    # sol = run("openai/gpt-6-sol", policies, "high", claims, labels)