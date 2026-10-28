"""Run a deterministic end-to-end demo, or explicitly enable the OpenAI adapter."""

import argparse
import json
import os

import httpx
from fastapi.testclient import TestClient

from onboarding.agent import Agent, Context, OpenAIModel
from onboarding.api import create_app


class ScriptedModel:
    """Offline fixture, not an LLM or evidence of model decision quality."""
    def __init__(self):
        self.step = 0

    def respond(self, messages):
        self.step += 1
        if self.step == 1:
            return [{"type": "function_call", "call_id": "demo-check", "name": "check_availability",
                     "arguments": '{"service":"onboarding"}'}]
        if self.step == 2:
            return [{"type": "function_call", "call_id": "demo-book", "name": "book_appointment",
                     "arguments": '{"slot_id":"slot-1"}'}]
        result = json.loads(messages[-1]["output"])
        text = "Synthetic appointment booked." if result.get("status") == "booked" else "Booking requires confirmation."
        return [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call OpenAI; requires API key and model")
    parser.add_argument("--message", default="Check availability and book slot-1 if approved.")
    parser.add_argument("--consent", action="store_true")
    parser.add_argument("--confirm-slot", default=None)
    args = parser.parse_args()
    context = Context("demo-customer", consent=args.consent, confirmed_slot=args.confirm_slot)
    if args.live:
        key = os.environ["APPOINTMENT_API_KEY"]
        model = OpenAIModel(os.environ["OPENAI_MODEL"])
        with httpx.Client(base_url="http://127.0.0.1:8000", headers={"X-API-Key": key}, timeout=10,
                          follow_redirects=False) as api:
            result = Agent(model, api).run(args.message, context)
    else:
        with TestClient(create_app("offline-demo-key"), headers={"X-API-Key": "offline-demo-key"}) as api:
            result = Agent(ScriptedModel(), api).run(args.message, context)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
