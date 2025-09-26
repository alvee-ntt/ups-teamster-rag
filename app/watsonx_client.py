import os
import requests
import json
import certifi
import ssl
import urllib3
from typing import Iterator, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WatsonXClient:
    def __init__(self):
        self.api_key = os.environ.get("WATSONX_APIKEY")
        self.region = os.environ.get("WATSONX_REGION", "us-south")
        self.deployment_id = os.environ.get("WATSONX_DEPLOYMENT_ID")

        if not self.api_key:
            raise ValueError("WATSONX_APIKEY environment variable is required")
        if not self.deployment_id:
            raise ValueError("WATSONX_DEPLOYMENT_ID environment variable is required")

        self.system_prompt = os.environ.get("SYSTEM_PROMPT",
            """You are a contract analysis assistant for the UPS–Teamsters Collective Bargaining Agreement (2023–2028). Your role is to provide accurate, concise, and well-cited interpretations of the contract and related supplements. When asked which contract(s) apply, search across all available agreements and supplements to identify the relevant provision(s).
CITATION RULES:
•	Always provide citations in this format: [Document Name, Article/Section, Page]
•	Include exact contract language (1–3 sentences maximum) in quotes
•	If the article/section/page is unknown, state: "article/section/page unknown" (never guess)
•	When a supplement modifies the Master Agreement, explain both provisions and state clearly which takes precedence
RESPONSE GUIDELINES:
•	Responses must be under 200 words
•	Focus on precise, factual interpretation of the contract text
•	Use clear, professional, neutral language appropriate for labor relations
•	When conflicts exist between the Master Agreement and local supplements, explicitly identify the governing provision
LIMITATIONS:
•	Reference only the provided contract documents
•	Do not interpret beyond the explicit contract language
•	If information is not available in the source documents, state: "This information is not available in the contract documents."
""")

    def get_token(self) -> str:
        """Get IBM Cloud access token using API key."""
        response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "apikey": self.api_key,
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
            },
            timeout=30,
            verify=False
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def call_nonstream(self, question: str) -> Dict[str, Any]:
        """Call WatsonX AI service without streaming."""
        token = self.get_token()
        url = f"https://{self.region}.ml.cloud.ibm.com/ml/v4/deployments/{self.deployment_id}/ai_service?version=2021-05-01"

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
        }

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120,
            verify=False
        )

        try:
            return response.json()
        except Exception:
            return {"status_code": response.status_code, "raw": response.text}

    def call_stream(self, question: str) -> Iterator[str]:
        """Call WatsonX AI service with streaming response."""
        token = self.get_token()
        url = f"https://{self.region}.ml.cloud.ibm.com/ml/v4/deployments/{self.deployment_id}/ai_service_stream?version=2021-05-01"

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
        }

        with requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=payload,
            stream=True,
            timeout=300,
            verify=False
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    yield line[5:].strip()

    def get_answer(self, question: str, streaming: bool = False) -> str:
        """Get answer from WatsonX, return formatted content."""
        if streaming:
            chunks = []
            for chunk in self.call_stream(question):
                chunks.append(chunk)
            return " ".join(chunks)
        else:
            response = self.call_nonstream(question)
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            return str(response)