import json
import boto3
from typing import List

class BedrockEmbeddings:
    """Generates vector embeddings using Amazon Titan Embeddings v2."""

    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = "amazon.titan-embed-text-v2:0"

    def generate_embedding(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")