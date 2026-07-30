import json
import boto3
from typing import Dict, Any, List

class BedrockClinicalRAG:
    """
    Orchestrates RAG generation via Amazon Bedrock for clinical summarization,
    enforcing strict context grounding to prevent hallucinations in medical workflows.
    """
    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)
        self.model_id = model_id

    def build_grounded_prompt(self, clinical_context: str, query: str) -> str:
        return f"""
        Human: You are a secure healthcare AI assistant supporting clinical review. 
        Synthesize the following clinical patient records to answer the query.
        
        CRITICAL RULES:
        1. Base your summary ONLY on the provided Context below.
        2. If the answer cannot be directly derived from the context, respond with "Insufficient clinical data provided."
        3. Do NOT assume or extrapolate medical conditions not explicitly stated.

        <context>
        {clinical_context}
        </context>

        Query: {query}

        Assistant:
        """

    def generate_summary(self, context_chunks: List[str], query: str) -> Dict[str, Any]:
        combined_context = "\n---\n".join(context_chunks)
        prompt = self.build_grounded_prompt(combined_context, query)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.1,  # Low temp for clinical precision
            "messages": [{"role": "user", "content": prompt}]
        })

        response = self.bedrock_runtime.invoke_model(
            body=body,
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        return {
            "summary": response_body["content"][0]["text"],
            "model_used": self.model_id,
            "source_chunks_count": len(context_chunks)
        }