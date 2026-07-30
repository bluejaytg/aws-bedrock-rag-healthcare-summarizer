from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from guardrails.phi_sanitizer import PHISanitizer
from ingestion.fhir_parser import FHIRParser
from ingestion.chunking_strategy import ClinicalChunker
from pipeline.bedrock_embeddings import BedrockEmbeddings
from pipeline.vector_store import InMemoryVectorStore
from pipeline.rag_orchestrator import BedrockRAGOrchestrator

app = FastAPI(title="AWS Bedrock Healthcare RAG API")

# Initialize Pipeline Components
chunker = ClinicalChunker()
embedder = BedrockEmbeddings()
vector_db = InMemoryVectorStore()
orchestrator = BedrockRAGOrchestrator()

class IngestRequest(BaseModel):
    raw_clinical_payload: dict

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
async def ingest_clinical_data(request: IngestRequest):
    try:
        # 1. Parse FHIR / Raw Payload
        raw_text = FHIRParser.extract_clinical_text(request.raw_clinical_payload)
        
        # 2. Sanitize PHI
        clean_text = PHISanitizer.sanitize(raw_text)
        
        # 3. Chunk
        chunks = chunker.chunk_text(clean_text)
        
        # 4. Embed & Store
        vectors = [embedder.generate_embedding(chunk) for chunk in chunks]
        vector_db.add_vectors(vectors, chunks)
        
        return {"status": "success", "chunks_stored": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_patient_record(request: QueryRequest):
    try:
        # 1. Embed query
        query_vec = embedder.generate_embedding(request.query)
        
        # 2. Retrieve context
        context_chunks = vector_db.similarity_search(query_vec, top_k=3)
        
        if not context_chunks:
            return {"summary": "No relevant clinical records found.", "source_chunks_used": 0}

        # 3. Orchestrate RAG summary
        result = orchestrator.generate_summary(context_chunks, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))