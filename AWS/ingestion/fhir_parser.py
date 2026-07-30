import json
from typing import Dict, Any, List

class FHIRParser:
    """Extracts unstructured clinical text and key observations from FHIR JSON payloads."""

    @staticmethod
    def extract_clinical_text(fhir_payload: Dict[str, Any]) -> str:
        clinical_notes = []
        
        # Parse DocumentReference or Condition resources
        resource_type = fhir_payload.get("resourceType", "")
        
        if resource_type == "Bundle":
            for entry in fhir_payload.get("entry", []):
                res = entry.get("resource", {})
                if "text" in res and "div" in res["text"]:
                    clinical_notes.append(res["text"]["div"])
                elif "note" in res:
                    for note in res["note"]:
                        clinical_notes.append(note.get("text", ""))
        elif "text" in fhir_payload:
            clinical_notes.append(fhir_payload["text"].get("div", ""))
            
        extracted_text = " ".join(clinical_notes) if clinical_notes else json.dumps(fhir_payload)
        return extracted_text