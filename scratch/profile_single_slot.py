import sys
sys.path.insert(0, ".")
import time
import json
from generation.ollama_client import OllamaClient
from generation.generator import generate_question, generate_answer
from ingestion.library import load_library

files = [
    ".cache/uploads/6a1831f8f825_Docker_in_Practice.pdf",
    ".cache/uploads/022a324a7b57_Kubernetes_in_Action.pdf",
    ".cache/uploads/2baa1d6c9eb4_Learning_DevOps.pdf",
]
lib = load_library(files)
chunks = lib["chunks"]
client = OllamaClient()

calls = []
orig_generate = client.generate

def tracked_generate(prompt, *args, **kwargs):
    t0 = time.perf_counter()
    resp = orig_generate(prompt, *args, **kwargs)
    dur = time.perf_counter() - t0
    calls.append({
        "dur": dur,
        "prompt_words": len(prompt.split()),
        "resp_words": len(resp.split()),
        "prompt": prompt[:120]
    })
    return resp

client.generate = tracked_generate

chunk = chunks[100]
ch_name = chunk.get("chapter", "unknown")
print(f"Testing Question Generation on chunk {ch_name}...")
t0 = time.perf_counter()
q_res = generate_question(chunk, "L2", client, marks=5, knowledge_dimension="Conceptual", force_regenerate=True)
t_q = time.perf_counter() - t0
print(f"Question generated in {t_q:.2f}s: valid={q_res.get('valid')}, reason={q_res.get('reason')}")
print(f"Question text: {q_res.get('question')}")

t0 = time.perf_counter()
a_res = generate_answer(q_res["question"], chunk, 5, client, force_regenerate=True)
t_a = time.perf_counter() - t0
print(f"Answer generated in {t_a:.2f}s: valid={a_res.get('valid')}, reason={a_res.get('reason')}")
print(f"Total LLM Calls made: {len(calls)}")
for i, c in enumerate(calls, 1):
    print(f"  Call {i}: {c['dur']:.2f}s, prompt {c['prompt_words']} words, resp {c['resp_words']} words")
