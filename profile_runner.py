"""Profile script to measure baseline latency across all pipeline stages."""
import time
import os
import fitz
from pathlib import Path

# Data setup: Ensure a realistic 5-unit textbook PDF exists
SAMPLE_PDF_PATH = "data/samples/benchmark_textbook.pdf"

def create_benchmark_pdf(path=SAMPLE_PDF_PATH):
    if os.path.exists(path):
        return path
    doc = fitz.open()
    units = [
        ("Unit 1: Algorithmic Foundations", [
            "Divide and conquer algorithms partition problems into independent subproblems. Merge sort and quicksort exemplify this paradigm with distinct partition strategies.",
            "Recurrence relations characterize the time complexity of recursive divide-and-conquer routines. The Master Theorem provides asymptotic bounds for common recurrences.",
            "Dynamic programming solves optimization problems by memoizing overlapping subproblems. Optimal substructure guarantees that local optimal choices yield global solutions.",
            "Greedy algorithms make locally optimal decisions at each decision boundary. Matroid theory formalizes the structural conditions under which greedy strategies succeed."
        ]),
        ("Unit 2: Computer Systems and Architecture", [
            "Memory hierarchies bridge the processor-memory speed gap through multi-level caching. Cache coherence protocols like MESI maintain state consistency across multi-core CPUs.",
            "Pipelining increases instruction throughput by overlapping instruction execution phases. Structural, data, and control hazards introduce pipeline stalls requiring forwarding.",
            "Virtual memory decouples physical DRAM addressing using page tables and TLBs. Page replacement policies like LRU and clock algorithms minimize page fault overhead.",
            "Instruction-level parallelism leverages out-of-order execution and branch prediction. Speculative execution pipelines instructions ahead of conditional branches."
        ]),
        ("Unit 3: Database Management Systems", [
            "Relational algebra provides the formal foundation for declarative SQL query evaluation. Selection, projection, and join operations formulate query execution plans.",
            "Database normalization decomposes relations to eliminate update, insertion, and deletion anomalies. Boyce-Codd Normal Form ensures every determinant is a candidate key.",
            "Transaction processing guarantees Atomicity, Consistency, Isolation, and Durability (ACID). Two-phase locking protocols prevent concurrency anomalies such as dirty reads.",
            "Query optimization converts relational expressions into cost-effective execution trees. Histogram statistics estimate intermediate cardinality for join ordering."
        ]),
        ("Unit 4: Computer Networks and Protocols", [
            "The OSI and TCP/IP layered architectures establish modular communication standards. Encapsulation and decapsulation govern packet flow across network protocol stacks.",
            "Transmission Control Protocol (TCP) provides reliable byte-stream delivery with flow control. Congestion control algorithms like Tahoe, Reno, and BBR adapt window sizes.",
            "Routing algorithms discover optimal communication paths across network topologies. Link-state algorithms like Dijkstra compute shortest paths using global link states.",
            "Network security protocols implement authentication and integrity via cryptographic primitives. Transport Layer Security (TLS) establishes encrypted communication channels."
        ]),
        ("Unit 5: Artificial Intelligence and Learning", [
            "Supervised learning algorithms map input feature vectors to discrete or continuous targets. Linear regression and support vector machines optimize distinct empirical loss functions.",
            "Unsupervised learning identifies latent patterns in unlabeled dataset distributions. K-means clustering iteratively partitions feature vectors around centroid coordinates.",
            "Deep neural networks learn hierarchical representations through backpropagation of errors. Activation functions like ReLU introduce nonlinear expressive capacity into deep networks.",
            "Evaluation metrics like precision, recall, and F1-score quantify model discrimination performance. Cross-validation techniques assess model generalization across out-of-sample data."
        ])
    ]

    for unit_title, sections in units:
        page = doc.new_page()
        page.insert_text((50, 50), f"Chapter {unit_title}", fontsize=15)
        y = 90
        for i, sec in enumerate(sections, 1):
            page.insert_text((50, y), f"Section {i}: Technical Details", fontsize=12)
            y += 25
            # Repeat paragraph to reach standard chunk word count (~100 words)
            full_text = f"{sec} In this module, students learn foundational theories, formal definitions, and systematic approaches. Furthermore, formal mathematical invariants ensure computational robustness across edge cases. Real-world engineering implementations utilize robust fault-tolerant designs to achieve optimal efficiency."
            page.insert_textbox(fitz.Rect(50, y, 550, y + 100), full_text, fontsize=10)
            y += 120

    doc.save(path)
    doc.close()
    return path

def profile_run(use_mock=False):
    pdf_path = create_benchmark_pdf()
    timings = {}
    details = []

    print(f"\n--- Starting Profile Run (use_mock={use_mock}) ---")

    # a. Model loading
    t0 = time.perf_counter()
    from retrieval.co_mapper import _m
    model = _m()
    if use_mock:
        from generation.mock_client import MockClient
        client = MockClient()
    else:
        from generation.ollama_client import OllamaClient
        client = OllamaClient()
        _ = client.is_up()
    timings["a. Model Loading (SentenceTransformer + Client)"] = time.perf_counter() - t0

    # b. Textbook ingestion / chunking / embedding
    t0 = time.perf_counter()
    from ingestion.library import load_library
    lib = load_library([pdf_path])
    raw_chunks = lib["chunks"]
    timings["b. Textbook Ingestion & Chunking"] = time.perf_counter() - t0

    # c. CO mapping and retrieval
    t0 = time.perf_counter()
    from retrieval.co_mapper import score_chunks_against_outcomes
    from retrieval.sources import select_chunks_for_unit
    outcomes = {
        "CO1": "Understand foundational concepts and algorithmic paradigms.",
        "CO2": "Analyze system architectures and hardware trade-offs.",
        "CO3": "Apply database normalization and transaction management.",
        "CO4": "Evaluate networking protocols and security mechanisms.",
        "CO5": "Design machine learning models and representation learning pipelines."
    }
    scored_chunks = score_chunks_against_outcomes(raw_chunks, outcomes)
    _ = [select_chunks_for_unit(scored_chunks, None, u) for u in range(1, 6)]
    timings["c. CO Mapping & Retrieval"] = time.perf_counter() - t0

    # d. Blueprint allocation
    t0 = time.perf_counter()
    from blueprint import build_blueprint
    bp = build_blueprint(preset=50, units=5, questions_per_unit=1, seed=42)
    slots = bp["slots"]
    timings["d. Blueprint Allocation"] = time.perf_counter() - t0

    # e. Generation & LLM calls
    t0 = time.perf_counter()
    from generation.generator import generate_paper_by_slots
    
    # We will instrument client.generate to record individual call metrics
    orig_generate = client.generate
    llm_call_stats = []
    
    import threading
    stats_lock = threading.Lock()
    
    def instrumented_generate(prompt, *args, **kwargs):
        t_call_start = time.perf_counter()
        call_type = "model_answer" if "Model Answer:" in prompt else "question_draft"
        prompt_len = len(prompt)
        prompt_words = len(prompt.split())
        resp = orig_generate(prompt, *args, **kwargs)
        dur = time.perf_counter() - t_call_start
        with stats_lock:
            llm_call_stats.append({
                "type": call_type,
                "chars": prompt_len,
                "words": prompt_words,
                "duration": dur,
                "resp_len": len(resp)
            })
        return resp
        
    client.generate = instrumented_generate

    results = generate_paper_by_slots(
        chunks=scored_chunks,
        slots=slots,
        client=client,
        seed=42,
    )
    timings["e. LLM Generation (Questions + Answers)"] = time.perf_counter() - t0

    # f. Totals and export
    t0 = time.perf_counter()
    from totals import summarize
    from export_utils import create_docx_exam_paper, create_pdf_exam_paper
    questions = results["questions"]
    tot_summary = summarize(questions, attempted_only=True)
    _ = create_docx_exam_paper(
        questions=questions,
        course_name="Computer Science Benchmark",
        subject_code="CS101",
        outcomes=outcomes,
        include_answers=True,
        include_totals=True,
        totals_summary=tot_summary
    )
    _ = create_pdf_exam_paper(
        questions=questions,
        course_name="Computer Science Benchmark",
        subject_code="CS101",
        outcomes=outcomes,
        include_answers=True,
        include_totals=True,
        totals_summary=tot_summary
    )
    timings["f. Totals & Export (Docx + PDF)"] = time.perf_counter() - t0

    total_time = sum(timings.values())
    
    print("\n" + "="*70)
    print(f"{'Pipeline Stage':<45} | {'Seconds':<10} | {'% of Total':<10}")
    print("="*70)
    for stage, sec in timings.items():
        pct = (sec / total_time) * 100 if total_time > 0 else 0
        print(f"{stage:<45} | {sec:>8.3f}s  | {pct:>8.2f}%")
    print("="*70)
    print(f"{'TOTAL END-TO-END TIME':<45} | {total_time:>8.3f}s  | 100.00%")
    print("="*70)

    if llm_call_stats:
        print("\n--- LLM Calls Detailed Breakdown ---")
        q_calls = [c for c in llm_call_stats if c["type"] == "question_draft"]
        a_calls = [c for c in llm_call_stats if c["type"] == "model_answer"]
        print(f"Total LLM Calls: {len(llm_call_stats)} (Questions: {len(q_calls)}, Answers: {len(a_calls)})")
        if q_calls:
            avg_q_dur = sum(c["duration"] for c in q_calls) / len(q_calls)
            avg_q_chars = sum(c["chars"] for c in q_calls) / len(q_calls)
            print(f"Question Draft Calls: {len(q_calls)} calls, avg {avg_q_dur:.3f}s/call, avg prompt {avg_q_chars:.0f} chars")
        if a_calls:
            avg_a_dur = sum(c["duration"] for c in a_calls) / len(a_calls)
            avg_a_chars = sum(c["chars"] for c in a_calls) / len(a_calls)
            print(f"Model Answer Calls:   {len(a_calls)} calls, avg {avg_a_dur:.3f}s/call, avg prompt {avg_a_chars:.0f} chars")
        print("="*70 + "\n")

    return timings, llm_call_stats

if __name__ == "__main__":
    profile_run(use_mock=False)

