from __future__ import annotations

from pathlib import Path
import time

from decay_lab.memory_store import MemoryStore
from decay_lab.decay_engine import DecayEngine
from decay_lab.retrieval import Retriever
from decay_lab.visualization import pretty_print_top


MIN_RELEVANCE = 0.01


def _seed_if_empty(store: MemoryStore) -> None:
    memories = store.list_all()
    if memories:
        return

    now = time.time()
    base = [
        ("m1", "I love pizza and italian food", {"topic": "food"}, 1.0, now - 60 * 60 * 2),
        ("m2", "The weather today is sunny and warm", {"topic": "weather"}, 1.0, now - 60 * 60 * 6),
        ("m3", "Weekly meeting moved to next Tuesday", {"topic": "work"}, 1.2, now - 60 * 30),
        ("m4", "Running helps improve cardiovascular health", {"topic": "fitness"}, 1.3, now - 60 * 60 * 24),
        ("m5", "Study notes: retrieval ranking and pruning", {"topic": "ml"}, 1.5, now - 60 * 60 * 1),
    ]

    for mid, content, tags, strength, created_at in base:
        # Create Memory via direct fields to set created_at deterministically
        from decay_lab.models import Memory

        m = Memory(
            id=mid,
            content=content,
            tags=tags,
            created_at=created_at,
            last_accessed_at=None,
            strength=strength,
            metadata={},
        )
        store.upsert(m)


def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "memories.json"
    store = MemoryStore(path=str(data_path))
    _seed_if_empty(store)

    decay_engine = DecayEngine(lambda_decay=1e-4)
    retriever = Retriever(store=store, decay_engine=decay_engine)

    print("=== Decay-Lab Demo ===")
    print("Stored memories are ranked by:")
    print("  score = lexical_relevance * ensemble_strength")
    print(f"  only showing matches with relevance >= {MIN_RELEVANCE}")
    print()

    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        results, pruned = retriever.retrieve_and_touch(
            query,
            limit=5,
            min_relevance=MIN_RELEVANCE,
            prune_keep_ids=None,
        )
        print()
        if not results:
            print("No relevant memories found.")
        else:
            for r in results:
                m = r.memory
                last = m.last_accessed_at if m.last_accessed_at is not None else m.created_at
                print(
                    f"- {m.id} | score={r.score:.4f} | relevance={r.relevance:.4f} "
                    f"| strength={r.strength:.4f} | last={last:.0f} | {m.content}"
                )
        print()
        if pruned:
            print(f"[store] pruned {pruned} memories")
        pretty_print_top(store, decay_engine, k=3)
        print()


if __name__ == "__main__":
    main()
