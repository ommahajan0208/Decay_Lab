import os
import time
import math
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory

from decay_lab.memory_store import MemoryStore
from decay_lab.decay_engine import DecayEngine, SMART_BRAIN, DUMB_BRAIN, ADAPTIVE_BRAIN
from decay_lab.retrieval import Retriever
from decay_lab.visualization import compute_strength_series
from decay_lab.simulations import run_both, EXAM_TOPICS, STUDY_SCHEDULE

app = Flask(__name__, static_folder="web")

# ── Core Components ────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "data", "memories.json")
store = MemoryStore(path=data_path)
decay_engine = DecayEngine(profile=SMART_BRAIN)
retriever = Retriever(store=store, decay_engine=decay_engine)

# ── Global Simulation State ────────────────────────────────────
global_time_offset: float = 0.0          # extra seconds added to now()
graveyard: List[Dict] = []               # all pruned memories with cause-of-death
bandit_history: List[Dict] = []          # RL feedback history for dashboard

# Models list for bandit dashboard labels
DECAY_MODEL_NAMES = ["HLR", "Power-Law", "Reinforcement"]


def sim_now() -> float:
    """Current simulated time = real time + offset."""
    return time.time() + global_time_offset


def _fmt_sim_time(t: float) -> str:
    """Format offset as 'Day X, HH:MM'."""
    offset = t - (time.time() - global_time_offset)  # relative to sim start
    days = int(offset // 86400)
    hours = int((offset % 86400) // 3600)
    minutes = int((offset % 3600) // 60)
    return f"Day {days}, {hours:02d}:{minutes:02d}"


# ── Static Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/favicon.ico")
def favicon():
    return "", 204


# ── Time Control API ───────────────────────────────────────────
@app.route("/api/time")
def get_time():
    t = sim_now()
    return jsonify({
        "offset_seconds": global_time_offset,
        "offset_hours": global_time_offset / 3600,
        "simulated_ts": t,
        "display": _fmt_sim_time(t),
    })

@app.route("/api/time/advance", methods=["POST"])
def advance_time():
    global global_time_offset, graveyard
    data = request.json or {}
    hours = float(data.get("hours", 1.0))
    global_time_offset += hours * 3600.0

    # Run garbage collect at new sim time and add to graveyard
    pruned_count, pruned_list = store.garbage_collect(decay_engine, now=sim_now())
    graveyard.extend(pruned_list)

    return jsonify({
        "offset_hours": global_time_offset / 3600,
        "display": _fmt_sim_time(sim_now()),
        "pruned": pruned_count,
        "new_graveyard": pruned_list,
    })

@app.route("/api/time/reset", methods=["POST"])
def reset_time():
    global global_time_offset, graveyard
    global_time_offset = 0.0
    graveyard = []
    return jsonify({"status": "reset", "display": _fmt_sim_time(sim_now())})


# ── Sleep Cycle API ────────────────────────────────────────────
@app.route("/api/sleep", methods=["POST"])
def sleep_cycle():
    global global_time_offset, graveyard
    global_time_offset += 8 * 3600.0   # +8 hours
    result = store.apply_sleep_consolidation(decay_engine, now=sim_now())
    graveyard.extend(result["pruned"])
    return jsonify({
        "offset_hours": global_time_offset / 3600,
        "display": _fmt_sim_time(sim_now()),
        "consolidated": result["consolidated"],
        "pruned": result["pruned_count"],
        "new_graveyard": result["pruned"],
    })


# ── Graveyard API ──────────────────────────────────────────────
@app.route("/api/graveyard")
def get_graveyard():
    return jsonify({"graveyard": graveyard})


# ── Brain Profile API ──────────────────────────────────────────
@app.route("/api/profile", methods=["POST"])
def set_profile():
    data = request.json
    profile_name = data.get("profile", "smart")
    if profile_name == "smart":
        decay_engine.set_profile(SMART_BRAIN)
    elif profile_name == "dumb":
        decay_engine.set_profile(DUMB_BRAIN)
    else:
        decay_engine.set_profile(ADAPTIVE_BRAIN)

    pruned_count, pruned_list = store.garbage_collect(decay_engine, now=sim_now())
    graveyard.extend(pruned_list)
    return jsonify({"status": "success", "profile": profile_name, "pruned": pruned_count})


# ── RL Feedback API ────────────────────────────────────────────
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.json
    reward = float(data.get("reward", 0.0))
    old_weights = decay_engine.bandit.get_weights()[:]
    decay_engine.bandit.update(reward)
    new_weights = decay_engine.bandit.get_weights()

    entry = {
        "reward": reward,
        "old_weights": [round(w, 3) for w in old_weights],
        "new_weights": [round(w, 3) for w in new_weights],
        "delta": [round(new_weights[i] - old_weights[i], 3) for i in range(3)],
    }
    bandit_history.append(entry)

    return jsonify({
        "status": "success",
        "current_weights": new_weights,
        "entry": entry,
    })

@app.route("/api/bandit")
def bandit_state():
    b = decay_engine.bandit
    weights = b.get_weights()
    return jsonify({
        "scores": [round(s, 4) for s in b.scores],
        "weights": [round(w, 4) for w in weights],
        "temperature": b.temperature,
        "learning_rate": b.learning_rate,
        "model_names": DECAY_MODEL_NAMES,
        "history": bandit_history[-10:],  # last 10 feedback events
    })

@app.route("/api/bandit/tune", methods=["POST"])
def tune_bandit():
    data = request.json or {}
    if "temperature" in data:
        decay_engine.bandit.temperature = max(0.1, float(data["temperature"]))
    if "learning_rate" in data:
        decay_engine.bandit.learning_rate = max(0.01, float(data["learning_rate"]))
    return jsonify({"status": "ok", "temperature": decay_engine.bandit.temperature,
                    "learning_rate": decay_engine.bandit.learning_rate})


# ── Memories API ───────────────────────────────────────────────
@app.route("/api/memories")
def get_memories():
    now = sim_now()
    memories = store.list_all()
    result = []
    for m in memories:
        s = decay_engine.effective_strength(m, now=now)
        result.append({**m.to_dict(), "current_strength": round(s, 4)})
    return jsonify({"memories": result})

@app.route("/api/memories/add", methods=["POST"])
def add_memory():
    """Add a new memory, with interference detection."""
    global graveyard
    data = request.json or {}
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "content required"}), 400

    from decay_lab.models import Memory as MemModel
    import uuid

    now = sim_now()
    new_mem = MemModel.new(
        id=f"mem_{uuid.uuid4().hex[:8]}",
        content=content,
        strength=1.0,
        created_at=now,
    )

    # Interference detection via SBERT if available
    interference_events = []
    if retriever.use_semantic and retriever._bi_encoder is not None:
        from sentence_transformers.util import cos_sim
        existing = store.list_all()
        if existing:
            new_emb = retriever._bi_encoder.encode(content, convert_to_tensor=True)
            existing_embs = retriever._bi_encoder.encode([m.content for m in existing], convert_to_tensor=True)
            sims = cos_sim(new_emb, existing_embs)[0]
            for i, m in enumerate(existing):
                sim = float(sims[i])
                if sim > 0.85:
                    penalty = round(sim * 0.3, 3)
                    m.strength = max(0.0, m.strength - penalty)
                    store.upsert(m)
                    interference_events.append({
                        "id": m.id,
                        "content": m.content,
                        "similarity": round(sim, 3),
                        "penalty": penalty,
                    })

    store.upsert(new_mem)
    return jsonify({
        "status": "added",
        "id": new_mem.id,
        "interference": interference_events,
    })


# ── Search & Retrieval ─────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
def search():
    global graveyard
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"results": []})

    now = sim_now()
    results, _ = retriever.retrieve_and_touch(query, limit=5, min_relevance=0.01, now=now)

    pruned_count, pruned_list = store.garbage_collect(decay_engine, now=now)
    graveyard.extend(pruned_list)

    response = []
    for r in results:
        m = r.memory
        response.append({
            "id": m.id,
            "content": m.content,
            "score": r.score,
            "relevance": r.relevance,
            "strength": r.strength,
            "last_accessed": m.last_accessed_at or m.created_at,
        })
    return jsonify({"results": response, "pruned": pruned_count, "new_graveyard": pruned_list})


# ── Brain Race API ─────────────────────────────────────────────
@app.route("/api/brain_race")
def brain_race():
    """Compute strengths for all 3 profiles simultaneously at sim_now."""
    now = sim_now()
    memories = store.list_all()
    profiles = [
        ("smart", SMART_BRAIN),
        ("dumb", DUMB_BRAIN),
        ("adaptive", ADAPTIVE_BRAIN),
    ]

    result = {}
    for name, profile in profiles:
        eng = DecayEngine(profile=profile)
        mem_data = []
        alive = 0
        total_strength = 0.0
        for m in memories:
            s = eng.effective_strength(m, now=now)
            alive_flag = s >= profile.prune_threshold
            if alive_flag:
                alive += 1
            total_strength += s
            mem_data.append({
                "id": m.id,
                "content": m.content[:40],
                "strength": round(s, 4),
                "alive": alive_flag,
            })
        result[name] = {
            "memories": mem_data,
            "alive": alive,
            "total": len(memories),
            "avg_strength": round(total_strength / max(1, len(memories)), 4),
        }

    return jsonify(result)


# ── Decay Chart Series ─────────────────────────────────────────
@app.route("/api/series")
def get_series():
    now = sim_now()
    series = compute_strength_series(store, decay_engine, query="", steps=20, horizon_seconds=43200, base_now=now)

    labels = []
    datasets_map = {}
    for time_offset, strengths in series:
        hours = time_offset / 3600
        labels.append(f"+{hours:.1f}h")
        for mid, strength in strengths.items():
            if mid not in datasets_map:
                datasets_map[mid] = []
            datasets_map[mid].append(strength)

    # Add Ebbinghaus overlay: R = e^(-t/S), S=1.84 in hours
    ebbinghaus = []
    for i in range(20):
        t_hours = (i / 19) * 12
        R = math.exp(-t_hours / 1.84)
        ebbinghaus.append(round(R, 4))

    datasets = []
    memories = {m.id: m for m in store.list_all()}
    for mid, data in datasets_map.items():
        m = memories.get(mid)
        content_preview = m.content[:28] + "..." if m and len(m.content) > 28 else (m.content if m else mid)
        datasets.append({
            "label": f"{content_preview}",
            "memoryId": mid,
            "data": data,
            "tension": 0.4,
        })

    return jsonify({
        "labels": labels,
        "datasets": datasets,
        "ebbinghaus": ebbinghaus,
        "sim_time": _fmt_sim_time(now),
        "offset_hours": global_time_offset / 3600,
    })


# ── Student Simulation ─────────────────────────────────────────
@app.route("/api/simulation/student")
def student_simulation():
    smart, dumb = run_both()

    def format_result(result):
        return {
            "profile": result.profile_name,
            "remembered_count": result.remembered_count,
            "total": result.total,
            "timeline": result.timeline,
            "snapshots": [
                {
                    "id": s.id,
                    "content": s.content,
                    "strength_at_exam": s.strength_at_exam,
                    "recalled_on_days": s.recalled_on_days,
                    "remembered": s.remembered,
                }
                for s in result.snapshots
            ],
        }

    return jsonify({
        "smart": format_result(smart),
        "dumb": format_result(dumb),
        "study_schedule": {str(d): ids for d, ids in STUDY_SCHEDULE.items()},
        "topics": [{"id": mid, "content": content} for mid, content in EXAM_TOPICS],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
