import os
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import base64

app = Flask(__name__, static_folder=".")
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), "badfriend.db")

# ── OpenAI ────────────────────────────────────────
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if os.environ.get("OPENAI_API_KEY") else None
except ImportError:
    openai_client = None

# ── Database ──────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT    NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT   NOT NULL
        );
        CREATE TABLE IF NOT EXISTS moods (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            mood      TEXT NOT NULL,
            note      TEXT NOT NULL DEFAULT '',
            date      TEXT NOT NULL,
            time      TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sleep_records (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            bedtime        TEXT NOT NULL,
            wakeup         TEXT NOT NULL,
            duration_hours REAL NOT NULL,
            insight        TEXT NOT NULL,
            date           TEXT NOT NULL,
            timestamp      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS habits (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            emoji      TEXT NOT NULL DEFAULT '⭐',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS habit_logs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
            date     TEXT NOT NULL,
            UNIQUE(habit_id, date)
        );
    """)
    conn.commit()
    conn.close()

# ── Helpers ───────────────────────────────────────

def row_to_dict(row):
    return dict(row) if row else None

def rows_to_list(rows):
    return [dict(r) for r in rows]

def now_parts():
    n = datetime.datetime.now()
    return n.strftime("%Y-%m-%d"), n.strftime("%H:%M"), n.isoformat()

# ── Frontend ──────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── Chat ──────────────────────────────────────────

SYSTEM_PROMPT = """You are Bad Friend — a close, honest, and emotionally intelligent AI companion.
You are NOT a formal assistant. You talk like a real friend who genuinely cares.

Your personality:
- Warm, casual, and natural — never robotic or formal
- Empathetic first: you always acknowledge feelings before giving advice
- Honest but never harsh — you tell the truth kindly
- Curious: you ask follow-up questions to understand better
- Practical: after listening, you offer real, actionable guidance
- Supportive across all areas: productivity, study, work, emotions, sleep, habits, mood

How you respond:
- Start by acknowledging what the person is feeling or saying
- Ask a follow-up question when something needs more context
- Give concrete advice when the person seems ready for it
- Keep responses conversational — like texting a friend, not writing an essay
- Never use bullet points or formal headers in chat responses
- Mix emotional support with practical help naturally
- Use "I" and "you" like a real person
- Occasionally use casual language like "honestly", "look", "here's the thing"

You can speak both Arabic and English fluently. Match the language the user writes in.

Remember: you're a friend, not a therapist or a bot. Keep it real."""


def fallback_response(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["حزين","زهقت","مكتئب","وحيد","sad","depressed","crying","hopeless","lonely","alone"]):
        return "والله أنا سامعك وحاسس إن في شي تقيل على قلبك. ما لازم تشيل هالحمل لحالك — قولي أكثر، إيش اللي صاير معك؟"
    if any(w in msg for w in ["متوتر","ضغط","قلقان","stressed","anxious","anxiety","panic","worried","nervous","stress"]):
        return "Okay, let's slow down together for a second. When you're stressed like this, your brain goes into overdrive — that's exhausting. What's the biggest thing pressing on you right now? Let's pick that one thing apart and figure out what's actually in your control."
    if any(w in msg for w in ["غاضب","زعلان","angry","frustrated","mad","annoyed"]):
        return "Honestly, it sounds like something really got to you — and that frustration is valid. What happened? Walk me through it. Sometimes just saying it out loud helps untangle things."
    if any(w in msg for w in ["كسلان","ما أقدر أركز","procrastinat","cant focus","can't focus","distracted","lazy","unmotivated"]):
        return "Okay real talk — procrastination is almost never about laziness. It's usually about fear, overwhelm, or just not knowing where to start. What's the one thing you've been avoiding the most? Just that one thing."
    if any(w in msg for w in ["مذاكرة","امتحان","study","studying","exam","test","homework","school","university"]):
        return "Studying is hard when your brain doesn't cooperate. Try this: 25 minutes of focused work, then a 5-minute break — the Pomodoro method. What subject is giving you the most trouble right now?"
    if any(w in msg for w in ["شغل","وظيفة","work","job","boss","deadline","career"]):
        return "Work stuff can really drain you — especially when it feels never-ending. What's the pressure point right now? Is it the workload, the people, or just feeling stuck?"
    if any(w in msg for w in ["نوم","أرق","تعبان","sleep","insomnia","tired","exhausted","fatigue"]):
        return "Sleep struggles are real and they affect everything. No screens 30 min before bed, keep the room cool, and wake up at the same time every day. Your body loves routine. What time are you usually trying to sleep?"
    if any(w in msg for w in ["عادة","روتين","habit","routine","exercise","workout","gym"]):
        return "Building habits is about making them so small they're impossible to fail. Don't say 'I'll exercise every day' — say 'I'll put on my shoes.' What habit are you trying to build?"
    if any(w in msg for w in ["هلا","مرحبا","اهلا","السلام","hi","hello","hey","sup"]):
        return "هلا! أنا هنا 😊 كيف حالك اليوم؟ وش في بالك؟"
    if any(w in msg for w in ["شكرا","ممنون","thanks","thank you"]):
        return "عادي، هذا اللي أنا موجود عشانه. في شي ثاني تبي تحكي فيه؟"
    if any(w in msg for w in ["بطل","ممتاز","رائع","great","awesome","amazing","good"]):
        return "هذا الكلام! أنا فخور فيك. اشرح لي أكثر — إيش اللي صار؟"
    return "أنا مصغي. قولي أكثر — إيش اللي صاير معك الحين؟ أبي أفهم الصورة كاملة."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    if openai_client:
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-10:]:
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_message})
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=500, temperature=0.85)
            return jsonify({"reply": response.choices[0].message.content.strip()})
        except Exception:
            pass
    return jsonify({"reply": fallback_response(user_message)})

# ── Tasks ─────────────────────────────────────────

@app.route("/tasks", methods=["GET"])
def get_tasks():
    db = get_db()
    rows = db.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
    return jsonify({"tasks": rows_to_list(rows)})

@app.route("/tasks", methods=["POST"])
def add_task():
    data = request.get_json() or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    date, time_, ts = now_parts()
    db = get_db()
    cur = db.execute("INSERT INTO tasks (title, completed, created_at) VALUES (?,0,?)", (title, ts))
    db.commit()
    task = row_to_dict(db.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone())
    task["completed"] = bool(task["completed"])
    return jsonify({"task": task}), 201

@app.route("/tasks/<int:task_id>", methods=["PATCH"])
def update_task(task_id):
    data = request.get_json() or {}
    db = get_db()
    row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return jsonify({"error": "Task not found"}), 404
    if "completed" in data:
        db.execute("UPDATE tasks SET completed=? WHERE id=?", (1 if data["completed"] else 0, task_id))
        db.commit()
    task = row_to_dict(db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    task["completed"] = bool(task["completed"])
    return jsonify({"task": task})

@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    db = get_db()
    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    db.commit()
    return jsonify({"success": True})

# ── Mood ──────────────────────────────────────────

@app.route("/mood", methods=["GET"])
def get_mood():
    db = get_db()
    rows = db.execute("SELECT * FROM moods ORDER BY id DESC LIMIT 30").fetchall()
    return jsonify({"moods": rows_to_list(rows)})

@app.route("/mood", methods=["POST"])
def save_mood():
    data = request.get_json() or {}
    mood = data.get("mood", "").strip()
    note = data.get("note", "").strip()
    valid = ["happy", "neutral", "sad", "stressed"]
    if mood not in valid:
        return jsonify({"error": f"Mood must be one of {valid}"}), 400
    date, time_, ts = now_parts()
    db = get_db()
    cur = db.execute("INSERT INTO moods (mood, note, date, time, timestamp) VALUES (?,?,?,?,?)",
                     (mood, note, date, time_, ts))
    db.commit()
    entry = row_to_dict(db.execute("SELECT * FROM moods WHERE id=?", (cur.lastrowid,)).fetchone())
    return jsonify({"entry": entry}), 201

# ── Sleep ─────────────────────────────────────────

def parse_time(t):
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            return datetime.datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None

def sleep_insight(hours):
    if hours < 5:   return "نوم قصير جداً — جسمك يحتاج أكثر من كذا. حاول تنام أبكر الليلة."
    if hours < 6:   return "You're running on low fuel. Chronic under-sleep builds up and hits hard. Can you get to bed a bit earlier?"
    if hours < 7:   return "Getting closer! Aim for that 7-8 hour sweet spot. Small improvements add up over time."
    if hours <= 9:  return "Solid sleep! That's the range where your brain consolidates memories and your body recovers best. Keep it up."
    return "That's a long sleep — sometimes a sign your body was catching up. How do you feel? Rested or still groggy?"

@app.route("/sleep", methods=["GET"])
def get_sleep():
    db = get_db()
    rows = db.execute("SELECT * FROM sleep_records ORDER BY id DESC LIMIT 30").fetchall()
    return jsonify({"records": rows_to_list(rows)})

@app.route("/sleep", methods=["POST"])
def save_sleep():
    data = request.get_json() or {}
    bedtime_str = data.get("bedtime", "").strip()
    wakeup_str  = data.get("wakeup", "").strip()
    if not bedtime_str or not wakeup_str:
        return jsonify({"error": "Both times required"}), 400
    bedtime = parse_time(bedtime_str)
    wakeup  = parse_time(wakeup_str)
    if not bedtime or not wakeup:
        return jsonify({"error": "Invalid time format. Use HH:MM"}), 400
    if wakeup <= bedtime:
        wakeup += datetime.timedelta(days=1)
    hours   = round((wakeup - bedtime).total_seconds() / 3600, 1)
    insight = sleep_insight(hours)
    date, _, ts = now_parts()
    db = get_db()
    cur = db.execute(
        "INSERT INTO sleep_records (bedtime, wakeup, duration_hours, insight, date, timestamp) VALUES (?,?,?,?,?,?)",
        (bedtime_str, wakeup_str, hours, insight, date, ts))
    db.commit()
    record = row_to_dict(db.execute("SELECT * FROM sleep_records WHERE id=?", (cur.lastrowid,)).fetchone())
    return jsonify({"record": record}), 201

# ── Habits ────────────────────────────────────────

@app.route("/habits", methods=["GET"])
def get_habits():
    db = get_db()
    today = datetime.date.today().isoformat()
    habits = rows_to_list(db.execute("SELECT * FROM habits ORDER BY id ASC").fetchall())
    for h in habits:
        logged_dates = [r["date"] for r in db.execute(
            "SELECT date FROM habit_logs WHERE habit_id=? ORDER BY date DESC LIMIT 7", (h["id"],)).fetchall()]
        h["done_today"] = today in logged_dates
        h["streak"] = _calc_streak(logged_dates)
        h["weekly"] = logged_dates
    return jsonify({"habits": habits})

def _calc_streak(dates):
    if not dates:
        return 0
    sorted_dates = sorted(set(dates), reverse=True)
    today = datetime.date.today()
    streak = 0
    check = today
    for d in sorted_dates:
        if datetime.date.fromisoformat(d) == check:
            streak += 1
            check -= datetime.timedelta(days=1)
        else:
            break
    return streak

@app.route("/habits", methods=["POST"])
def add_habit():
    data = request.get_json() or {}
    name  = data.get("name", "").strip()
    emoji = data.get("emoji", "⭐").strip() or "⭐"
    if not name:
        return jsonify({"error": "Name required"}), 400
    _, _, ts = now_parts()
    db = get_db()
    cur = db.execute("INSERT INTO habits (name, emoji, created_at) VALUES (?,?,?)", (name, emoji, ts))
    db.commit()
    habit = row_to_dict(db.execute("SELECT * FROM habits WHERE id=?", (cur.lastrowid,)).fetchone())
    habit["done_today"] = False
    habit["streak"] = 0
    habit["weekly"] = []
    return jsonify({"habit": habit}), 201

@app.route("/habits/<int:habit_id>", methods=["DELETE"])
def delete_habit(habit_id):
    db = get_db()
    db.execute("DELETE FROM habit_logs WHERE habit_id=?", (habit_id,))
    db.execute("DELETE FROM habits WHERE id=?", (habit_id,))
    db.commit()
    return jsonify({"success": True})

@app.route("/habits/<int:habit_id>/log", methods=["POST"])
def log_habit(habit_id):
    today = datetime.date.today().isoformat()
    db = get_db()
    existing = db.execute("SELECT id FROM habit_logs WHERE habit_id=? AND date=?", (habit_id, today)).fetchone()
    if existing:
        db.execute("DELETE FROM habit_logs WHERE habit_id=? AND date=?", (habit_id, today))
        db.commit()
        return jsonify({"done": False})
    db.execute("INSERT INTO habit_logs (habit_id, date) VALUES (?,?)", (habit_id, today))
    db.commit()
    return jsonify({"done": True})

# ── Stats ─────────────────────────────────────────

@app.route("/stats", methods=["GET"])
def get_stats():
    db = get_db()
    today = datetime.date.today().isoformat()
    total_tasks = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    done_tasks  = db.execute("SELECT COUNT(*) FROM tasks WHERE completed=1").fetchone()[0]
    today_mood  = row_to_dict(db.execute(
        "SELECT mood FROM moods WHERE date=? ORDER BY id DESC LIMIT 1", (today,)).fetchone())
    last_sleep  = row_to_dict(db.execute(
        "SELECT duration_hours FROM sleep_records ORDER BY id DESC LIMIT 1").fetchone())
    mood_last7  = rows_to_list(db.execute(
        "SELECT mood, date FROM moods ORDER BY id DESC LIMIT 7").fetchall())
    return jsonify({
        "total_tasks": total_tasks,
        "done_tasks":  done_tasks,
        "today_mood":  today_mood["mood"] if today_mood else None,
        "last_sleep":  last_sleep["duration_hours"] if last_sleep else None,
        "mood_last7":  mood_last7,
    })

# ── Boot ──────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
