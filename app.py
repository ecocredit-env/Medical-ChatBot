"""Web interface for the medical chatbot.

    python app.py          then open http://127.0.0.1:5000
"""
import os
import secrets
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request, session

from medical_chatbot import ChatBot, Session
from medical_chatbot.safety import EMERGENCY_NUMBER

MAX_SESSIONS = 1000       # oldest conversations are dropped beyond this
MAX_MESSAGE_LENGTH = 500

app = Flask(__name__)
app.secret_key = os.environ.get("CHATBOT_SECRET_KEY") or secrets.token_hex(32)
bot = ChatBot()
_sessions: "OrderedDict[str, Session]" = OrderedDict()


def _get_session() -> Session:
    sid = session.get("sid")
    if sid is None or sid not in _sessions:
        sid = secrets.token_hex(16)
        session["sid"] = sid
        _sessions[sid] = Session()
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(sid)
    return _sessions[sid]


@app.get("/")
def index():
    return render_template("index.html", emergency_number=EMERGENCY_NUMBER)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Send JSON like {\"message\": \"I have a fever\"}."), 400
    reply = bot.reply(message[:MAX_MESSAGE_LENGTH], _get_session())
    return jsonify(reply.as_dict())


@app.post("/api/reset")
def reset():
    sid = session.pop("sid", None)
    _sessions.pop(sid, None)
    return jsonify(ok=True)


@app.get("/health")
def health():
    return jsonify(status="ok", topics=len(bot.topics))


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")), debug=False)
