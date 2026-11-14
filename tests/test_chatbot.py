"""Run with:  pytest"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_phrases import EVAL
from medical_chatbot import ChatBot, Session
from medical_chatbot.entities import extract_entities
from medical_chatbot.preprocess import normalise
from medical_chatbot import safety


@pytest.fixture(scope="module")
def bot():
    return ChatBot()


# ------------------------------------------------------------ preprocessing
def test_normalise_expands_and_unifies():
    assert normalise("I CAN'T sleep!!") == "i cannot sleep"
    assert normalise("Loose motions & tummy ache") == "diarrhoea stomach ache"
    assert normalise("Temp is 101.5°F.") == "temperature is 101.5 degrees f"


# ----------------------------------------------------------------- entities
@pytest.mark.parametrize("text,days", [
    ("fever for 3 days", 3), ("cough since two weeks", 14),
    ("headache for a couple of days", 2), ("pain since yesterday", 1),
    ("started 12 hours ago", 0.5), ("no duration here", None),
])
def test_duration(text, days):
    assert extract_entities(normalise(text)).duration_days == days


@pytest.mark.parametrize("text,celsius", [
    ("temperature is 102 F", 38.9), ("fever of 39.5 c", 39.5),
    ("my temp is 101", 38.3), ("it is 38.2", 38.2), ("I am 25 years old", None),
])
def test_temperature(text, celsius):
    assert extract_entities(normalise(text)).temperature_c == celsius


def test_age_group():
    assert extract_entities(normalise("my baby is hot")).age_group == "infant"
    assert extract_entities(normalise("my daughter has a cough")).age_group == "child"
    assert extract_entities(normalise("he is 6 years old")).age_group == "child"
    assert extract_entities(normalise("I am 25 years old")).age_group is None


# ------------------------------------------------------------------- safety
@pytest.mark.parametrize("text", [
    "I have chest pain", "he can't breathe", "my father collapsed",
    "she is having a seizure", "the cut won't stop bleeding",
    "his face is drooping and slurred speech", "my child swallowed bleach",
    "I am coughing up blood", "my throat is swelling after a bee sting",
])
def test_emergencies_are_caught(bot, text):
    r = bot.reply(text)
    assert r.intent == "emergency" and r.urgent
    assert safety.EMERGENCY_NUMBER in r.text


@pytest.mark.parametrize("text", [
    "I want to kill myself", "I feel suicidal", "I don't want to live anymore",
])
def test_crisis_messages_get_support(bot, text):
    r = bot.reply(text)
    assert r.intent == "crisis" and r.urgent
    assert "14416" in r.text


@pytest.mark.parametrize("text", [
    "I have a mild headache", "I have a cold", "how to sleep better",
    "I cut myself while shaving",
])
def test_ordinary_messages_are_not_emergencies(bot, text):
    assert not bot.reply(text).urgent


# ------------------------------------------------------- intent recognition
def test_heldout_intent_accuracy(bot):
    """Phrases the bot was not built from; see tests/eval_phrases.py."""
    correct = 0
    for text, expected in EVAL:
        r = bot.reply(text)
        got = None if r.intent == "fallback" else r.intent.split("+")[0]
        correct += got == expected
    assert correct / len(EVAL) >= 0.90


def test_out_of_scope_is_not_guessed(bot):
    out = [t for t, e in EVAL if e is None]
    assert all(bot.reply(t).intent == "fallback" for t in out)


def test_typos_are_corrected(bot):
    assert bot.reply("i have fevr and hedache").intent == "fever+headache"


def test_two_symptoms_in_one_message(bot):
    r = bot.reply("I have a sore throat and a fever")
    assert set(r.intent.split("+")) == {"sore_throat", "fever"}


def test_medication_and_diagnosis_are_declined(bot):
    assert bot.reply("what dose of paracetamol for fever").intent == "medication"
    assert bot.reply("can you diagnose my headache").intent == "diagnosis"


# ------------------------------------------------------------------ context
def test_followup_uses_previous_symptom(bot):
    s = Session()
    bot.reply("I burned my hand", s)
    r = bot.reply("when should I see a doctor?", s)
    assert r.intent == "burn" and r.source == "context"
    assert [x["heading"] for x in r.sections] == ["See a doctor if"]


def test_followup_without_context_asks(bot):
    assert bot.reply("is it serious?").intent == "clarify"


def test_duration_followup_escalates(bot):
    s = Session()
    assert not bot.reply("I have a fever", s).urgent
    assert not bot.reply("it has been 1 day", s).urgent
    r = bot.reply("actually it has been 5 days", s)
    assert r.urgent and "see a doctor" in r.text


def test_long_duration_and_high_temperature_escalate(bot):
    assert bot.reply("cough for 4 weeks").urgent
    assert bot.reply("fever of 104 F").urgent
    assert not bot.reply("fever of 100 F since yesterday").urgent


def test_infant_fever_is_urgent(bot):
    assert bot.reply("my baby has a fever").urgent


# ---------------------------------------------------------------- web app
@pytest.fixture()
def client():
    import app as web
    web.app.config.update(TESTING=True)
    return web.app.test_client()


def test_web_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200 and b"Medical ChatBot" in res.data


def test_api_chat_and_session_context(client):
    r1 = client.post("/api/chat", json={"message": "I sprained my ankle"}).get_json()
    assert r1["intent"] == "sprain" and r1["sections"]
    r2 = client.post("/api/chat", json={"message": "what can I do?"}).get_json()
    assert r2["intent"] == "sprain" and r2["source"] == "context"
    client.post("/api/reset")
    r3 = client.post("/api/chat", json={"message": "what can I do?"}).get_json()
    assert r3["intent"] == "clarify"


def test_api_rejects_bad_input(client):
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
    assert client.post("/api/chat", data="not json").status_code == 400


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"
