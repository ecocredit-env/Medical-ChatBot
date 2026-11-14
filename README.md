# Medical ChatBot

A health information chatbot written in Python. You describe a symptom in
plain language and it replies with self-care steps and the warning signs that
mean you should see a doctor. It runs as a web app or in the terminal, fully
offline, with no API keys.

> **This is not a doctor.** It gives general information only. It cannot
> diagnose, prescribe, or replace medical care. In an emergency call 112.

## What it can do

- **Understands 18 common health topics:** fever, headache, cold and cough,
  sore throat, diarrhoea, vomiting, constipation, acidity, minor cuts, minor
  burns, sprains, back pain, mild allergy, dehydration, sleep, stress, blood
  pressure and diabetes.
- **Spots emergencies first.** Messages about chest pain, trouble breathing,
  heavy bleeding, stroke signs, seizures, poisoning and similar get an
  immediate "call 112" reply. Messages about self-harm get a supportive reply
  with a helpline.
- **Extracts details from the message.** How long it has lasted ("for 4
  days"), a temperature ("102 F"), and whether it is about a baby or child.
  It uses them: a fever of 4 days or 39.4 C and above is flagged for a doctor
  visit.
- **Remembers the conversation.** After "I burned my hand", the follow-up
  "when should I see a doctor?" is answered for burns.
- **Handles two symptoms and typos.** "i have fevr and hedache" is answered
  for both fever and headache.
- **Knows its limits.** It declines to give medicine doses or a diagnosis,
  and says so when it does not understand instead of guessing.

## How it works

Each message passes through these steps in order:

| Step | Module | What it does |
|---|---|---|
| 1. Preprocessing | `preprocess.py` | Lower-cases, expands contractions, unifies spellings ("loose motions" becomes "diarrhoea") |
| 2. Safety rules | `safety.py` | Red-flag phrases stop everything and return emergency or crisis help |
| 3. Entity extraction | `entities.py` | Duration, temperature and age group, by regular expressions |
| 4. Rule-based intents | `classifier.py` | Keyword and phrase matching, with conservative typo correction |
| 5. ML back-up | `classifier.py` | TF-IDF character n-grams + logistic regression (scikit-learn), used only when no rule matches and confidence is at least 0.35 |
| 6. Response | `bot.py` | Combines the topic, the extracted details and the conversation context |

The health content lives in one file, `medical_chatbot/data/intents.json`, so
topics can be added or edited without touching the code.

## Run it

```bash
pip install -r requirements.txt

python app.py     # web app at http://127.0.0.1:5000
python cli.py     # or chat in the terminal
pytest            # run the 43 tests
```

The web app also exposes a small JSON API:

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I have a fever of 103 F for 4 days"}'
```

## Example

```
You: I have a fever for 4 days
Bot: You said this has lasted 4 days. That is longer than usual for fever,
     so please see a doctor.
     ...
You: what can I do meanwhile?
Bot: What usually helps:
     - Rest and drink plenty of fluids.
     ...
```

## Testing

43 automated tests cover preprocessing, entity extraction, the safety rules,
conversation context and the web API.

Intent recognition is checked on 61 held-out phrases
(`tests/eval_phrases.py`) that are worded differently from the bot's training
examples. It gets 58 of 61 right (95%), including all 8 out-of-scope
questions, which it correctly refuses to guess. The confidence threshold was
chosen using this same set, so treat 95% as an optimistic figure.

## Limitations

- **Small, hand-written knowledge base.** It covers common minor problems
  only, and the content has not been reviewed by a medical professional.
- **Keyword rules do not understand negation.** "I have no fever" is still
  treated as a fever question, and "no chest pain" still triggers the
  emergency reply. The safety rules lean towards false alarms on purpose.
- **Small training set.** The ML model learns from about 350 short example
  phrases, so unusual wording may not be understood.
- **English only**, and conversation memory is kept in server memory, so it
  is lost when the server restarts.
- Emergency number and helpline default to India (112, Tele-MANAS 14416). Set
  `CHATBOT_EMERGENCY_NUMBER` and `CHATBOT_CRISIS_LINE` to change them.

## Project layout

```
app.py                     Flask web app and JSON API
cli.py                     Terminal chat
templates/index.html       Chat page
medical_chatbot/
    preprocess.py          Text normalisation
    safety.py              Emergency and crisis rules
    entities.py            Duration, temperature, age group
    classifier.py          Rule-based + ML intent classification
    bot.py                 Conversation logic
    data/intents.json      Health topics and example phrases
tests/                     Test suite and held-out phrases
```
