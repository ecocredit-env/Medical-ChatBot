"""The chatbot: combines safety rules, intent classification, entity
extraction and conversation context into one reply."""
from dataclasses import dataclass, field

from . import safety
from .classifier import IntentClassifier, Match
from .entities import Entities, extract_entities
from .preprocess import normalise

DISCLAIMER = ("This is general information, not a diagnosis. See a doctor if "
              "you are worried or not improving.")
HIGH_FEVER_C = 39.4
MAX_TOPICS = 2


@dataclass
class Reply:
    text: str
    intent: str
    source: str = "rule"              # rule | ml | context | safety | fallback
    confidence: float = 1.0
    urgent: bool = False
    title: str = ""
    sections: list = field(default_factory=list)   # [{"heading", "items"}]
    notes: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class Session:
    """What the bot remembers within one conversation."""
    last_topic: str | None = None
    turns: int = 0


def _fmt_days(days: float) -> str:
    if days < 1:
        return "less than a day"
    if days >= 60:
        return f"about {round(days / 30)} months"
    if days >= 14:
        return f"about {round(days / 7)} weeks"
    n = round(days)
    return f"{n} day" + ("" if n == 1 else "s")


class ChatBot:
    def __init__(self, classifier: IntentClassifier | None = None):
        self.classifier = classifier or IntentClassifier()
        intents = self.classifier.intents
        self.topics = {t["tag"]: t for t in intents["topics"]}
        self.general = {g["tag"]: g for g in intents["general"]}

    # ---------------------------------------------------------- helpers --
    def _escalation_notes(self, topic: dict, ents: Entities) -> tuple[list, bool]:
        notes, urgent = [], False
        limit = topic.get("escalate_after_days")
        if ents.duration_days is not None and limit:
            if ents.duration_days >= limit:
                notes.append(f"You said this has lasted {_fmt_days(ents.duration_days)}. "
                             f"That is longer than usual for {topic['name']}, so "
                             "please see a doctor.")
                urgent = True
        if ents.temperature_c is not None:
            if ents.temperature_c >= HIGH_FEVER_C:
                notes.append(f"A temperature of {ents.temperature_c:.1f} C is high. "
                             "Please see a doctor today.")
                urgent = True
            elif ents.temperature_c >= 38.0:
                notes.append(f"A temperature of {ents.temperature_c:.1f} C is a fever.")
            elif topic["tag"] == "fever":
                notes.append(f"A temperature of {ents.temperature_c:.1f} C is not "
                             "usually counted as a fever (38 C or above).")
        if ents.age_group == "infant" and topic["tag"] == "fever":
            notes.append("For a baby under 3 months, any fever needs a doctor "
                         "right away.")
            urgent = True
        elif ents.age_group in ("infant", "child"):
            notes.append("Children can get worse quickly. If your child is "
                         "unusually sleepy, not drinking, or you are worried, "
                         "see a doctor.")
        return notes, urgent

    def _topic_reply(self, tags: list[str], ents: Entities, match: Match,
                     which: str = "both") -> Reply:
        sections, notes, urgent, summaries = [], [], False, []
        for tag in tags:
            t = self.topics[tag]
            label = f" ({t['name']})" if len(tags) > 1 else ""
            summaries.append(t["summary"])
            if which in ("both", "selfcare"):
                sections.append({"heading": "What usually helps" + label,
                                 "items": t["self_care"]})
            if which in ("both", "doctor"):
                sections.append({"heading": "See a doctor if" + label,
                                 "items": t["see_doctor"]})
            n, u = self._escalation_notes(t, ents)
            notes += [x for x in n if x not in notes]
            urgent = urgent or u
        title = " and ".join(self.topics[t]["name"] for t in tags).capitalize()
        if match.source == "ml":
            notes.insert(0, f"I understood this as: {self.topics[tags[0]]['name']}. "
                            "Tell me if I got it wrong.")
        summary = " ".join(summaries) if which == "both" else ""
        lines = notes + ([summary] if summary else [])
        for s in sections:
            lines.append(s["heading"] + ":")
            lines += ["- " + i for i in s["items"]]
        lines.append(DISCLAIMER)
        return Reply(text="\n".join(lines), intent="+".join(tags),
                     source=match.source, confidence=match.confidence,
                     urgent=urgent, title=title, sections=sections,
                     notes=notes + ([summary] if summary else []),
                     entities=ents.as_dict())

    def _plain(self, text, intent, source="rule", confidence=1.0, urgent=False,
               ents: Entities | None = None) -> Reply:
        return Reply(text=text, intent=intent, source=source,
                     confidence=confidence, urgent=urgent, notes=[text],
                     entities=ents.as_dict() if ents else {})

    def _fallback(self, ents) -> Reply:
        names = ", ".join(t["name"] for t in list(self.topics.values())[:8])
        return self._plain(
            "Sorry, I didn't understand that. Try describing one symptom in a "
            f"few words. I know about: {names}, and more.",
            "fallback", source="fallback", confidence=0.0, ents=ents)

    # ------------------------------------------------------------ reply --
    def reply(self, message: str, session: Session | None = None) -> Reply:
        session = session if session is not None else Session()
        session.turns += 1
        text = normalise(message or "")
        if not text:
            return self._plain("Please type your question or symptom.", "empty",
                               source="fallback", confidence=0.0)

        # 1. Safety always comes first.
        flag = safety.check(text)
        if flag == "crisis":
            return self._plain(safety.crisis_message(), "crisis", "safety", urgent=True)
        if flag == "emergency":
            return self._plain(safety.emergency_message(), "emergency", "safety",
                               urgent=True)

        ents = extract_entities(text)
        matches = self.classifier.classify(text)
        by_group = {}
        for m in matches:
            by_group.setdefault(m.group, []).append(m)
        topics = by_group.get("topics", [])
        general = {m.tag: m for m in by_group.get("general", [])}
        followups = by_group.get("followups", [])
        if topics:
            session.last_topic = topics[0].tag
        rule_hit = any(m.source == "rule" for m in matches)

        # 2. Requests the bot must decline, whatever the symptom.
        for tag in ("medication", "diagnosis"):
            if tag in general:
                return self._plain(self.general[tag]["response"], tag,
                                   general[tag].source, general[tag].confidence,
                                   ents=ents)

        # 3. A bare detail ("it has been 5 days", "it is 102 F") about the
        #    symptom already under discussion.
        if ents and session.last_topic and not topics and not rule_hit:
            topic = self.topics[session.last_topic]
            notes, urgent = self._escalation_notes(topic, ents)
            if not urgent:
                if ents.duration_days is not None and topic.get("escalate_after_days"):
                    notes.append(f"{_fmt_days(ents.duration_days).capitalize()} is "
                                 f"within the usual range for {topic['name']}. "
                                 "Keep watching for the warning signs.")
            if notes:
                m = Match(topic["tag"], "topics", "context", 1.0)
                r = self._topic_reply([topic["tag"]], Entities(), m, "doctor")
                r.notes = notes
                r.text = "\n".join(notes) + "\n" + r.text
                r.urgent = urgent
                r.entities = ents.as_dict()
                return r

        # 4. A symptom named in this message.
        if topics:
            which = "both"
            if followups:
                which = "doctor" if followups[0].tag == "followup_doctor" else "selfcare"
            tags = [m.tag for m in topics[:MAX_TOPICS]]
            return self._topic_reply(tags, ents, topics[0], which)

        # 5. Follow-up questions use the symptom from earlier in the chat.
        if followups:
            if session.last_topic:
                which = "doctor" if followups[0].tag == "followup_doctor" else "selfcare"
                m = Match(session.last_topic, "topics", "context",
                          followups[0].confidence)
                return self._topic_reply([session.last_topic], ents, m, which)
            return self._plain("Which symptom do you mean? For example: fever, "
                               "headache, cough or stomach upset.", "clarify",
                               followups[0].source, followups[0].confidence, ents=ents)

        # 6. Greetings, thanks and other general messages.
        if general:
            m = next(iter(general.values()))
            return self._plain(self.general[m.tag]["response"], m.tag, m.source,
                               m.confidence, ents=ents)

        return self._fallback(ents)
