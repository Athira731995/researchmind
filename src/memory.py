"""Step 8 - Memory: remember recent turns, summarize old ones, and fix follow-up questions."""
from src import llm
import re

REFERENCE_WORDS = {"it", "its", "that", "this", "they", "them", "their",
                   "those", "these", "he", "she", "one"}


def needs_rewrite(question):
    """Only questions with pointing words, or very short ones, can depend on earlier turns."""
    words = re.findall(r"\w+", question.lower())
    return len(words) <= 3 or bool(set(words) & REFERENCE_WORDS)
def keeps_words(original, rewritten):
    """True if the rewrite still contains (most of) the original question's own words."""
    orig = [w for w in re.findall(r"\w+", original.lower()) if w not in REFERENCE_WORDS]
    new = set(re.findall(r"\w+", rewritten.lower()))
    return not orig or sum(w in new for w in orig) >= 0.8 * len(orig)

class Memory:
    def __init__(self, window=3):
        self.window = window      # how many recent exchanges to keep word for word
        self.turns = []           # list of {"role": ..., "content": ...}
        self.summary = ""         # short summary of older exchanges

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.window * 2:        # too long: summarize the oldest exchange
            oldest, self.turns = self.turns[:2], self.turns[2:]
            self._summarize(oldest)

    def _summarize(self, old):
        text = "\n".join(f"{t['role']}: {t['content']}" for t in old)
        prompt = (f"Current summary: {self.summary or 'none'}\n"
                  f"New messages:\n{text}\n"
                                    "Write an updated summary in at most 3 short sentences. "
                  "Use ONLY facts stated in the messages above. Add nothing new.")
        self.summary = llm.text_of(
            llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=150))

    def history_text(self):
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation: {self.summary}")
        parts += [f"{t['role']}: {t['content']}" for t in self.turns]
        return "\n".join(parts)

    def standalone(self, question):
        """Rewrite a follow-up question so it makes sense on its own."""
        if not self.turns or not needs_rewrite(question):
            return question
        prompt = ("Rewrite the NEW QUESTION as a complete standalone question. "
                  "Replace words like 'it' and 'that' with the topic of the conversation. "
                  "Keep all the other words of the new question exactly as they are. "
                  "Output only the question.\n\n"
                  "Example:\n"
                  "Conversation: user: What is a cat? assistant: A small pet.\n"
                  "New question: How big is it?\n"
                  "Standalone question: How big is a cat?\n\n"
                  f"Conversation:\n{self.last_exchange()}\n\n"
                  f"New question: {question}\nStandalone question:")
        out = llm.text_of(llm.chat([{"role": "user", "content": prompt}],
                                   temperature=0, max_tokens=60)).strip()
        return out if out and keeps_words(question, out) else question
    def last_exchange(self):
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns[-2:])