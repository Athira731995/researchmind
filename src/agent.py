"""Step 7 - Agent: the model decides which tool to use, and our code runs it."""
import ast, operator
from src import llm, guardrails

TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the knowledge base for facts about AI topics.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "what to look for"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Calculate an arithmetic expression such as 5000*0.99",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
]

AGENT_SYSTEM = """You are ResearchMind, a helpful assistant with two tools.

Rules:
1. Greetings and small talk: answer directly. Do NOT use any tool.
2. For ANY question about an AI topic, call search_docs FIRST. Never answer from memory.
3. For ANY arithmetic, call calculator. Never calculate in your head.
4. If a calculation needs a number from the documents, call search_docs FIRST to get that number, THEN call calculator. Never guess a number.
5. Base your final answer ONLY on tool results. Do not add outside facts. If the tool results do not contain the answer, say: I don't have enough information.
6. Keep the final answer to 3 sentences or fewer."""

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(node):
    """Evaluate simple math. We never use eval(), because model text could contain malicious code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError("unsupported expression")


def run_tool(name, args, store):
    """Our code runs the tool. Errors go back to the model as text so it can try again."""
    try:
        if name == "search_docs":
            chunks = guardrails.clean_context(store.hybrid(args["query"], k=3))
            return "\n".join(f"- {c['text']}" for c in chunks) or "Nothing found."
        if name == "calculator":
            return str(safe_eval(ast.parse(args["expression"], mode="eval").body))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def run_agent(question, store, max_steps=5, verbose=True):
    ok, q = guardrails.check_input(question)
    if q.lower().strip(" !.?").startswith(("hello", "hi", "hey", "who are you")):
        return ("I'm ResearchMind. I answer questions about AI topics from my documents "
                "and can do calculations.")
    if not ok:
        return q

    # Code does the retrieval step itself, so the model can't skip it or guess
    found = run_tool("search_docs", {"query": q}, store)
    if verbose:
        print(f"  [auto-search] -> {found[:70]}")
    messages = [{"role": "user", "content":
                 f"<context>\n{found}\n</context>\nQuestion: {q}"}]

    for step in range(1, max_steps + 1):
        resp = llm.chat(messages, system=AGENT_SYSTEM, tools=TOOLS, temperature=0)
        msg = resp.message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            result = run_tool(name, args, store)
            if verbose:
                print(f"  [step {step}] {name}({args}) -> {result[:70]}")
            messages.append({"role": "tool", "content": result, "tool_name": name})
    return "Stopped: step limit reached."