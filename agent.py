import os
import json
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv("key.env")

client = genai.Client()
MODEL = "gemini-2.5-flash"

TYPOLOGY_PATH = "typology.md"
GOLD_CONCORDIA = "annotated_concordia.tsv"
GOLD_MCGILL    = "annotated_mcgill.tsv"

CATEGORIES = [
    "Complaints", "Courses", "Student Life", "Textbooks",
    "Graduate", "Administration", "Advice", "Grades", "Unclassified"
]

#  Load data 
def load_data():
    df_c = pd.read_csv(GOLD_CONCORDIA, sep="\t")
    df_m = pd.read_csv(GOLD_MCGILL,   sep="\t")
    for df in [df_c, df_m]:
        df["Coding"] = df["Coding"].str.lower().str.strip()
    return pd.concat([df_c, df_m], ignore_index=True)


#  Tool implementations 
def lookup_category(category_name: str) -> str:
    """Return the full definition + edge cases for a category from the typology."""
    with open(TYPOLOGY_PATH, "r") as f:
        typology = f.read()
    lines = typology.split("\n")
    capturing = False
    result = []
    for line in lines:
        if category_name.lower() in line.lower() and line.strip() and line.strip()[0].isdigit():
            capturing = True
        elif capturing and line.strip() and line.strip()[0].isdigit():
            break
        if capturing:
            result.append(line)
    return "\n".join(result) if result else f"Category '{category_name}' not found."


def get_similar_examples(title: str, df: pd.DataFrame, n: int = 3) -> str:
    """Return n labeled examples that share keywords with the given title."""
    title_words = set(title.lower().split())
    scores = []
    for _, row in df.iterrows():
        row_words = set(str(row["Title"]).lower().split())
        overlap = len(title_words & row_words)
        scores.append((overlap, row["Title"], row["Coding"]))
    scores.sort(reverse=True)
    top = scores[:n]
    if not top or top[0][0] == 0:
        return "No similar examples found."
    return "\n".join(f'- "{t}" -> {c}' for _, t, c in top)


def flag_for_human_review(title: str, reason: str) -> str:
    """Flag a post as ambiguous and log it."""
    with open("flagged_for_review.txt", "a") as f:
        f.write(f"TITLE: {title}\nREASON: {reason}\n\n")
    return f"Flagged: {title}"


def dispatch_tool(tool_name: str, args: dict, df: pd.DataFrame) -> str:
    if tool_name == "lookup_category":
        return lookup_category(args["category_name"])
    elif tool_name == "get_similar_examples":
        return get_similar_examples(args["title"], df)
    elif tool_name == "flag_for_human_review":
        return flag_for_human_review(args["title"], args["reason"])
    return "Unknown tool."


#  Tool schemas 
TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="lookup_category",
            description="Look up the full definition and edge cases for a category before deciding.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "category_name": types.Schema(
                        type="STRING",
                        description="The category name to look up, e.g. 'Grades'"
                    )
                },
                required=["category_name"]
            )
        ),
        types.FunctionDeclaration(
            name="get_similar_examples",
            description="Retrieve similar previously-labeled posts to guide the annotation decision.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "title": types.Schema(
                        type="STRING",
                        description="The post title to find similar examples for"
                    )
                },
                required=["title"]
            )
        ),
        types.FunctionDeclaration(
            name="flag_for_human_review",
            description="Flag a post as ambiguous and escalate to human review instead of guessing.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "title": types.Schema(type="STRING", description="The post title"),
                    "reason": types.Schema(type="STRING", description="Why this post is ambiguous")
                },
                required=["title", "reason"]
            )
        )
    ])
]

SYSTEM_PROMPT = """You are an expert data annotator for university Reddit posts.
Classify post titles into exactly one category.

You have three tools:
- lookup_category: use when unsure about a category definition or edge cases
- get_similar_examples: use when a post is ambiguous and you want to see how similar posts were labeled
- flag_for_human_review: use ONLY when the post is genuinely unclassifiable after using the other tools

Valid categories: Complaints, Courses, Student Life, Textbooks, Graduate, Administration, Advice, Grades, Unclassified

After reasoning, respond with ONLY the category name."""


#  Pass 1: Agent with tool use 
def annotate_with_tools(title: str, df: pd.DataFrame, max_turns: int = 5) -> tuple:
    """Run the agentic loop. Returns (final_label, tool_calls_log)."""
    messages = [
        types.Content(role="user", parts=[
            types.Part(text=f'Classify this Reddit post title: "{title}"')
        ])
    ]
    tool_calls_log = []

    for _ in range(max_turns):
        response = None
        # Increased maximum retries to weather quota cooldowns
        for attempt in range(6):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TOOLS,
                        temperature=0.0
                    )
                )
                break  # Success!
            except Exception as e:
                err_str = str(e).upper()
                if "429" in err_str or "EXHAUSTED" in err_str:
                    # Hit rate limits: require a cooling period
                    wait_time = 15 + (attempt * 5)
                    print(f"    [Rate Limit (429), cooling down for {wait_time}s (Attempt {attempt+1}/6)...]")
                    time.sleep(wait_time)
                elif "503" in err_str or "UNAVAILABLE" in err_str:
                    # Concurrency spike
                    wait_time = 5 * (attempt + 1)
                    print(f"    [Server busy (503), waiting {wait_time}s to retry (Attempt {attempt+1}/6)...]")
                    time.sleep(wait_time)
                else:
                    raise e  
        
        if response is None:
            print("    [Critical: Quota/API limits exhausted. Gracefully skipping to unclassified.]")
            return "unclassified", tool_calls_log

        candidate = response.candidates[0].content
        messages.append(candidate)

        tool_parts = [p for p in candidate.parts if p.function_call]
        text_parts  = [p for p in candidate.parts if p.text]

        if not tool_parts:
            label = text_parts[0].text.strip().lower() if text_parts else "unclassified"
            return label, tool_calls_log

        # Execute tools and feed results back
        function_responses = []
        for part in tool_parts:
            fc = part.function_call
            result = dispatch_tool(fc.name, dict(fc.args), df)
            tool_calls_log.append({"tool": fc.name, "args": dict(fc.args), "result": result})
            function_responses.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": result}
                ))
            )

        messages.append(types.Content(role="user", parts=function_responses))
        # Keep an intentional delay between steps to preserve RPM pool
        time.sleep(2)

    return "unclassified", tool_calls_log


#  Pass 2: Reflection / critic 
def reflect(title: str, first_label: str) -> tuple:
    """Ask a critic to verify the label. Returns (final_label, was_revised)."""
    with open(TYPOLOGY_PATH, "r") as f:
        typology = f.read()

    prompt = f"""You are a strict annotation reviewer.

Post title: "{title}"
Proposed label: "{first_label}"

Full typology:
{typology}

Is this label correct?
- If YES: respond exactly as  CONFIRMED: {first_label}
- If NO:  respond exactly as  REVISED: <correct_category>

Only use one of those two formats."""

    response = None
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            break
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "EXHAUSTED" in err_str:
                wait_time = 15 + (attempt * 5)
                print(f"    [Reviewer Rate Limit (429), cooling down for {wait_time}s...]")
                time.sleep(wait_time)
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait_time = 5 * (attempt + 1)
                print(f"    [Reviewer backend busy (503), waiting {wait_time}s to retry...]")
                time.sleep(wait_time)
            else:
                raise e

    if response is None:
        return first_label, False

    text = response.text.strip()
    if text.upper().startswith("REVISED:"):
        revised = text.split(":", 1)[1].strip().lower()
        return revised, True
    return first_label, False


#  Evaluation 
def evaluate(df_results: pd.DataFrame) -> dict:
    gold  = df_results["human_label"]
    first = df_results["first_pass_label"]
    final = df_results["final_label"]

    reflection_helped = ((first != final) & (final == gold)).sum()
    reflection_hurt   = ((first != final) & (first == gold)).sum()

    return {
        "first_pass_accuracy":        round((first == gold).mean(), 3),
        "post_reflection_accuracy":   round((final == gold).mean(), 3),
        "reflection_revision_rate":   round((first != final).mean(), 3),
        "times_reflection_helped":    int(reflection_helped),
        "times_reflection_hurt":      int(reflection_hurt),
        "avg_tool_calls_per_post":    round(df_results["num_tool_calls"].mean(), 2),
        "flagged_for_human_review":   int((df_results["final_label"] == "flagged").sum()),
    }


#  Main 
def main(n_samples: int = 20):
    print("Loading data...")
    df = load_data()
    sample = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    records = []
    for i, row in sample.iterrows():
        title       = row["Title"]
        human_label = row["Coding"]
        print(f"\n[{i+1}/{len(sample)}] {title[:60]}...")

        # Pass 1: agent with tools
        first_label, tool_log = annotate_with_tools(title, df)
        print(f"  First pass : {first_label}  ({len(tool_log)} tool calls)")

        # Pass 2: reflection critic
        final_label, revised = reflect(title, first_label)
        if revised:
            print(f"  Reflection : REVISED -> {final_label}")
        else:
            print(f"  Reflection : confirmed {final_label}")

        records.append({
            "title":            title,
            "human_label":      human_label,
            "first_pass_label": first_label,
            "final_label":      final_label,
            "was_revised":      revised,
            "num_tool_calls":   len(tool_log),
            "tool_calls":       json.dumps(tool_log),
        })

        # Base loop buffer delay to systematically respect free tier restrictions
        time.sleep(5)

    df_results = pd.DataFrame(records)
    df_results.to_csv("agent_results.csv", index=False)

    metrics = evaluate(df_results)
    print("\n── Evaluation ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.  agent_results.csv  |  metrics.json  |  flagged_for_review.txt")


if __name__ == "__main__":
    main(n_samples=20)