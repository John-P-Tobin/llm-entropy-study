"""
experiment.py
-------------
Main experiment for measuring response entropy.
"""

import math
import csv
import os
from openai import OpenAI

from measurement import TOP_K
from entropy import shannon_entropy

MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
N_RUNS = 20
N_AVG = 10

FORMAT_CONSTRAINT_LEVELS = {
    "tight":  "Answer every question in exactly one word. No punctuation, no explanation.",
    "medium": "Answer every question in a single short sentence.",
    "loose":  "Respond to every message however you like, at any length.",
}

HUMAN_SCRIPT = [
    ("narrow", "What is the capital of France?"),
    ("open",   "Tell me an interesting idea, anything at all."),
    ("medium", "Name a common breakfast food."),
    ("narrow", "What is two plus two?"),
    ("open",   "Describe an imaginary creature in one phrase."),
    ("medium", "Name a primary color."),
    ("narrow", "What is the chemical symbol for water?"),
    ("medium", "Name a day of the week."),
    ("open",   "Suggest a topic worth exploring."),
    ("narrow", "How many continents are there?"),
]


def measure_response_entropy(client, messages):
    """
    Measure average entropy over the first N_AVG response tokens.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=N_AVG,
        logprobs=True,
        top_logprobs=TOP_K,
    )

    # Get token-level logprobs from the response.
    content = response.choices[0].logprobs.content

    per_token_H = []
    for tok_entry in content:
        probs = [math.exp(item.logprob) for item in tok_entry.top_logprobs]
        per_token_H.append(shannon_entropy(probs, normalize=True))

    avg_H = sum(per_token_H) / len(per_token_H) if per_token_H else 0.0
    generated_text = response.choices[0].message.content
    return avg_H, per_token_H, generated_text


def run_conversation(client, system_prompt):
    """
    Run one scripted conversation and record entropy at each turn.
    """
    messages = [{"role": "system", "content": system_prompt}]
    per_turn_H = []
    per_turn_content_labels = []

    # Baseline before conversation history builds up.
    baseline_messages = messages + [{"role": "user", "content": HUMAN_SCRIPT[0][1]}]
    turn0_H, _, _ = measure_response_entropy(client, baseline_messages)

    for content_label, human_turn in HUMAN_SCRIPT:
        messages.append({"role": "user", "content": human_turn})

        # Measure entropy for this turn.
        H_t, _, _ = measure_response_entropy(client, messages)
        per_turn_H.append(H_t)
        per_turn_content_labels.append(content_label)

        # Generate the response that gets added to the conversation.
        full_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        assistant_text = full_response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

    return per_turn_H, per_turn_content_labels, turn0_H


def run_experiment(client):
    """
    Run all format-constraint conditions.
    """
    results = {}
    for level, system_prompt in FORMAT_CONSTRAINT_LEVELS.items():
        print(f"\n=== Format constraint: {level!r} ===")
        runs = []
        for r in range(N_RUNS):
            print(f"  run {r + 1}/{N_RUNS} ...")
            traj = run_conversation(client, system_prompt)
            runs.append(traj)
        results[level] = runs
    return results


def save_results_csv(results, path="entropy_trajectories.csv"):
    """
    Save the experiment results as a CSV file.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["format_constraint", "run", "turn", "content_constraint", "entropy_bits"])
        for level, runs in results.items():
            for r, (per_turn_H, labels, turn0_H) in enumerate(runs):
                writer.writerow([level, r, -1, "baseline", f"{turn0_H:.6f}"])
                for t, (H, lab) in enumerate(zip(per_turn_H, labels)):
                    writer.writerow([level, r, t, lab, f"{H:.6f}"])
    print(f"\nSaved raw trajectories to {path}")


if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=API_KEY)

    results = run_experiment(client)
    save_results_csv(results)

    print("\n--- Mean H per turn (bits) across runs ---")
    for level, runs in results.items():
        n_turns = len(runs[0][0])
        means = []
        for t in range(n_turns):
            col = [runs[r][0][t] for r in range(len(runs))]
            means.append(sum(col) / len(col))
        labels = runs[0][1]
        pretty = ", ".join(f"{m:.2f}({lab[0]})" for m, lab in zip(means, labels))
        print(f"  {level:7s}: {pretty}")

    print("\n--- Mean H by content constraint (averaged over turns AND runs) ---")
    for level, runs in results.items():
        bucket = {"narrow": [], "medium": [], "open": []}
        for per_turn_H, labels, _ in runs:
            for H, lab in zip(per_turn_H, labels):
                bucket[lab].append(H)
        line = []
        for lab in ["narrow", "medium", "open"]:
            vals = bucket[lab]
            mean = sum(vals) / len(vals) if vals else 0.0
            line.append(f"{lab}={mean:.2f}")
        print(f"  {level:7s}: " + "  ".join(line))

    print("\n--- Baseline (turn-zero) mean H per format level ---")
    for level, runs in results.items():
        vals = [t0 for _, _, t0 in runs]
        print(f"  {level:7s}: {sum(vals)/len(vals):.2f}")

