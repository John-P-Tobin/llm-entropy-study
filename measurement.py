
"""
measurement.py
--------------
Gets next-token probability distributions from the OpenAI API.
"""

import math
import os
from openai import OpenAI

TOP_K = 20


def get_next_token_distribution(client, prompt, temperature, model="gpt-4o-mini"):
    """
    Return the top next-token probabilities for a prompt.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1,
        logprobs=True,
        top_logprobs=TOP_K,
    )

    # Pull the top token log-probabilities from the response.
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

    tokens = [item.token for item in top_logprobs]
    probs = [math.exp(item.logprob) for item in top_logprobs]

    return tokens, probs


if __name__ == "__main__":
    # Load the API key from an environment variable.
    API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=API_KEY)

    test_prompt = "Tell me a one-word noun, any noun."
    test_T = 1.0

    tokens, probs = get_next_token_distribution(client, test_prompt, test_T)

    print(f"Prompt: {test_prompt!r}")
    print(f"Temperature: {test_T}")
    print(f"Top {len(tokens)} next-token probabilities:")
    for tok, p in zip(tokens, probs):
        print(f"  {tok!r:20s}  p = {p:.6f}")

    print(f"\nSum of top-{TOP_K} probabilities: {sum(probs):.6f}")
    print(f"(The remaining mass {1 - sum(probs):.6f} lives in the tail.)")

