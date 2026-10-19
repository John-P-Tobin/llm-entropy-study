# How Random is Human-AI Interaction?

This project measures how much the way you word a prompt changes how "random" a language model's answer is.

Instead of changing the model's temperature (which is what most research does), I kept temperature fixed and changed two things about the prompt:

- The questions are split into 3 categories: narrow (one right answer), medium, or open-ended
- The format for the answer I tell the LLM to give: tight ("answer in one word only"), medium, or loose ("respond freely")

For each response I pulled the top 20 token probabilities from the API and calculated the Shannon entropy of the next-token distribution. Higher entropy means the model is less sure what to say next.

The full write-up is in [paper.pdf](paper.pdf).

## Main results

- More open questions always gave higher entropy, in every format.
- The surprising part: telling the model to answer in one word actually made entropy higher for open questions, not lower. My read is that the format limit doesn't remove the model's uncertainty, it just forces all of it into one word. For factual questions it was the opposite, and the one-word format pushed entropy close to zero.
- I ran the whole 10-turn conversation 20 times per format and used Welch t-tests and a Chebyshev bound to make sure the differences weren't noise.

## Files

- `experiment.py` runs the 10-turn conversation script under each format and saves the results
- `measurement.py` gets the token probabilities from the API for each response
- `entropy.py` calculates the entropy from those probabilities
- `analysis.py` runs the statistics and makes the figures
- `entropy_trajectories.csv` has the raw results from all the runs
- `figures/` has the plots used in the paper

## How to run

You'll need Python 3 and an API key for the model, set as an environment variable.

pip install numpy pandas scipy matplotlib

Then run `experiment.py` to collect the data and `analysis.py` to make the figures. Collecting the data takes a while since it makes 600+ API calls.
