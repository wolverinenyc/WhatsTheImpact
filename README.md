# What's the Impact

Notebooks that go with the tutorials and interview challenges at
[whatstheimpact.com](https://whatstheimpact.com). Everything here is free to
run and nothing is gated.

## Challenges

Practice problems in the formats companies are actually using.

### Netflix completion rates

An AI-assisted coding challenge built around a root cause question. A product
team ran an initiative to get more people finishing what they start watching
and wants to know whether it worked. Working that out honestly takes more than
one query.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wolverinenyc/WhatsTheImpact/blob/main/challenges/netflix-completion-challenge.ipynb)

The first cell generates the dataset from a fixed seed, so there is nothing to
download and your numbers will match mine exactly. A walkthrough of how I would
approach it in the room is
[on the site](https://whatstheimpact.com/ai-assisted-coding-interview/).

## Tutorials

### Booking-level CUPED

A toy Airbnb experiment randomized by market. Standard CUPED adjusts each market
by its spring cancel rate, but the test runs in July, when families show up in
some markets far more than others. Predicting each booking's cancel probability
from what was booked, then averaging those predictions up to the market, removes
more of the variance. Based on Camelo, Kearns, Cersosimo & Drerup (2026),
"Variance Reduction Below the Randomization Grain," on the Instacart tech blog.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wolverinenyc/WhatsTheImpact/blob/main/tutorials/airbnb_cancellations_cuped.ipynb)

The data is simulated. The notebook downloads it from `tutorials/data/`, and the
script that generates it is in the same folder.

## About

I'm Jonathan Hershaff, a senior data scientist and PhD economist working in
experimentation and observational causal inference. I write about data science
interviews and publish simulation write-ups at
[whatstheimpact.com](https://whatstheimpact.com).
