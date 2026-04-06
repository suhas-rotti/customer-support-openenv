---
title: "Smart Customer Support Environment"
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
---

# Smart Customer Support Environment

## Environment Description

Smart Customer Support Environment is a lightweight OpenEnv-style reinforcement
learning project for text-based customer support. Each episode starts with a
customer query, and the agent must respond with a helpful support message. The
environment then scores the response using a normalized reward from `0.0` to
`1.0`.

The project is designed for hackathon submissions, baseline evaluation,
inference testing, and simple deployment on Hugging Face using FastAPI.

## Observation Space

- Type: `text`
- Description: a customer support query written in natural language

Examples:

- refund requests
- delayed delivery complaints
- password reset issues
- escalation and angry customer cases

## Action Space

- Type: `text`
- Description: a natural-language support response produced by the agent

The action is a plain text reply that should address the customer's issue,
include relevant help steps, and use appropriate tone.

## Task Description

The environment includes three task levels:

- `easy`: simple single-intent requests such as refunds, delays, or password reset
- `medium`: multi-intent or moderately complex customer support issues
- `hard`: angry customers, escalation requests, repeated failures, or complex complaints

## Reward System

The reward function uses keyword-group coverage and response quality checks.

- required concepts receive the largest reward weight
- optional helpful details add bonus reward
- empathy and polite tone contribute additional reward
- short or weak replies are penalized
- final rewards are normalized between `0.0` and `1.0`

This logic is implemented in [env.py](C:\Users\suhas\Downloads\python\env.py)
and is intentionally simple, explainable, and beginner-friendly.

## Project Files

- [env.py](C:\Users\suhas\Downloads\python\env.py): core environment logic
- [run.py](C:\Users\suhas\Downloads\python\run.py): demo and baseline runner
- [inference.py](C:\Users\suhas\Downloads\python\inference.py): offline inference script with strict OpenEnv logging
- [app.py](C:\Users\suhas\Downloads\python\app.py): FastAPI server for deployment and validation
- [openenv.yaml](C:\Users\suhas\Downloads\python\openenv.yaml): environment metadata
- [Dockerfile](C:\Users\suhas\Downloads\python\Dockerfile): container setup for FastAPI deployment
- [requirements.txt](C:\Users\suhas\Downloads\python\requirements.txt): Python dependencies

## Setup Instructions

### Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run offline inference:

```bash
python inference.py
```

Run baseline or demo evaluation:

```bash
python run.py
```

### Docker Run

Build the image:

```bash
docker build -t smart-customer-support-env .
```

Run the container:

```bash
docker run --rm -p 7860:7860 smart-customer-support-env
```

## API Endpoints

- `GET /` : health check and metadata
- `GET /reset?level=easy` : start a new episode and return the customer query
- `POST /step` : score an action with JSON body `{"action": "..."}` 
- `GET /state` : return the current environment state

## Baseline Scores

The current baseline scores from [run.py](C:\Users\suhas\Downloads\python\run.py)
are:

| Policy | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Random | 0.008 | 0.031 | 0.031 |
| Rule-based | 0.860 | 0.933 | 0.835 |

## Notes

- `inference.py` reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` for
  hackathon compatibility, but it does not make real API calls.
- the current inference flow stays fully offline and safe
- the environment logic in [env.py](C:\Users\suhas\Downloads\python\env.py)
  remains unchanged
