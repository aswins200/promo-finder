# Promo Finder (Problem 1)
Reads microdrama screenplays and proposes 20-60s promo clips (hook + cliffhanger), ranked.

## Run (no key, free, ~2 seconds)
    python3 promofinder.py --input "<folder with episode-*.txt>" --out raw_outputs --judge heuristic

## Run with an LLM (model is 3 env vars; Anthropic API or any OpenAI-compatible endpoint)
    export LLM_PROVIDER=anthropic LLM_MODEL=<model-name> LLM_API_KEY=<key>
    python3 promofinder.py --input "<folder>" --out raw_outputs_llm --judge llm
    # Gemini/Groq/OpenRouter/local: LLM_PROVIDER=openai LLM_BASE_URL=<their OpenAI-compatible URL>

Outputs: ranked_promos.md (human), promos.csv (editor), promos.json (machine).
No dependencies beyond Python 3.8+. The screenplay files are NOT in this repo (Kuku confidential).

## Honest status
- raw_outputs/ was produced by the heuristic judge. The LLM judge path is written and its parsing/validation
  was tested with a mocked reply, but NOT run against a live model in this submission.
- raw_outputs_v1_before_fix/ is the first run, kept to show the iteration.
