# Prompt Templates

Put large static LLM prompts here instead of embedding them in `agent.py`.

Example usage:

```python
from prometheus_api import load_prompt

SYSTEM_PROMPT = load_prompt("system_prompt.txt")
prompt = SYSTEM_PROMPT.format(input_data=params["input_data"])
```

`load_prompt()` reads UTF-8 text from this directory by default when your agent
runs from `agent.py`. Prompt files are ordinary source files. Keep placeholders
explicit, escape literal JSON braces as `{{` and `}}` when using `str.format`,
and add tests for important prompt formatting paths.
