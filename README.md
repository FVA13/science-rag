# science-rag

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The RAG-system specialized in answering scientific question.
A user asks a question on scientific topic and receives an answer based on relevant articles with links attached.

Run chatbot with:
`pixi run python science_rag/llm_agent.py`

Stack:
- *pixi* for environment and dependencies management

## Project Organization

```
├── LICENSE            <- MIT license
├── data
│   ├── processed      <- The processed data.
│   └── raw            <- The original data.
│
├── models             <- Cached models (if loaded from Hugging Face)
│
├── notebooks          <- Jupyter notebooks with experiments
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         science_rag and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── science_rag   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes science_rag a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── llm_agent.py            <- Code with agent
    │
    └── main.py                 <- Code to run web-interface
```

--------

