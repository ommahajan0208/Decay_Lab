# Quickstart

## Prerequisites

- Python 3.10+

## Setup

From the workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

## Run Tests

```powershell
python -m unittest -v decay_lab.tests.test_decay
```

## Run the App

```powershell
python -m decay_lab.app
```

## How to Use

1. Start the app:

```powershell
python -m decay_lab.app
```

2. When prompted, type a query and press Enter.

Example queries: `pizza`, `weather`, `meeting`, `running`.

3. The app prints ranked memories using relevance-weighted ensemble strength, then waits for another query.

4. Type `exit` or `quit` to stop the app.

## Expected Output

The test run prints something like:

```text
...
Ran 2 tests in ...s

OK
```

The app run starts with a header like:

```text
=== Decay-Lab Demo ===
Stored memories are ranked by:
```

After each query, you'll see lines in this form:

```text
- <memory_id> | score=<number> | last=<timestamp> | <content>
```

The app also prints extra top-memory details using the visualization helper.

## Troubleshooting

### `ModuleNotFoundError: No module named 'decay_lab'`

Run tests and the app from the workspace root, the folder containing the `decay_lab/` package.

Use these commands from the workspace root:

```powershell
python -m unittest -v decay_lab.tests.test_decay
python -m decay_lab.app
```

### Windows venv activation

If you installed dependencies using a venv, ensure it's activated in the current terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```
