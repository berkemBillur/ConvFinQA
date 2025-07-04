"""
Main typer app for ConvFinQA
"""

import typer
from rich import print as rich_print

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Simple in-memory chat context so we can keep state across turns when the
# CLI loop is running.  Typer itself does not provide `.state` like FastAPI.
# ---------------------------------------------------------------------------

_CHAT_STATE = {
    "dataset": None,
    "predictor": None,
    "record": None,
    "turn_index": 0,
    "history": [],
}


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Ask questions about a specific record"""
    history = []
    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        # -----------------------------------------------------------------
        # Lazy-initialise dataset, predictor and record only the first time
        if _CHAT_STATE["dataset"] is None:
            from src.data.dataset import ConvFinQADataset  # local import to keep startup fast
            from src.predictors.multi_agent.predictor import ConvFinQAMultiAgentPredictorV2

            dataset = ConvFinQADataset()
            dataset.load()
            predictor = ConvFinQAMultiAgentPredictorV2()

            record = dataset.get_record_by_id(record_id)
            if record is None:
                rich_print(f"[red]❌ Record '{record_id}' not found in dataset[/red]")
                raise typer.Exit(1)

            # Cache objects across turns
            _CHAT_STATE["dataset"] = dataset
            _CHAT_STATE["predictor"] = predictor
            _CHAT_STATE["record"] = record
            _CHAT_STATE["turn_index"] = 0
            _CHAT_STATE["history"] = []  # list of {question, answer}

        # -----------------------------------------------------------------
        record = _CHAT_STATE["record"]
        predictor = _CHAT_STATE["predictor"]
        turn_index = _CHAT_STATE["turn_index"]
        history = _CHAT_STATE["history"]

        # Replace the question for this turn in the record copy so predictor uses user prompt
        # We create a shallow copy of the original record with updated conv_question
        try:
            original_question_list = record.dialogue.conv_questions
            if turn_index < len(original_question_list):
                # Temporarily overwrite existing question (non-destructive)
                original_question_list[turn_index] = message
            else:
                original_question_list.append(message)
        except Exception:
            pass  # Fallback – should not happen

        try:
            response = predictor.predict_turn(record, turn_index, history)
        except Exception as exc:
            rich_print(f"[red]⚠️ Predictor failed: {exc}[/red]")
            response = "<error>"

        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")

        # Update conversation state
        history.append({"question": message, "answer": str(response)})
        _CHAT_STATE["turn_index"] = turn_index + 1


@app.command()
def myfunc() -> None:
    """My hello world function"""
    from src.data.dataset import ConvFinQADataset

    dataset = ConvFinQADataset()
    dataset.load()
    summary = dataset.summary()
    rich_print("[green]Dataset loaded. Split counts:[/green]")
    for split, count in summary.items():
        rich_print(f"  • {split}: {count} conversations")


if __name__ == "__main__":
    app()
