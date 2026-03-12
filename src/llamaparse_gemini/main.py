import sys
import asyncio

from .workflow import BrokerageStatementWorkflow, FileEvent, OutputEvent


async def run_workflow(input_file: str) -> OutputEvent:
    wf = BrokerageStatementWorkflow(timeout=600)
    result = await wf.run(start_event=FileEvent(input_file=input_file))
    return result


def main() -> None:
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        result = asyncio.run(run_workflow(input_file=input_file))
        if result.error is not None:
            print("An error occurred: ", result.error)
        else:
            print("Final response:\n", result.final_result)
    else:
        raise ValueError("You should provide exactly one file from command line")
