import os
import pandas as pd

from workflows import Workflow, Context, step
from workflows.events import StartEvent, Event, StopEvent
from workflows.resource import Resource
from typing import Annotated, cast
from pydantic import BaseModel, ConfigDict
from llama_cloud_services.parse.types import JobResult as ParsingJobResult
from llama_cloud_services import LlamaParse
from jinja2 import Template
from google.genai import Client as GenAIClient
from .resources import get_llama_parse, get_llm, get_prompt_template


class WorkflowState(BaseModel):
    parsing_job_result: ParsingJobResult | None = None
    extracted_text: str = ""
    extracted_tables: list[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileEvent(StartEvent):
    input_file: str


class ParsingDoneEvent(Event):
    """
    Event to signal that parsing is done.
    Parsing results are available through WorkflowState.
    """

    pass


class TextExtractionDoneEvent(Event):
    pass


class TableExtractionDoneEvent(Event):
    pass


class OutputEvent(StopEvent):
    final_result: str | None = None
    error: str | None = None


class BrokerageStatementWorkflow(Workflow):
    @step
    async def parse_file(
        self,
        ev: FileEvent,
        ctx: Context[WorkflowState],
        parser: Annotated[LlamaParse, Resource(get_llama_parse)],
    ) -> ParsingDoneEvent | OutputEvent:
        try:
            print("Starting to parse file...")
            result = cast(
                ParsingJobResult, (await parser.aparse(file_path=ev.input_file))
            )
            if result.error is not None:
                return OutputEvent(
                    error=f"Error {result.error_code} occurred while parsing: {result.error}"
                )
            async with ctx.store.edit_state() as state:
                state.parsing_job_result = result
            print("Parsing done!")
            return ParsingDoneEvent()
        except Exception as e:
            return OutputEvent(error=f"An error occurred while parsing: {e}")

    @step
    async def extract_text(
        self,
        ev: ParsingDoneEvent,
        ctx: Context[WorkflowState],
    ) -> TextExtractionDoneEvent:
        print("Extracting text...")
        parsing_result = cast(
            ParsingJobResult, (await ctx.store.get_state()).parsing_job_result
        )
        # get the entire markdown text
        text = await parsing_result.aget_markdown()
        async with ctx.store.edit_state() as state:
            state.extracted_text = text
        print("Text extraction finished!")
        return TextExtractionDoneEvent()

    @step
    async def extract_tables(
        self,
        ev: ParsingDoneEvent,
        ctx: Context[WorkflowState],
        parser: Annotated[LlamaParse, Resource(get_llama_parse)],
    ) -> TableExtractionDoneEvent:
        print("Extracting tables...")
        parsing_result = cast(
            ParsingJobResult, (await ctx.store.get_state()).parsing_job_result
        )
        json_job_result = await parsing_result.aget_json()
        # get the tables and download them as CSV files
        os.makedirs("tables/", exist_ok=True)
        await parser.aget_tables(json_result=[json_job_result], download_path="tables/")
        csv_files = [os.path.join("tables", f) for f in os.listdir("tables/")]
        tables = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                markdown_table = df.to_markdown()
                tables.append(markdown_table)
            except Exception as e:
                print(f"Impossible to parse table from file {csv_file} because of: {e}")
        async with ctx.store.edit_state() as st:
            st.extracted_tables = tables

        print("Table extraction finished!")
        return TableExtractionDoneEvent()

    @step
    async def ask_llm(
        self,
        ev: TableExtractionDoneEvent | TextExtractionDoneEvent,
        ctx: Context[WorkflowState],
        llm: Annotated[GenAIClient, Resource(get_llm)],
        template: Annotated[Template, Resource(get_prompt_template)],
    ) -> OutputEvent:
        if (
            ctx.collect_events(
                ev,
                [TableExtractionDoneEvent, TextExtractionDoneEvent],
            )
            is None
        ):
            return None  # type: ignore

        # when data extraction is complete:
        state = await ctx.store.get_state()

        prompt = template.render(
            extracted_text=state.extracted_text,
            extracted_tables="\n\n".join(state.extracted_tables),
        )

        try:
            response = await llm.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            if response.text is None:
                return OutputEvent(error="Could not generate the final response")
            return OutputEvent(final_result=response.text)
        except Exception as e:
            return OutputEvent(
                error=f"An error occured while generating the final response: {e}"
            )
