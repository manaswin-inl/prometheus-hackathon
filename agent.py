# This file is part of Prometheus
# https://inl.gov/super-intelligence/prometheus/
#
# Copyright 2026 Battelle Energy Alliance, LLC
# All rights reserved, see NOTICE.txt for full restrictions
# https://github.com/idaholab/prometheus_core/blob/main/NOTICE.txt
#
# Licensed under the MIT License, please see LICENSE for details
# https://github.com/idaholab/prometheus_core/blob/main/LICENSE
#
# SPDX-License-Identifier: MIT

"""
team-8-agent Agent.

This is an agent which will be used to read  agent to extract relevant tabular data from files and display plots

This is a Prometheus specialist agent.  Subclass PrometheusAgent and implement
`_execute` — the base class handles the HTTP server, authentication, agent-card
serving, and registry registration for you.
"""

import logging
from typing import Any, Optional
import os
from prometheus_api import load_prompt  # noqa: F401 - used by the LLM pattern below
from prometheus_api.agent import EventQueue, MissingInput, PrometheusAgent, RequestContext
from prometheus_api.storage_client.connection import ConnectionConfig
import asyncio
import json
import tempfile
import pandas as pd
from io import StringIO
from pathlib import Path
from plotly_tool import DEFAULT_DPI, describe_dataframe, run_plot_tool
from table_tools import apply_column_renames, convert_units, find_missing_columns

# Read secrets and configuration from environment variables.
# Never hard-code API keys or URLs here.
# Example (add `import os` to the imports above):
MY_API_KEY = os.getenv("LITELLM_MASTER_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _try_storage_config() -> Optional[ConnectionConfig]:
    """Return a ConnectionConfig from env, or None if credentials are missing."""
    try:
        return ConnectionConfig.from_env()
    except ValueError:
        return None


def _parse_llm_json(text: str) -> Any:
    """Parse a model reply that should be JSON, tolerating a ```json fence."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


# Context-state key for a plot paused on INPUT_REQUIRED (unknown column names).
_PENDING_PLOT = "pending_plot"
_KEEP_AS_IS = "(skip this column)"


class Team8AgentAgent(PrometheusAgent):
    """team-8-agent Agent."""

    # Optional: register agent-owned HTTP endpoints in __init__.
    #
    # def __init__(self, *args: Any, **kwargs: Any) -> None:
    #     super().__init__(*args, **kwargs)
    #     self.add_http_endpoint(
    #         "/api/preview/{artifact_id}",
    #         self.preview_artifact,
    #         methods=["GET"],
    #         endpoint_id="artifact_preview",
    #         purpose="artifact",
    #         description="Return preview metadata for one artifact.",
    #         content_types=["application/json"],
    #     )
    #     self.add_static_endpoint(
    #         "/viewer",
    #         "viewer-assets",
    #         endpoint_id="viewer_assets",
    #         purpose="viewer_plugin",
    #         description="Serve viewer plugin JavaScript and CSS.",
    #         content_types=["text/javascript", "text/css"],
    #     )
    #
    # async def preview_artifact(self, request):
    #     from starlette.responses import JSONResponse
    #
    #     return JSONResponse({"artifact_id": request.path_params["artifact_id"]})

    async def _execute(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        """Process one request.

        Args:
            params: Parameters from the caller's JSON payload, plus ``skill_id``
                from the message metadata (the skill the orchestrator selected).
            context: The A2A request context.  Use ``context.context_id`` as a
                stable key for per-conversation state.
            event_queue: Pass to the ``send_*`` helpers to emit progress updates
                and artifacts.

        Returns:
            A plain ``dict``.  The base class wraps it as the final ``completed``
            task status automatically — do not enqueue that event yourself.
        """
        # ------------------------------------------------------------------
        # 1. Read inputs from params.  These keys must match the properties
        #    declared in agent_card.json's inputSchema.
        # ------------------------------------------------------------------
        # input_data = params.get("input_data")

        # A plot paused on INPUT_REQUIRED resumes here when the user answers.
        state = self.load_context_state(str(context.context_id))
        if state.get(_PENDING_PLOT) and self.get_provided_inputs(context):
            return await self._resume_plot(state, context, event_queue)

        handler = {
            "team_8_agent_extract": self._extract,
            "team_8_agent_plot": self._plot,
            "team_8_agent_table": self._table,
        }.get(params.get("skill_id"))
        if handler is None:
            return {
                "status": "error",
                "error": f"Unknown skill: {params.get('skill_id')}",
            }

        return await handler(params, context, event_queue)

    async def _extract(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:

        artifact_id = params.get("document_artifact_id")
        description = params.get("table_description")

        if not artifact_id and not description:
            return {"status": "error", "error": "input_data is required"}

        file_path = await self.get_artifact_file_path(
            artifact_id, context=context, artifacts=params.get("artifacts")
        )
        system_prompt = load_prompt("extract.txt").format(
            file_path=file_path.read_text(), description=description
        )
        async with self.llm_client(context) as llm:
            response = await llm.medium(system_prompt)
        # return {"status": "success", "summary": response.text}

        # ------------------------------------------------------------------
        # 2. Send optional progress updates so the UI stays responsive.
        #    The base class sends the initial "working" update automatically;
        #    call this only when you want intermediate messages.
        # ------------------------------------------------------------------
        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Processing your request…",
        )

        result = json.loads(response.text)
        df = pd.read_csv(StringIO(result["csv"]))
        # Keep only the base name so an LLM-chosen filename can't escape the temp dir.
        filename = Path(result["filename"]).name or "extracted_table.csv"
        storage_config = _try_storage_config()
        # Write into a per-request temp dir that is removed once the upload is done.
        with tempfile.TemporaryDirectory(prefix="extract_") as tmp_dir:
            file_path = Path(tmp_dir) / filename
            df.to_csv(file_path, index=False)
            artifact_ref = await self.send_deeplynx_artifact(
                context=context,
                event_queue=event_queue,
                file_path=file_path,
                name=filename,
                description="Extracted csv file",
                storage_config=storage_config,
                mime_type="text/csv",
                tags=["extracted", "csv"],
                metadata={"description": description},
            )

        # ------------------------------------------------------------------
        # 3. Do your actual work here.  Replace this with real logic.
        # ------------------------------------------------------------------
        result = {
            "status": "created",
            # "conversation_id": conversation_id,
            "message": "CSV Extracted Successfully",
            "artifact": artifact_ref.to_dict(),
            "artifact_id": artifact_ref.artifact_id,
            "artifact_name": artifact_ref.name,
            "artifact_uri": artifact_ref.uri,
            # "input_received": description,
            # "summary": response.text,
        }

        # ------------------------------------------------------------------
        # 4. Return a plain dict.  Keep keys consistent with the
        #    "output_format" you described in agent_card.json.
        # ------------------------------------------------------------------
        return result

    async def _status(self, context: RequestContext, event_queue: EventQueue, message: str) -> None:
        """Progress line for the UI's "thinking" indicator."""
        await self.send_status_update(context=context, event_queue=event_queue, message=message)

    async def _load_csv(
        self, artifact_id: str, params: dict[str, Any], context: RequestContext
    ) -> tuple[Path, pd.DataFrame]:
        csv_path = await self.get_artifact_file_path(
            artifact_id, context=context, artifacts=params.get("artifacts")
        )
        return csv_path, pd.read_csv(csv_path)

    async def _plot(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:

        artifact_id = params.get("extracted_artifact_id")
        description = params.get("plot_description")

        if not artifact_id or not description:
            return {
                "status": "error",
                "error": "extracted_artifact_id and plot_description are required",
            }

        await self._status(context, event_queue, "Loading the table…")
        csv_path, df = await self._load_csv(artifact_id, params, context)
        await self._status(
            context,
            event_queue,
            f"Loaded {csv_path.name}: {len(df)} rows, {len(df.columns)} columns "
            f"({', '.join(map(str, df.columns[:8]))}{'…' if len(df.columns) > 8 else ''}).",
        )

        system_prompt = load_prompt("plot.txt").format(
            csv_data=df.to_csv(index=False),
            description=description,
        )
        await self._status(context, event_queue, "Choosing a chart type and building the plot spec…")
        async with self.llm_client(context) as llm:
            response = await llm.medium(system_prompt)

        try:
            llm_result = _parse_llm_json(response.text)
        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON: %r", response.text)
            return {"status": "error", "error": "Model did not return valid JSON"}
        llm_summary = llm_result["summary"]
        plot_payload = llm_result["payload"]
        if llm_summary:
            await self._status(context, event_queue, f"Plan: {llm_summary}")

        # PNG is always produced (300 dpi unless the user asks otherwise); the
        # interactive HTML is only produced when the user asks for it.
        dpi = params.get("dpi") or (plot_payload.get("save") or {}).get("dpi") or DEFAULT_DPI
        formats = ["png"]
        if params.get("interactive") or llm_result.get("html_requested"):
            formats.append("html")

        job = {
            "artifact_id": artifact_id,
            "artifacts": params.get("artifacts"),
            "description": description,
            "summary": llm_summary,
            "payload": plot_payload,
            "dpi": dpi,
            "formats": formats,
        }

        await self._status(context, event_queue, "Checking that every column in the spec exists…")
        missing = find_missing_columns(plot_payload, df)
        if missing:
            return await self._ask_for_columns(job, missing, df, context, event_queue)

        return await self._render_plot(job, df, context, event_queue)

    async def _ask_for_columns(
        self,
        job: dict[str, Any],
        missing: list[dict],
        df: pd.DataFrame,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        """Pause the plot and ask the user which real column each unknown one means."""
        columns = [str(c) for c in df.columns]
        names = ", ".join(f"'{m['name']}'" for m in missing)
        await self._status(context, event_queue, f"Couldn't find {names} in the table; asking you.")

        fields = []
        for i, m in enumerate(missing):
            # Closest matches first, then the remaining columns, so the best
            # guess is the default choice in the dropdown.
            options = m["suggestions"] + [c for c in columns if c not in m["suggestions"]]
            hint = f" Closest match: {m['suggestions'][0]}." if m["suggestions"] else ""
            fields.append(
                MissingInput(
                    name=f"column_{i}",
                    description=(
                        f"The plot refers to a column '{m['name']}' that is not in the table.{hint}"
                    ),
                    user_prompt=f"Which column should be used for '{m['name']}'?",
                    type="string",
                    options=options + [_KEEP_AS_IS],
                    required=True,
                )
            )

        context_id = str(context.context_id)
        state = self.load_context_state(context_id)
        state[_PENDING_PLOT] = {**job, "missing": [m["name"] for m in missing]}
        self.save_context_state(context_id, state)

        await self.send_input_required(
            context=context,
            event_queue=event_queue,
            missing_inputs=fields,
            hint=f"I couldn't find {names} in the table. Pick the column you meant.",
        )
        return {}

    async def _resume_plot(
        self, state: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:
        """Continue a plot after the user mapped the unknown columns."""
        job = state.pop(_PENDING_PLOT)
        self.save_context_state(str(context.context_id), state)
        provided = self.get_provided_inputs(context)

        renames = {}
        for i, name in enumerate(job.pop("missing")):
            choice = provided.get(f"column_{i}")
            if choice and choice != _KEEP_AS_IS:
                renames[name] = choice
        await self._status(
            context,
            event_queue,
            "Using " + ", ".join(f"'{v}' for '{k}'" for k, v in renames.items()) + "."
            if renames
            else "Continuing without the unknown columns.",
        )

        _, df = await self._load_csv(job["artifact_id"], job, context)
        job["payload"] = apply_column_renames(job["payload"], renames)
        return await self._render_plot(job, df, context, event_queue)

    async def _render_plot(
        self,
        job: dict[str, Any],
        df: pd.DataFrame,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        storage_config = _try_storage_config()
        if storage_config is None:
            return {
                "status": "error",
                "error": "DeepLynx storage is not configured; cannot upload the plot.",
                "summary": job["summary"],
            }

        dpi, formats = job["dpi"], job["formats"]
        await self._status(
            context,
            event_queue,
            f"Rendering {' and '.join(f.upper() for f in formats)} ({dpi:g} dpi)…",
        )
        # Render into a per-request temp dir that is removed once the uploads are
        # done, so rendered files don't pile up on disk.
        with tempfile.TemporaryDirectory(prefix="plot_") as tmp_dir:
            # The renderer is synchronous (kaleido drives a browser), so run it
            # off the event loop to keep heartbeats and status updates flowing.
            plot_result = await asyncio.to_thread(
                run_plot_tool, df, job["payload"], output_dir=tmp_dir, formats=formats, dpi=dpi
            )
            if not plot_result["ok"]:
                return {
                    "status": "error",
                    "error": plot_result.get("error", "Plot generation failed"),
                    "save_errors": plot_result.get("save_errors"),
                    "summary": job["summary"],
                }
            for fmt, err in (plot_result.get("save_errors") or {}).items():
                logger.warning("Could not save %s plot: %s", fmt, err)
            for warning in plot_result.get("warnings") or []:
                await self._status(context, event_queue, f"Note: {warning}")

            mime_types = {"html": "text/html", "png": "image/png"}
            descriptions = {
                "html": "Interactive plot (Plotly HTML)",
                "png": f"Plot image ({dpi:g} dpi)",
            }
            await self._status(context, event_queue, "Uploading the plot…")
            artifact_refs = []
            for fmt, path in plot_result["files"].items():
                path = Path(path)
                artifact_ref = await self.send_deeplynx_artifact(
                    context=context,
                    event_queue=event_queue,
                    file_path=path,
                    name=path.name,
                    description=descriptions[fmt],
                    storage_config=storage_config,
                    mime_type=mime_types[fmt],
                    tags=["plot", fmt],
                    metadata={"description": job["description"]},
                )
                artifact_refs.append(artifact_ref)

        result = {
            "status": "created",
            "message": f"Plot Created Successfully ({len(artifact_refs)} file(s))",
            "artifacts": [ref.to_dict() for ref in artifact_refs],
            "artifact_ids": [ref.artifact_id for ref in artifact_refs],
            "artifact_names": [ref.name for ref in artifact_refs],
            "artifact_uris": [ref.uri for ref in artifact_refs],
            "summary": job["summary"],
        }
        if plot_result.get("warnings"):
            result["warnings"] = plot_result["warnings"]

        return result

    async def _table(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:
        """Describe a table, or convert its columns to other units (new CSV artifact)."""
        artifact_id = params.get("table_artifact_id")
        request = params.get("table_request")
        if not artifact_id or not request:
            return {
                "status": "error",
                "error": "table_artifact_id and table_request are required",
            }

        await self._status(context, event_queue, "Loading the table…")
        csv_path, df = await self._load_csv(artifact_id, params, context)
        await self._status(
            context, event_queue, f"Loaded {csv_path.name}: {len(df)} rows, {len(df.columns)} columns."
        )

        system_prompt = load_prompt("table.txt").format(
            schema=describe_dataframe(df),
            sample=df.head(10).to_csv(index=False),
            request=request,
        )
        await self._status(context, event_queue, "Working out what you're asking for…")
        async with self.llm_client(context) as llm:
            response = await llm.medium(system_prompt)
        try:
            plan = _parse_llm_json(response.text)
        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON: %r", response.text)
            return {"status": "error", "error": "Model did not return valid JSON"}

        conversions = plan.get("conversions") or []
        if not conversions:
            return {
                "status": "success",
                "message": plan.get("description") or "No description was produced.",
                "columns": [str(c) for c in df.columns],
                "rows": len(df),
            }

        await self._status(
            context,
            event_queue,
            "Converting "
            + ", ".join(f"{c['column']} {c['from_unit']}→{c['to_unit']}" for c in conversions)
            + "…",
        )
        try:
            converted, changes = convert_units(df, conversions)
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        storage_config = _try_storage_config()
        if storage_config is None:
            return {
                "status": "error",
                "error": "DeepLynx storage is not configured; cannot upload the table.",
            }

        filename = f"{csv_path.stem}_converted.csv"
        await self._status(context, event_queue, "Uploading the converted table…")
        with tempfile.TemporaryDirectory(prefix="table_") as tmp_dir:
            file_path = Path(tmp_dir) / filename
            converted.to_csv(file_path, index=False)
            artifact_ref = await self.send_deeplynx_artifact(
                context=context,
                event_queue=event_queue,
                file_path=file_path,
                name=filename,
                description="Table with converted units: " + "; ".join(changes),
                storage_config=storage_config,
                mime_type="text/csv",
                tags=["converted", "csv"],
                metadata={"request": request, "conversions": changes},
            )

        return {
            "status": "created",
            "message": "Converted " + "; ".join(changes),
            "description": plan.get("description"),
            "conversions": changes,
            "artifact": artifact_ref.to_dict(),
            "artifact_id": artifact_ref.artifact_id,
            "artifact_name": artifact_ref.name,
            "artifact_uri": artifact_ref.uri,
        }

    # ----------------------------------------------------------------------
    # PATTERNS — uncomment and adapt as needed.
    # ----------------------------------------------------------------------
    #
    # Calling an LLM through the platform's LiteLLM gateway:
    #
    #
    # Dispatching when agent_card.json declares more than one skill:
    #

    #
    # Persisting per-conversation state across turns (Redis-backed, with an
    # in-memory fallback for local development):
    #
    #     state = self.load_context_state(str(context.context_id))
    #     state["count"] = state.get("count", 0) + 1
    #     self.save_context_state(str(context.context_id), state)
    #
    # See TUTORIAL.md ("Advanced topics") for emitting artifacts, asking the
    # user for missing inputs (INPUT_REQUIRED), and DeepLynx file storage.


if __name__ == "__main__":
    agent = Team8AgentAgent(agent_name="team-8-agent Agent")
    agent.run()
