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
from prometheus_api.agent import EventQueue, PrometheusAgent, RequestContext
from prometheus_api.storage_client.connection import ConnectionConfig
import json
import pandas as pd
from io import StringIO
from pathlib import Path
from plotly_tool import run_plot_tool

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

        handler = {
            "team_8_agent_extract": self._extract,
            "team_8_agent_plot": self._plot,
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
        # file_path = os.path.join(os.getcwd(), result["filename"])
        file_path = Path.cwd() / result["filename"]
        df.to_csv(file_path, index=False)
        storage_config = _try_storage_config()
        artifact_ref = await self.send_deeplynx_artifact(
            context=context,
            event_queue=event_queue,
            file_path=file_path,
            name=result["filename"],
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

    async def _plot(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:

        artifact_id = params.get("extracted_artifact_id")
        description = params.get("plot_description")

        if not artifact_id and not description:
            return {"status": "error", "error": "input_data is required"}

        csv_path = await self.get_artifact_file_path(
            artifact_id, context=context, artifacts=params.get("artifacts")
        )

        # 2. Put its contents into the message
        df = pd.read_csv(csv_path)
        csv_text = df.to_csv()

        system_prompt = load_prompt("plot.txt").format(
            csv_data=csv_text,
            description=description,
        )
        async with self.llm_client(context) as llm:
            response = await llm.medium(system_prompt)

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Processing your request…",
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            llm_result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON: %r", response.text)
            return {"status": "error", "error": "Model did not return valid JSON"}
        llm_summary = llm_result["summary"]
        plot_payload = llm_result["payload"]

        # Call the plotting tool.
        plot_result = run_plot_tool(df, plot_payload)
        if not plot_result["ok"]:
            return {
                "status": "error",
                "error": plot_result.get("error", "Plot generation failed"),
                "summary": llm_summary,
            }

        mime_types = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "html": "text/html",
        }
        storage_config = _try_storage_config()
        artifact_refs = []
        for fmt, img_path in plot_result["files"].items():
            img_path = Path(img_path)
            print("************************************")
            print("************************************")
            print(img_path, flush=True)
            print("************************************")
            print("************************************")
            artifact_ref = await self.send_deeplynx_artifact(
                context=context,
                event_queue=event_queue,
                file_path=img_path,
                name=img_path.name,
                description="Generated plot image",
                storage_config=storage_config,
                mime_type=mime_types.get(fmt, "application/octet-stream"),
                tags=["plot", fmt],
                metadata={"description": description},
            )
            artifact_refs.append(artifact_ref)

        result = {
            "status": "created",
            "message": f"Plot Created Successfully ({len(artifact_refs)} file(s))",
            "artifacts": [ref.to_dict() for ref in artifact_refs],
            "artifact_ids": [ref.artifact_id for ref in artifact_refs],
            "artifact_names": [ref.name for ref in artifact_refs],
            "artifact_uris": [ref.uri for ref in artifact_refs],
            "summary": llm_summary,
        }

        return result

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
