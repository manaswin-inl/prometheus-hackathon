# Agent-Owned Workflows

Put workflow definitions here when this agent should publish workflows to the
platform workflow library.

PrometheusAgent loads `*.yaml`, `*.yml`, and `*.json` files from this directory
at startup and inserts them into the served agent card. The client-agent then
discovers those workflows and exposes them in the platform workflow library.

This directory includes `example_workflow.yaml.example` so generated agents do
not publish a sample workflow by default. Rename it to `example_workflow.yaml`
after you have reviewed the workflow id, steps, inputs, and output bindings.
