"""Tissue simulation runner — Docker container entrypoint.

Reads configuration from TISSUE_CONFIG environment variable,
initializes the engine, starts the WebSocket streamer, and runs
the simulation to completion.

Includes auto-terminate on wall-clock limit.

Usage::

    TISSUE_CONFIG='{"n_cells": 10000, ...}' python -m cognisom.core.tissue_runner

"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main():
    log.info("Tissue simulation runner starting")

    # Parse config from environment
    config_json = os.environ.get("TISSUE_CONFIG", "{}")
    max_runtime = float(os.environ.get("MAX_RUNTIME_HOURS", "2.0"))

    from cognisom.core.tissue_config import TissueScaleConfig
    config = TissueScaleConfig.from_json(config_json)
    config.max_runtime_hours = max_runtime

    log.info(
        "Config: %d cells, %s grid, %d GPUs, %.1f hr duration, %.1f hr max runtime",
        config.n_cells, config.grid_shape, config.n_gpus,
        config.duration, config.max_runtime_hours,
    )

    # Start WebSocket result streamer
    from cognisom.core.result_streamer import ResultStreamer
    streamer = ResultStreamer(host="0.0.0.0", port=8600)
    streamer.start()

    # Initialize engine
    from cognisom.core.tissue_engine import TissueSimulationEngine
    engine = TissueSimulationEngine(config)
    engine.initialize()

    # Signal handler for graceful shutdown
    def handle_signal(sig, frame):
        log.warning("Received signal %d, shutting down", sig)
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Run simulation with streaming callback
    def on_snapshot(snapshot):
        streamer.push_snapshot(snapshot)
        log.info(
            "Snapshot pushed: step=%d, cells=%d, clients=%d",
            snapshot.get("step", 0),
            snapshot.get("metrics", {}).get("total_cells", 0),
            streamer.n_clients,
        )

    try:
        result = engine.run(callback=on_snapshot)
        log.info("Simulation complete: %s", json.dumps({
            k: v for k, v in result.items()
            if not isinstance(v, (list, dict))
        }))

        # Push final summary
        streamer.push_snapshot(result)
        time.sleep(5)  # Give clients time to receive final snapshot

    except Exception as e:
        log.error("Simulation failed: %s", e, exc_info=True)
    finally:
        streamer.stop()
        log.info("Runner exiting")


if __name__ == "__main__":
    main()
