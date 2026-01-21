import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, NoiseTypes

# --------- CONFIG ---------- # replace with actual board things
BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
WINDOW_SECONDS = 5          # for client display (client can choose too)
CHUNK_SAMPLES = 20          # samples per websocket message (smaller = lower latency)
SEND_INTERVAL_MS = 50       # how often to send (pacing)
USE_FILTERS = False         # keep False for the absolute minimal test
# ---------------------------

app = FastAPI()

SITE_DIR = Path(__file__).parent # in same folder as index.html and style.css
app.mount("/static", StaticFiles(directory=str(SITE_DIR)), name="static")


def init_board():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board = BoardShim(BOARD_ID, params)

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    channels = BoardShim.get_exg_channels(BOARD_ID)
    if not channels:
        channels = BoardShim.get_eeg_channels(BOARD_ID)

    board.prepare_session()
    board.start_stream(45000)

    return board, fs, channels


board, fs, channels = init_board()
print("BOARD INIT OK")


@app.get("/")
def root():
    # Serve your existing index.html
    html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    # Fix paths: your HTML uses ./style.css, but we serve static under /static
    html = html.replace('href="./style.css"', 'href="/static/style.css"')
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    n_channels = len(channels)
    await websocket.send_text(json.dumps({"type": "meta", "fs": fs, "channels": n_channels}))

    # Continuous stream: always send the latest CHUNK_SAMPLES per channel
    try:
        while True:
            data = board.get_current_board_data(CHUNK_SAMPLES)  # shape: (rows, CHUNK_SAMPLES)

            # Extract channels -> (CHUNK_SAMPLES, n_channels)
            Y = np.stack([data[ch, :] for ch in channels], axis=1).astype(np.float64, copy=True)

            if USE_FILTERS:
                # minimal, safe preprocessing (optional)
                for ci in range(n_channels):
                    y = Y[:, ci]
                    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
                    DataFilter.remove_environmental_noise(y, fs, NoiseTypes.SIXTY.value)

            # Send row-major samples
            await websocket.send_text(json.dumps({"type": "data", "y": Y.tolist()}))
            await asyncio.sleep(SEND_INTERVAL_MS / 1000.0)

    except Exception:
        # client disconnected or server stop
        pass

# # Troubleshooting
# print("Loaded routes:")
# for r in app.router.routes:
#     print(type(r).__name__, getattr(r, "path", None))


@app.websocket("/ws_test")
async def ws_test(websocket: WebSocket):
    print("WS_TEST: connect attempt")
    await websocket.accept()
    print("WS_TEST: accepted")
    await websocket.send_text("ok")

@app.on_event("shutdown")
def shutdown_event():
    try:
        board.stop_stream()
    finally:
        board.release_session()
