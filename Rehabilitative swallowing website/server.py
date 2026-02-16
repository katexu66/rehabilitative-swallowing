import asyncio
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes, AggOperations

# PARAMETERS
BOARD_ID = BoardIds.GANGLION_BOARD.value
WINDOW_SECONDS = 5          # for client display (client can choose too)
CHUNK_SAMPLES = 20          # samples per websocket message (smaller = lower latency)
SEND_INTERVAL_MS = 50       # how often to send (pacing)
PORT = "COM4"

app = FastAPI()

SITE_DIR = Path(__file__).parent # in same folder as index.html and style.css
app.mount("/static", StaticFiles(directory=str(SITE_DIR)), name="static")

# directory to save data to
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def init_board():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BOARD_ID, params)

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    channels = BoardShim.get_emg_channels(BOARD_ID)
    if not channels: # if emg_channels is empty use exg channels
        channels = BoardShim.get_exg_channels(BOARD_ID)

    board.prepare_session()
    board.start_stream(45000)

    return board, fs, channels


board, fs, channels = init_board()
print("BOARD INIT OK")

@app.get("/")
def root():
    html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

app.state.pending = {}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    raw_log, env_log = [], []
    app.state.pending[session_id] = (raw_log, env_log)

    n_channels = len(channels)
    await websocket.send_text(json.dumps({"type": "meta", "fs": fs, "channels": n_channels, "session_id": session_id}))

    roll_period = max(1, int(0.05 * fs))  # 50 ms window in samples

    # Continuous stream: always send the latest CHUNK_SAMPLES per channel
    try:
        while True:
            data = board.get_current_board_data(CHUNK_SAMPLES)  # shape: (rows, CHUNK_SAMPLES)

            raw = np.stack([data[ch, :] for ch in channels], axis=1).astype(np.float64, copy=True)  # (chunk, C)

            # Process per channel (from plottingV3.py)
            env = np.empty_like(raw)
            for ci in range(raw.shape[1]):
                y = raw[:, ci].copy()

                DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
                DataFilter.remove_environmental_noise(y, fs, NoiseTypes.SIXTY.value)
                DataFilter.perform_bandpass(y, fs, 40.0, min(100.0, fs/2 - 1.0), 4, FilterTypes.BUTTERWORTH.value, 0)

                raw[:, ci] = y
                y_rect = np.abs(y)
                DataFilter.perform_rolling_filter(y_rect, roll_period, AggOperations.MEAN.value)
                env[:, ci] = y_rect

            # store data
            raw_log.append(raw.copy())
            env_log.append(env.copy())

            # await websocket.send_text(json.dumps({"type": "data", "y": y.tolist()}))
            # send both raw and envelope
            await websocket.send_text(json.dumps({"type": "data", "raw": raw.tolist(), "env": env.tolist()}))
            await asyncio.sleep(SEND_INTERVAL_MS / 1000.0)

    finally:
        # client disconnected or server stop
        # if raw_log: np.save(raw_path, np.concatenate(raw_log, axis=0))
        # if env_log: np.save(env_path, np.concatenate(env_log, axis=0))
        pass

# options for data (saving, discarding, adding metadata)

@app.post("/save/{session_id}")
def save(session_id: str):
    raw_log, env_log = app.state.pending.pop(session_id)
    np.save(DATA_DIR / f"raw_{session_id}.npy", np.concatenate(raw_log, axis=0))
    np.save(DATA_DIR / f"env_{session_id}.npy", np.concatenate(env_log, axis=0))
    return {"ok": True}

@app.post("/discard/{session_id}")
def discard(session_id: str):
    app.state.pending.pop(session_id, None)
    return {"ok": True}

@app.post("/meta/{session_id}")
def save_meta(session_id: str, meta: dict = Body(...)):
    path = DATA_DIR / f"meta_{session_id}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True}

# previous sessions

@app.get("/sessions")
def list_sessions():
    # returns newest first
    ids = sorted(
        [p.stem.replace("raw_", "") for p in DATA_DIR.glob("raw_*.npy")],
        reverse=True
    )
    out = []
    for sid in ids: # testing metadata for sessions
        mpath = DATA_DIR / f"meta_{sid}.json"
        meta = json.loads(mpath.read_text()) if mpath.exists() else {}
        out.append({"id": sid, "label": meta.get("label", ""), "notes": meta.get("notes", "")})
    return {"sessions": out}

@app.get("/session/{session_id}")
def load_session(session_id: str, decim: int = 1):
    raw_path = DATA_DIR / f"raw_{session_id}.npy"
    env_path = DATA_DIR / f"env_{session_id}.npy"
    if not raw_path.exists() or not env_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    raw = np.load(raw_path)   # shape (N, C)
    env = np.load(env_path)

    # optional downsample for faster browser rendering
    decim = max(1, int(decim))
    raw = raw[::decim]
    env = env[::decim]

    return JSONResponse({"session_id": session_id, "raw": raw.tolist(), "env": env.tolist()})

# troubleshooting
@app.websocket("/ws_test")
async def ws_test(websocket: WebSocket):
    print("WS_TEST: connect attempt")
    await websocket.accept()
    print("WS_TEST: accepted")
    await websocket.send_text("ok")

@app.on_event("shutdown")
def shutdown_event():
    try:
        # if raw_log: np.save(raw_path, np.concatenate(raw_log, axis=0))
        # if env_log: np.save(env_path, np.concatenate(env_log, axis=0))
        board.stop_stream()
    finally:
        board.release_session()
