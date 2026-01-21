# In terminal: run `python -m pip install brainflow`
# python -m pip install brainflow pyqtgraph pyqt5 numpy
# python -m pip install fastapi uvicorn
# python -m pip install websockets

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import pyqtgraph as pg

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    BoardShim.enable_dev_board_logger() # optional, enables logging for debugging

    # Setting parameters
    params = BrainFlowInputParams()
    # For Windows using BLED112 dongle:
    params.serial_port = 'COM4' # Replace with serial port name; can check in Device Manager -> Ports
    # For Linux/macOS using built-in BLE, would instead use `params.mac_address`:
    # params.mac_address = 'D2:B4:11:22:33:44' # Replace with Ganglion's 4-character ID or full MAC address
    # can use BrainFlow find_mac() to find full MAC address; this will run by default if you don't specify one

    board_id = BoardIds.GANGLION_BOARD.value # GANGLION_BOARD= 1
    board = BoardShim(board_id, params)
    fs = BoardShim.get_sampling_rate(board_id)

    try:
        print("Preparing session...")
        board.prepare_session() # establish connection - same functionality as "Start Session" in OpenBCI GUI
        print("Starting stream...")
        board.start_stream() # begin stream - same functionality as "Start Data Stream" in OpenBCI GUI

        # Stream data
        time.sleep(10) # for 10 seconds

        # Collect data (NumPy array)
        # data = board.get_current_board_data(256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_board_data()  # get all data acquired since stream started and remove it from internal buffer
        print(f"Acquired {data.shape[1]} samples")

        # Optional: Print raw EEG data from the first channel
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        first_channel_data = data[eeg_channels[0]]
        print(f"First EEG channel data sample: {first_channel_data[-1]}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # close connection
        if board.is_prepared():
            print("Stopping stream and releasing session...")
            board.stop_stream()
            board.release_session()
            print("Session released.")

if __name__ == "__main__":
    main()
