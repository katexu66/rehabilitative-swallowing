import sys
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes, AggOperations


def main():
    BoardShim.enable_dev_board_logger()

    board_id = BoardIds.GANGLION_BOARD.value
    params = BrainFlowInputParams()
    params.serial_port = "COM8"     # for Windows with BLE dongle; COM varies depending on port, usually COM8 for Abi, COM5 for Kate
    # params.mac_address = "XX:XX:XX:XX:XX"  # for MAC; should auto look for it if you don't set though (TEST THIS)

    board = BoardShim(board_id, params)
    fs = BoardShim.get_sampling_rate(board_id) # Ganglion board sampling rate = 200 Hz

    emg_channels = BoardShim.get_emg_channels(board_id)
    if not emg_channels: # if emg_channels is empty use exg channels
        emg_channels = BoardShim.get_exg_channels(board_id)

    # streaming
    board.prepare_session()
    board.start_stream(45000)  # internal ring buffer size (# samples kept in memory per channel)
    # 45000 samples ~ 3.75 min; can reduce for lower memory

    # graphing
    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="Filtered Ganglion EMG data")
    raw_plot = win.addPlot(row=1, col=1)
    env_plot = win.addPlot(row=1, col=2)
    raw_plot.showGrid(x=True, y=True)
    env_plot.showGrid(x=True, y=True)

    seconds = 5 # display last 5 seconds of data
    n_points = int(seconds * fs) # seconds * sampling rate = num of samples in that window
    x = np.arange(n_points) / fs # convert sample indices to time (seconds) for x-axis

    raw_curves = []
    raw_buffers = []
    env_curves = []
    env_buffers = []
    for i, _ch in enumerate(emg_channels):
        raw_buf = np.zeros(n_points, dtype=np.float64) # creates buffer array filled with zeros
        env_buf = np.zeros(n_points, dtype=np.float64)
        # for raw EMG signal
        raw_buffers.append(raw_buf)
        c_raw = raw_plot.plot(x, raw_buf)  # one curve per channel
        raw_curves.append(c_raw)
        # for EMG signal envelope
        env_buffers.append(env_buf)
        c_env = env_plot.plot(x, env_buf)  # plot envelope
        env_curves.append(c_env)

    # update loop
    def update():
        data = board.get_current_board_data(n_points)  # get latest n_points samples/channel as NumPy array with shape (rows, n_points)

        for i, ch in enumerate(emg_channels):
            y = np.array(data[ch, :], dtype=np.float64, copy=True)

            y = y[-n_points:]                    # trim if longer
            if y.size < n_points:
                y = np.pad(y, (n_points - y.size, 0))  # left-pad zeros
            raw_buffers[i][:] = y

            # EMG filtering
            DataFilter.detrend(y, DetrendOperations.CONSTANT.value) # detrend
            DataFilter.remove_environmental_noise(y, fs, NoiseTypes.SIXTY.value) # notch filter for power cord noise (60 Hz in US)
            # bandwidth=2 Hz, order=4
            DataFilter.perform_bandpass(
                y, fs, 40.0, 100.0, 4, FilterTypes.BUTTERWORTH.value, 0
            ) # bandpass filter: 40–100 Hz (based on Nyquist frequency of 100 Hz for Ganglion 200 Hz sampling and 40 Hz for ECG content)

            raw_buffers[i][:] = y
            raw_curves[i].setData(x, raw_buffers[i])

            # Smoothing & rectifying to create envelope
            y_rect = np.abs(y) # rectify raw signal
            window_sz = 10 # duration of window for rolling filter in ms
            DataFilter.perform_rolling_filter(y_rect, period=(50), operation=AggOperations.MEAN.value); # creates a moving average
            y_env = y_rect # signal envelope
            
            env_buffers[i][:] = y_env
            env_curves[i].setData(x, env_buffers[i])

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # update plot every 50 ms (20 updates/second); the lower, the smoother, but higher CPU

    win.show()

    try:
        app.exec()
    finally:
        # end session/stream
        timer.stop()
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
