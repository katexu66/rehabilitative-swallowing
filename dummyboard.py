import sys
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes


def main():
    BoardShim.enable_dev_board_logger()

    # ---- DUMMY BOARD ----
    board_id = BoardIds.SYNTHETIC_BOARD.value
    params = BrainFlowInputParams()  # no serial/mac needed
    board = BoardShim(board_id, params)

    fs = BoardShim.get_sampling_rate(board_id)

    # Synthetic board supports EXG channels; treat them as "EMG-like" for plotting
    channels = BoardShim.get_exg_channels(board_id)
    if not channels:
        channels = BoardShim.get_eeg_channels(board_id)

    board.prepare_session()
    board.start_stream(45000)

    # ---- PLOT ----
    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title="BrainFlow Synthetic Board")
    plot = win.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True)

    seconds = 5
    n_points = int(seconds * fs)
    x = np.arange(n_points) / fs

    curves = []
    buffers = []
    for _ in channels:
        buf = np.zeros(n_points, dtype=np.float64)
        buffers.append(buf)
        curves.append(plot.plot(x, buf))

    def update():
        data = board.get_current_board_data(n_points)  # (rows, samples)

        for i, ch in enumerate(channels):
            y = np.array(data[ch, :], dtype=np.float64, copy=True)

            # pad/trim to window
            y = y[-n_points:]
            if y.size < n_points:
                y = np.pad(y, (n_points - y.size, 0))

            # minimal filtering
            DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
            DataFilter.remove_environmental_noise(y, fs, NoiseTypes.SIXTY.value)
            DataFilter.perform_bandpass(
                y, fs, 20.0, min(80.0, fs / 2 - 1.0), 4, FilterTypes.BUTTERWORTH.value, 0
            )

            buffers[i][:] = y
            curves[i].setData(x, buffers[i])

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # 20 FPS redraw

    win.show()

    try:
        app.exec()
    finally:
        timer.stop()
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
