import subprocess
import sys
import time
import webbrowser

def main():
    # start the server in the background
    p = subprocess.Popen([sys.executable, "-m", "uvicorn",
                          "server_dummy:app",
                          "--host", "127.0.0.1", "--port", "8000"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

    # give it a moment to start
    time.sleep(1.0)

    # open the UI
    webbrowser.open("http://127.0.0.1:8000")

    # keep launcher alive until server exits
    try:
        p.wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
