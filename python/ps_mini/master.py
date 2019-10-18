import subprocess
import time
import yaml


class Master(object):
    def __init__(self):
        with open("config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.pserver_endpoints = data["endpoints"]
                self.worker_num = data["worker_num"]
            except yaml.YAMLError as exc:
                print(exc)

    def start_pserver(self):
        for p in self.pserver_endpoints:
            cmd = "python pserver/pserver.py -e " + str(p) + " &"
            subprocess.run(cmd, shell=True, check=True, text=True)

    def start_workers(self):
        for i in range(self.worker_num):
            cmd = ("python worker.py -p " + " ".join(self.pserver_endpoints) +
                   " -i " + str(i) + " &")
            subprocess.run(cmd, shell=True, check=True, text=True)


if __name__ == "__main__":
    master = Master()

    # Start Pserver
    master.start_pserver()
    time.sleep(3)

    # Start Worker
    master.start_workers()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        exit(0)
