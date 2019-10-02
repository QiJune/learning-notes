import argparse
import subprocess
import time
from concurrent import futures

import core_pb2
import core_pb2_grpc
import grpc
import yaml


class MasterServicer(core_pb2_grpc.MasterServicer):
    def __init__(self, endpoint):
        self.endpoint = endpoint
        with open("config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.pserver_endpoints = data["endpoints"]
                self.worker_num = data["worker_num"]
            except yaml.YAMLError as exc:
                print(exc)

    def get_pserver(self, request, _):
        response = core_pb2.EndPoint()
        print(self.pserver_endpoints)
        response.endpoint.extend(self.pserver_endpoints)
        return response

    def start_pserver(self):
        for p in self.pserver_endpoints:
            cmd = "python pserver.py -e " + str(p) + " &"
            subprocess.run(cmd, shell=True, check=True, text=True)

    def start_workers(self):
        for i in range(self.worker_num):
            cmd = (
                "python worker.py -e " + self.endpoint + " -i " + str(i) + " &"
            )
            subprocess.run(cmd, shell=True, check=True, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    args = parser.parse_args()

    # Start Master
    master = MasterServicer(args.endpoint)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    core_pb2_grpc.add_MasterServicer_to_server(master, server)

    print("Starting Master. Listening on endpoint %s." % args.endpoint)
    server.add_insecure_port(args.endpoint)
    server.start()

    # Start Pserver
    master.start_pserver()
    time.sleep(3)

    # Start Worker
    master.start_workers()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
