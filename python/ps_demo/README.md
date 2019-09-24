```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. kvstore.proto


python pserver.py

python worker.py
```

