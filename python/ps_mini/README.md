```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./ps_mini/proto/core.proto

python master.py

ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
```

