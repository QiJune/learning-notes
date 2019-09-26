```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. core.proto

python master.py -e localhost:8000

ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
```

