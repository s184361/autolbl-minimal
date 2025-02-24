from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

server_process = execute_shell_command(
    """
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
--port 30000 --host 0.0.0.0
"""
)

wait_for_server("http://localhost:30000")

print_highlight("Server is ready")
