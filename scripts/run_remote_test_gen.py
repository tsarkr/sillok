import json
import subprocess
import sys

session = "runtimes/r-14a1abd8-dd39-4a69-89f7-6b73c3385cb5/sessions/gen-session"

remote_code = """
import subprocess, sys

# 1. 태조실록 즉위 기사 생성
cmd1 = [
    sys.executable, "scripts/05_generate.py",
    "--checkpoint", "checkpoints/sillok_100m_t4/best.pt",
    "--tokenizer", "artifacts/tokenizer/tokenizer.json",
    "--king", "태조실록(太祖實錄)",
    "--date", "태조 1년 7월 17일 | 1392-07-17L0 | 丙申",
    "--subject", "왕실-국왕/임면",
    "--title-ko", "태조가 수창궁에서 왕위에 오르다",
    "--mode", "hanja",
    "--max-new-tokens", "120",
    "--temperature", "0.7"
]
p1 = subprocess.run(cmd1, cwd="/content/sillok", capture_output=True, text=True)
print("=== [GEN 1] 태조실록 즉위 기사 생성 결과 ===")
print(p1.stdout)
if p1.stderr:
    print("STDERR 1:", p1.stderr[:400])

# 2. 세종실록 훈민정음 창제 기사 생성
cmd2 = [
    sys.executable, "scripts/05_generate.py",
    "--checkpoint", "checkpoints/sillok_100m_t4/best.pt",
    "--tokenizer", "artifacts/tokenizer/tokenizer.json",
    "--king", "세종실록(世宗實錄)",
    "--date", "세종 25년 12월 30일 | 1443-12-30L0 | 庚戌",
    "--subject", "문화-출판(出版), 어문학(語文學)",
    "--title-ko", "임금이 친히 훈민정음 28자를 창제하다",
    "--mode", "hanja",
    "--max-new-tokens", "120",
    "--temperature", "0.7"
]
p2 = subprocess.run(cmd2, cwd="/content/sillok", capture_output=True, text=True)
print("=== [GEN 2] 세종실록 훈민정음 기사 생성 결과 ===")
print(p2.stdout)
if p2.stderr:
    print("STDERR 2:", p2.stderr[:400])
"""

reqs = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "sillok-gen-tester", "version": "0.1"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "execute_code", "arguments": {"session": session, "code": remote_code}},
    },
]

proc = subprocess.run(
    ["python3", "/home/nsua/colab-gpu-bridge/colab_mcp_proxy.py"],
    input="\n".join(json.dumps(r) for r in reqs) + "\n",
    text=True,
    capture_output=True,
    timeout=120,
)

for line in proc.stdout.splitlines():
    try:
        obj = json.loads(line)
        if obj.get("id") == 2:
            res = obj.get("result", {})
            sc = res.get("structuredContent", {})
            print("STDOUT:")
            print(sc.get("stdout"))
            if sc.get("stderr"):
                print("STDERR:", sc.get("stderr"))
            if sc.get("executionError"):
                print("EXECUTION ERROR:", sc.get("executionError"))
    except Exception:
        pass
