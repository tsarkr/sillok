import json
import subprocess
import sys

session = "runtimes/r-14a1abd8-dd39-4a69-89f7-6b73c3385cb5/sessions/gen-session"

remote_code = """
import subprocess, sys

# 3. 역방향: 한문 원문을 읽고 한글 제목 요약 생성
cmd3 = [
    sys.executable, "scripts/05_generate.py",
    "--checkpoint", "checkpoints/sillok_100m_t4/best.pt",
    "--tokenizer", "artifacts/tokenizer/tokenizer.json",
    "--king", "태조실록(太祖實錄)",
    "--date", "태조 1년 7월 17일 | 1392-07-17L0 | 丙申",
    "--subject", "왕실-국왕/임면",
    "--body-hanja", "○丙申/十七日丙申, 太祖卽位于壽昌宮。 先是, 是日領三司事趙浚、判三司事鄭道傳、同知密直司事南誾等, 協謀推戴, 奉王妃安氏敎書, 廢恭讓王, 百官奉國寶, 號泣陳請, 累日不允。 至是, 浚等率百官, 詣壽昌宮, 奉推戴書以進, 太祖不得已卽位。",
    "--mode", "title",
    "--max-new-tokens", "40",
    "--temperature", "0.3"
]
p3 = subprocess.run(cmd3, cwd="/content/sillok", capture_output=True, text=True)
print("=== [GEN 3] 역방향 한글 제목 요약 생성 결과 ===")
print(p3.stdout)
if p3.stderr:
    print("STDERR 3:", p3.stderr[:400])

# 4. 사론(史論) 생성 테스트
cmd4 = [
    sys.executable, "scripts/05_generate.py",
    "--checkpoint", "checkpoints/sillok_100m_t4/best.pt",
    "--tokenizer", "artifacts/tokenizer/tokenizer.json",
    "--king", "중종실록(中宗實錄)",
    "--date", "중종 14년 11월 15일 | 1519-11-15L0 | 己卯",
    "--subject", "정치-정관/임면",
    "--title-ko", "조광조 등을 하옥하고 사사를 명하다",
    "--mode", "hanja",
    "--max-new-tokens", "150",
    "--temperature", "0.7"
]
p4 = subprocess.run(cmd4, cwd="/content/sillok", capture_output=True, text=True)
print("=== [GEN 4] 중종실록 기묘사화 기사 생성 결과 ===")
print(p4.stdout)
if p4.stderr:
    print("STDERR 4:", p4.stderr[:400])
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
