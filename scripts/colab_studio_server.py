import os, sys, time, json, re, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hanja

MODEL_ID = "darkstar-11e/sillok-100m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16 if DEVICE=="cuda" else torch.float32, device_map=DEVICE)
model.eval()
print("Model loaded successfully!")

GLOSSARY_DB = {
    "議政府": "영의정·좌의정·우의정이 국정을 총괄하던 조선 최고 행정·의결 관청",
    "承政院": "왕명 출납과 비서 업무를 전담하던 핵심 관서 (도승지 등 6승지)",
    "弘文館": "궁중의 경서·문한을 관장하고 왕의 자문에 응하던 옥당(玉堂)",
    "司憲府": "백관의 비리를 감찰하고 풍속을 바로잡던 대간(臺諫) 관청",
    "司諫院": "국왕에 대한 간쟁과 논박을 담당하던 언론 관서",
    "吏曹": "문관의 인사, 서열, 포상을 담당하던 6조의 으뜸 관서",
    "戶曹": "호구 조사, 조세 징수, 국가 재정을 총괄하던 관서",
    "禮曹": "국가의 전례, 외교 의례, 과거 시험, 교육을 관장하던 관서",
    "兵曹": "무관 인사와 군사 행정, 국방 업무를 담당하던 관서",
    "刑曹": "법률 집행, 사법 재판, 형벌을 관장하던 관서",
    "工曹": "국가 토목, 영선, 수공업 장인을 관리하던 관서",
    "備邊司": "임진왜란 이후 군사와 국정을 총괄한 최고 합의 기구",
    "義禁府": "왕명을 받들어 반역죄, 옥사 등 중죄인을 다스리던 특별 사법 기관",
    "集賢殿": "세종대왕이 훈민정음 창제와 학문 연구를 위해 설치한 학술 기관",
    "奎章閣": "정조가 설립한 왕실 도서관이자 정책 연구의 요람",
    "啓曰": "신하가 임금에게 정식으로 말씀을 올리다",
    "從之": "임금이 신하의 건의와 품의를 그대로 허락하여 따르다",
    "呈啓": "관청의 공문서를 받아서 임금께 정식으로 아뢰어 올림",
    "史臣曰": "사관이 춘추필법에 따라 인물과 역사적 사건을 준엄하게 비평함",
    "下敎": "임금이 신하들에게 명령이나 훈계를 내림",
    "親鞫": "임금이 의금부 등에서 중죄인을 직접 심문함",
}

def make_ruby(text):
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if hanja.is_hanja(ch):
            reading = hanja.translate(ch, "substitution")
            out.append(f"<ruby>{ch}<rt>{reading}</rt></ruby>")
        else:
            out.append(ch)
        i += 1
    return "".join(out)

def make_reading(text):
    return hanja.translate(text, "substitution")

def interpret_story(king, date, subject, title, generated):
    story = [
        f"1. [역사 배경] {king} 재위 시기, \\'{title}\\'에 관한 조정의 중대한 보고와 건의가 제기되었습니다.",
        f"2. [주요 논의] 해당 분야({subject})의 관원들이 국가의 선례와 전례를 살펴 계책을 아뢰었습니다.",
        f"3. [조정 결말] 조정의 신하들과 유학자들이 당시의 국정 과제를 논의하여 역사에 기록하였습니다."
    ]
    sagwan = f"사관(史臣)은 논한다: 예로부터 성군(聖君)의 치세는 신하의 직언을 귀담아듣고 법도를 바로 세우는 데서 시작되었다. \\'{title}\\' 일은 오늘날에도 거울로 삼을 만한 역사적 장면이다."
    
    found_gloss = []
    full_sample = title + " " + generated
    for term, desc in GLOSSARY_DB.items():
        if term in full_sample:
            found_gloss.append({"term": term, "desc": desc})
        elif hanja.translate(term, "substitution") in full_sample:
            found_gloss.append({"term": f"{term} ({hanja.translate(term, 'substitution')})", "desc": desc})
    return story, sagwan, found_gloss[:4]

class StudioHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "checkpoint": "darkstar-11e/sillok-100m",
                "global_step": 4400,
                "best_val_loss": 3.84,
                "parameters": 100700000,
                "device": DEVICE,
                "supported_kings": 30,
                "status": "ready"
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return

            king = payload.get("king", "세종실록(世宗實錄)")
            date = payload.get("date", "세종 28년 9월 29일 | 庚戌")
            subject = payload.get("subject", "문화-출판/인쇄")
            title = payload.get("title_ko", "")
            hanja_body = payload.get("body_hanja", "")
            target_mode = payload.get("target_mode", "hanja")
            max_new_tokens = int(payload.get("max_new_tokens", 65))
            temperature = float(payload.get("temperature", 0.6))
            rep_pen = float(payload.get("repetition_penalty", 1.15))

            if target_mode == "title":
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|hanja|>{hanja_body}<|title|>"
            elif target_mode == "saron":
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|title|>{title}<|hanja|>{hanja_body}<|postscript|>"
            else:
                prompt = f"<|bos|><|king|>{king}<|date|>{date}<|subject|>{subject}<|title|>{title}<|hanja|>"

            t0 = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.1),
                    repetition_penalty=rep_pen
                )
            elapsed = time.time() - t0
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            new_text = full_text[len(prompt):].replace("<|eos|>", "").strip()
            
            p_toks = inputs["input_ids"].shape[1]
            g_toks = outputs.shape[1] - p_toks
            tok_per_sec = round(g_toks / max(elapsed, 0.001), 1)

            ruby_html = make_ruby(new_text)
            reading = make_reading(new_text)
            story_sum, saron_c, gloss = interpret_story(king, date, subject, title, new_text)

            res = {
                "prompt": prompt,
                "generated": new_text,
                "full_text": full_text,
                "hangul_reading": reading,
                "ruby_html": ruby_html,
                "story_summary": story_sum,
                "sagwan_comment": saron_c,
                "glossary": gloss,
                "prompt_tokens": p_toks,
                "generated_tokens": g_toks,
                "elapsed_sec": round(elapsed, 3),
                "tokens_per_sec": tok_per_sec,
                "global_step": 4400,
                "device": DEVICE
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server = HTTPServer(("0.0.0.0", 8020), StudioHandler)
    print("Serving Studio API on 0.0.0.0:8020...")
    server.serve_forever()

if __name__ == "__main__":
    run()
