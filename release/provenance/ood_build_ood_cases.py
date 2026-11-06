"""Held-out workload (NOT used to build any vocabulary): different content than the coding benchmark."""
import json
from transformers import AutoTokenizer
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
tasks = [
 ("english_history", "Write a detailed essay on the causes and consequences of the fall of the Western Roman Empire, discussing economic, military, political and cultural factors and how historians have interpreted them."),
 ("english_biology", "Explain how the human immune system recognizes and destroys a virus the first time it infects someone and how memory is formed so that a second infection is milder. Cover innate and adaptive immunity in depth."),
 ("chinese_explain", "请详细解释什么是区块链技术,它的工作原理、共识机制、主要应用场景以及面临的挑战,并举例说明。"),
 ("chinese_story", "请写一个关于一位老船长和一只神秘海鸟的短篇故事,要有细腻的环境描写、人物对话和一个出人意料的结局。"),
 ("spanish_travel", "Escribe una guia de viaje detallada para una semana en el norte de Espana, con itinerario diario, comida tipica, consejos de transporte y recomendaciones culturales."),
 ("math_reasoning", "Solve step by step and explain every step: a train leaves city A at 9:00 traveling 80 km/h toward city B, 360 km away; another leaves B at 10:00 traveling 100 km/h toward A. When and where do they meet? Then generalize the method to any two trains and discuss common mistakes."),
 ("go_code", "Implement a concurrent web crawler in Go with a worker pool, per-host rate limiting, context cancellation, retry with backoff and deduplication of URLs. Provide complete code and explain the design."),
 ("json_data", "Generate a realistic JSON dataset of 25 fictional customers for an online bookstore with ids, names, emails, addresses, purchase histories with ISBNs and dates, and loyalty tiers, then explain the schema."),
]
cases = []
for label, task in tasks:
    ids = tok.apply_chat_template([{"role": "user", "content": task}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    cases.append({"label": label, "prompt_ids": ids, "output_tokens": 512})
json.dump(cases, open("/u701/j692wu/swift-profile/ood/decode-cases.json", "w"), indent=1)
print([len(c["prompt_ids"]) for c in cases])
