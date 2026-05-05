import json
from openai import OpenAI
from tqdm import tqdm

API_BASE = "https://api.openai.com/v1"
API_KEY = ""



if __name__ == "__main__":
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )

    benchmark_configs = {'sni': './data/sinst/11_/valid.jsonl',
                    'dolly': './data/dolly/valid.jsonl',
                    'self_instruct': './data/self-inst/valid.jsonl',
                    'vicuna': './data/vicuna/valid.jsonl'
                    }

    for key, path in benchmark_configs.items():
        print(key, path)
        with open(path) as f:
            data = [json.loads(l) for l in f.readlines()]

        gpt3_answers = []

        for item in tqdm(data):
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": item["prompt"]}],
                # temperature=0.7,
            )

            gpt3 = completion.choices[0].message.content

            gpt3_answers.append({
                "prompt": item["prompt"],
                "gpt3_answer": gpt3
            })

        with open(f"data/gpt3_answers/{key}_gpt3_answers.jsonl", "w") as f:
            for item in gpt3_answers:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")