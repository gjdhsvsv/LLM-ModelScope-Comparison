from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer, GenerationConfig
import torch
import time
import os

models = {
    "1": {
        "name": "Qwen-7B-Chat",
        "path": "/mnt/data/Qwen-7B-Chat",
        "type": "qwen"
    },
    "2": {
        "name": "ChatGLM3-6B",
        "path": "/mnt/data/chatglm3-6b",
        "type": "chatglm"
    },
    "3": {
        "name": "Baichuan2-7B-Chat",
        "path": "/mnt/data/Baichuan2-7B-Chat",
        "type": "baichuan"
    }
}

questions = {
    "1": "请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "2": "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "3": "他知道我知道你知道他不知道吗？这句话里，到底谁不知道？",
    "4": "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    "5": "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思。"
}

print("========== 模型选择 ==========")
print("1. Qwen-7B-Chat")
print("2. ChatGLM3-6B")
print("3. Baichuan2-7B-Chat")

model_choice = input("\n请选择要测试的模型编号：").strip()

if model_choice not in models:
    print("模型编号无效，程序退出。")
    exit()

info = models[model_choice]
model_name = info["name"]
model_path = info["path"]
model_type = info["type"]

if not os.path.exists(model_path):
    print(f"模型路径不存在：{model_path}")
    print("请先确认模型是否已经下载到 /mnt/data。")
    exit()

print(f"\n正在加载模型：{model_name}")
print(f"模型路径：{model_path}")

tokenizer = None
model = None

if model_type == "qwen":
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto"
    ).eval()

elif model_type == "chatglm":
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    ).float().eval()

elif model_type == "baichuan":
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    offload_dir = "/mnt/data/baichuan_offload"
    os.makedirs(offload_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={"cpu": "24GiB"},
        offload_folder=offload_dir,
        offload_state_dict=True
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(model_path)

print(f"{model_name} 加载完成。")
print("输入 1-5 选择问题，输入 all 依次测试全部问题，输入 0 退出。")

while True:
    print("\n========== 问题菜单 ==========")
    print("1. 冬天/夏天：能穿多少穿多少")
    print("2. 单身狗：谁都看不上")
    print("3. 他知道我知道你知道他不知道吗")
    print("4. 明明和白白谁喜欢谁")
    print("5. 多个“意思”分别是什么意思")
    print("all. 依次测试全部 5 个问题")
    print("0. 退出")

    choice = input("\n请输入问题编号：").strip()

    if choice == "0":
        print("已退出。")
        break

    if choice == "all":
        selected = questions.items()
    elif choice in questions:
        selected = [(choice, questions[choice])]
    else:
        print("输入无效，请输入 1-5、all 或 0。")
        continue

    for qid, prompt in selected:
        print(f"\n===== {model_name} 问题 {qid} =====")
        print("问题：")
        print(prompt)
        print("\n回答：")

        t0 = time.time()

        if model_type == "qwen":
            inputs = tokenizer(prompt, return_tensors="pt").input_ids

            streamer = TextStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            outputs = model.generate(
                inputs,
                streamer=streamer,
                max_new_tokens=300
            )

        elif model_type == "chatglm":
            response, history = model.chat(
                tokenizer,
                prompt,
                history=[]
            )
            print(response)

        elif model_type == "baichuan":
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = model.chat(
                tokenizer,
                messages
            )
            print(response)

        print(f"\n耗时：{time.time() - t0:.2f}s")
