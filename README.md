# 常见大语言模型部署与中文语义理解比较

## 1 项目简介

本项目为《人工智能导论》第三次课程作业“大语言模型部署体验”。项目基于 ModelScope 魔搭平台提供的 CPU Notebook 环境，完成多个开源中文大语言模型的下载、部署、问答测试与横向对比分析。

本项目主要完成以下内容：

1. 登录并使用 ModelScope 魔搭平台，关联阿里云账号，获取免费的 CPU 云计算资源；
2. 通过 Jupyter Notebook / Terminal 进入模型部署环境，完成 Python 依赖配置；
3. 下载并部署多个开源中文大语言模型；
4. 使用相同的中文语义理解问题测试不同模型；
5. 从语义理解、指代推理、中文歧义处理、输出结构和部署体验等方面进行横向比较。

本项目参与横向对比的模型包括：

* 通义千问 Qwen-7B-Chat
* 智谱 ChatGLM3-6B
* 百川 Baichuan2-7B-Chat

---

## 2 实验平台与环境

本项目使用 ModelScope 魔搭社区 CPU Notebook 资源完成实验。

实验环境如下：

| 项目        | 配置说明                                                              |
| --------- | ----------------------------------------------------------------- |
| 平台        | ModelScope 魔搭社区 CPU Notebook                                      |
| 运行方式      | Terminal + Python 推理脚本                                            |
| Python 环境 | Python 3.11                                                        |
| 核心依赖      | torch CPU、transformers、modelscope、sentencepiece、tiktoken、einops 等 |
| 模型部署方式    | 本地下载模型权重，并使用 transformers 加载推理                                    |
| 测试方式      | 三个模型使用同一组中文语义理解问题进行问答测试                                           |

由于 7B 级模型文件较大，实验过程中采用“逐个下载、逐个测试、逐步截图留证”的方式完成部署，避免存储空间和运行内存压力过大。

---

## 3 环境配置流程

进入 ModelScope Notebook 后，打开 Terminal，执行以下命令安装基础依赖：

```bash
pip install -U pip setuptools wheel

pip install \
torch==2.3.0+cpu \
torchvision==0.18.0+cpu \
--index-url https://download.pytorch.org/whl/cpu

pip install \
"intel-extension-for-transformers==1.4.2" \
"neural-compressor==2.5" \
"transformers==4.33.3" \
"modelscope==1.9.5" \
"pydantic==1.10.13" \
"sentencepiece" \
"tiktoken" \
"einops" \
"transformers_stream_generator" \
"uvicorn" \
"fastapi" \
"yacs" \
"setuptools_scm"

pip install fschat --use-pep517
pip install tqdm huggingface-hub
```

安装完成后，可通过以下命令检查核心依赖版本：

```bash
python -c "import torch, transformers, modelscope, pydantic; print(torch.__version__); print(transformers.__version__); print(modelscope.__version__); print(pydantic.__version__)"
```

---

## 4 模型下载

模型统一下载到 `/mnt/data` 目录下。

```bash
cd /mnt/data
```

下载 Qwen-7B-Chat：

```bash
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```

下载 ChatGLM3-6B：

```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

下载 Baichuan2-7B-Chat：

```bash
git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-7B-Chat.git
```

下载完成后，可通过以下命令查看模型目录和文件大小：

```bash
ls /mnt/data
du -sh /mnt/data/*
```

---

## 5 问答测试脚本

本项目编写了统一问答测试脚本 `ask_three_models.py`。该脚本可以在运行后选择不同模型，并对相同的问题进行测试，便于进行横向比较。

运行方式：

```bash
cd /mnt/workspace
python ask_three_models.py
```

运行后首先选择模型：

```text
1. Qwen-7B-Chat
2. ChatGLM3-6B
3. Baichuan2-7B-Chat
```

选择模型后，再选择测试问题编号：

```text
1. 冬天/夏天：能穿多少穿多少
2. 单身狗：谁都看不上
3. 他知道我知道你知道他不知道吗
4. 明明和白白谁喜欢谁
5. 多个“意思”分别是什么意思
```

脚本每次只加载一个模型，避免多个 7B 模型同时占用内存。

---

## 6 测试问题设计

本项目采用五个中文语义理解问题作为测试集，主要考察模型对中文歧义、语境反转、主语省略、嵌套指代和多义词的理解能力。

| 编号 | 测试问题                                                                                                            | 考察能力        |
| -- | --------------------------------------------------------------------------------------------------------------- | ----------- |
| Q1 | 请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少                                                                         | 语境反转理解      |
| Q2 | 请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上                                                                       | 主语省略与歧义消解   |
| Q3 | 他知道我知道你知道他不知道吗？这句话里，到底谁不知道？                                                                                     | 嵌套指代推理      |
| Q4 | 明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？                                                                              | 中文分词与人物关系理解 |
| Q5 | 领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思。 | 多义词语境判断     |

---

## 7 测试结果与横向对比

三个模型均使用相同问题进行测试，测试截图和详细分析见项目文件夹及实验报告。

| 对比维度   | Qwen-7B-Chat        | ChatGLM3-6B       | Baichuan2-7B-Chat |
| ------ | ------------------- | ----------------- | ----------------- |
| 中文语义理解 | 对语境反转、中文双关和多义词理解较稳定 | 能识别主要语义差异，回答较简洁   | 表达自然，但复杂问题回答速度较慢  |
| 指代推理   | 能较好解释嵌套指代关系         | 能给出核心答案，但解释层次有时较少 | 能理解部分指代关系，但稳定性略弱  |
| 中文歧义处理 | 对分词歧义和主语省略解释较清楚     | 能完成多数问题，但部分解释较短   | 面对复杂歧义时有时需要更长生成时间 |
| 输出结构化  | 较容易形成条目化、结构化回答      | 回答简洁直接            | 语言自然，但结构化程度不一定稳定  |
| 部署体验   | 教程示例较多，复现较方便        | 模型加载较慢，版本兼容需注意    | 权重较大，CPU 推理压力较大   |
| 综合评价   | 适合作为主要展示模型          | 适合作为对照模型          | 适合作为补充比较模型        |

---

## 8 项目文件说明

本项目主要包含以下内容：

```text
LLM-ModelScope-Comparison/
├── README.md
├── ask_three_models.py
├── hw3_大语言模型部署体验报告.docx
├── Qwen-7B-Chat_测试截图/
├── ChatGLM3-6B_测试截图/
├── Baichuan2-7B-Chat_测试截图/
└── 对比分析截图/
```

说明：

* `README.md`：项目说明文档；
* `ask_three_models.py`：三个模型统一问答测试脚本；
* `hw3_大语言模型部署体验报告.docx`：课程实验报告；
* 各截图文件夹：保存模型下载、部署和问答测试结果截图。

---

## 9 注意事项

由于 7B 级大语言模型权重文件较大，本仓库不上传模型权重文件。实际运行时，需要按照课程教程在 ModelScope 平台中通过 `git clone` 下载对应模型。

CPU Notebook 环境下模型加载和推理速度较慢，部分模型在生成回答时需要较长等待时间。实验过程中应保持 Notebook 实例在线，并及时保存部署截图和问答测试截图。

---

## 10 项目结论

本项目完成了在 ModelScope CPU Notebook 环境中对多个中文大语言模型的部署和测试。通过同一组中文歧义与推理问题可以看出，不同模型虽然都能生成较流畅的中文回答，但在复杂语义理解、指代关系分析、多义词解释和输出结构化方面仍然存在差异。

综合测试结果来看，Qwen-7B-Chat 在中文语义理解和结构化表达方面表现较稳定，适合作为主要展示模型；ChatGLM3-6B 回答较简洁，适合作为对照模型；Baichuan2-7B-Chat 可以补充体现不同模型在中文表达风格和部署体验上的差异。
