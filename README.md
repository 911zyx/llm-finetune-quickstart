# 小K · 开源大模型微调全流程实战

基于 **Qwen2.5-14B-Instruct + QLoRA(4bit)** 的中文学习助手微调项目,覆盖
**数据清洗 → 指令微调 → 效果评估 → 权重合并 → Web/API 部署** 的完整闭环。
单张 RTX 5080(16GB)即可复现,也可以一键降级到 Colab 免费 GPU。

## 🖥 演示效果

![Gradio 首页](docs/screenshots/01_home.png)

![微调后对话效果:结论先行 + 分点解释](docs/screenshots/02_chat.png)

![编程能力:代码报错诊断](docs/screenshots/03_coding.png)

## ✨ 项目亮点

- **全流程可复现**:每个环节一个脚本,配置集中在 `configs/sft_lora.yaml`
- **真实数据清洗**:格式校验 / 长度过滤 / 隐私脱敏 / 去重,带统计报告
- **消费级显卡训 14B**:QLoRA 4bit 量化,16GB 显存即可训练 14B 模型
- **双部署形态**:Gradio 聊天界面(演示)+ FastAPI(生产接口,OpenAI 风格)
- **量化评估**:验证集困惑度对比 + 微调前后生成对比

## 🗂 目录结构

```
llm-finetune-quickstart/
├── configs/sft_lora.yaml          # 唯一配置文件(模型/数据/LoRA/训练/部署)
├── data/
│   ├── raw/sample_instructions.jsonl   # 人工示例数据(含故意混入的脏数据)
│   ├── raw/generated_instructions.jsonl # self-instruct 合成数据(353 条)
│   └── processed/                      # 清洗产物(train.jsonl / val.jsonl)
├── scripts/
│   ├── clean_data.py              # ① 数据清洗
│   ├── generate_data.py           # ①.5 self-instruct 数据合成(可选)
│   ├── download_model_fast.py     # 多线程分块模型下载器(断点续传)
│   ├── train_sft.py               # ② LoRA/QLoRA 训练(TRL SFTTrainer)
│   ├── evaluate.py                # ③ 评估:困惑度 + 生成对比
│   └── merge_lora.py              # ④ 合并 adapter 为完整模型(可选)
├── app/
│   ├── gradio_app.py              # ⑤a Gradio 聊天演示
│   └── api_server.py              # ⑤b FastAPI 接口部署
├── requirements.txt
└── LICENSE                        # MIT
```

## 🚀 快速开始

### 0. 环境准备

```bash
# RTX 50 系(Blackwell)必须用 CUDA 12.8+ 版 PyTorch,否则无法识别显卡
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **Windows + bitsandbytes 装不上?** 把配置改为 `Qwen/Qwen2.5-7B-Instruct` +
> `use_qlora: false`(bf16 LoRA,16GB 显存同样可训),或使用 WSL2。

### 1. 数据清洗

```bash
python scripts/clean_data.py --config configs/sft_lora.yaml
```

输出统计报告示例(示例数据中故意混入了重复、过短、含手机号的脏数据):

```
原始样本:      45
格式错误丢弃:  0
长度异常丢弃:  1
含隐私信息丢弃: 2
重复丢弃:      1
最终保留:      41  (train=37, val=4)
```

### 2. 训练

```bash
python scripts/train_sft.py --config configs/sft_lora.yaml
```

14B + QLoRA + RTX 5080 约 1~2 小时(示例数据量小);adapter 保存在 `outputs/sft-lora/final/`。

### 3. 评估

```bash
python scripts/evaluate.py --mode ppl        # 困惑度:基座 vs 微调后
python scripts/evaluate.py --mode generate   # 生成对比:同一批问题两边作答
```

### 4. 部署演示

```bash
python app/gradio_app.py      # 浏览器聊天界面 http://127.0.0.1:7860
python app/api_server.py      # API 服务 http://127.0.0.1:8000/docs
```

## 📊 实测记录(RT 5080 16GB,2026-10)

本仓库代码已在以下环境完整跑通:

- **环境**:PyTorch 2.9.1+cu128(Blackwell 必需)+ triton-windows 3.8(TRL 融合 loss 依赖)+ TRL 1.15

**v3(44 人工 + 353 通用合成 + 350 编程合成,共 737 条;当前版本)**

- **编程数据**:`generate_data.py --preset coding` 生成 350 条编程数据(35 个编程主题),90% 含可运行代码块,生成失败率仅 5.7%
- **训练**:664 条训练样本 / 3 epochs / 126 步,14.7 分钟,train_loss **0.60**,token 准确率 86%
- **困惑度**:验证集 基座 10.70 → 微调后 **1.91**
- **编程表现**:写代码、看报错类问题均能保持「结论 + 分点 + 代码块」格式

**v2(44 条人工 + 353 条通用合成,共 393 条)**

- 训练 69 步 7.6 分钟,train_loss 0.83;验证集困惑度 基座 14.81 → 微调后 2.21

**v1(仅 44 条人工数据)**

- 训练 9 步 55 秒,train_loss 2.29;验证集困惑度 基座 52.62 → 微调后 5.37

> 结论:小数据学「风格」,数据量与维度(通用 + 编程)扩充后,拟合度与能力覆盖同步提升。

## 💻 硬件档位

| 显存 | 基座模型 | 配置改动 |
|---|---|---|
| 16GB(本项目) | Qwen2.5-14B-Instruct | `use_qlora: true`(默认) |
| 16GB(更稳) | Qwen2.5-7B-Instruct | `use_qlora: false` |
| 8GB | Qwen2.5-3B-Instruct | `use_qlora: true` |
| Colab 免费 T4 | Qwen2.5-0.5B/1.5B-Instruct | `use_qlora: false`, `bf16→fp16` |

## 📅 两周执行计划

| 阶段 | 时间 | 任务 | 产出 |
|---|---|---|---|
| 环境与基础 | Day 1-2 | 配好 CUDA/PyTorch,跑通 transformers 官方示例 | 可用的训练环境 |
| 数据 | Day 3-5 | 确定助手人设,撰写/收集 100+ 条指令数据,跑通清洗管线 | `data/processed/` |
| 首训 | Day 6-7 | 小规模试训,确认 loss 正常下降 | 第一个 adapter |
| 迭代 | Day 8-10 | 根据生成对比补数据、调超参(lr / epochs / r) | 效果更优的 adapter |
| 评估 | Day 11-12 | 困惑度 + 20 条人工对比,记录前后差异 | 评估结果截图 |
| 部署与交付 | Day 13-14 | Gradio 演示、API 服务,完善 README 并发布 | 可演示的 GitHub 仓库 |

## 🧩 换成自己的数据

原始数据为 JSONL,每行一个样本(Alpaca 格式,`input` 可为空字符串):

```json
{"instruction": "解释一下什么是过拟合", "input": "", "output": "结论:过拟合是……"}
```

1. 将文件放到 `data/raw/` 并修改配置中的 `data.raw_path`
2. 重新运行 `clean_data.py` → `train_sft.py`
3. 修改 `deploy.system_prompt` 定义你自己的人设

## ❓ 常见问题

- **`nvidia-smi` 报 NVML 错误 / torch 找不到显卡**:RTX 50 系需要 CUDA 12.8+ 的 PyTorch(cu128),见快速开始第 0 步;必要时重装最新 N 卡驱动。
- **显存溢出(OOM)**:先把 `per_device_train_batch_size` 降到 1、增大 `gradient_accumulation_steps`;再不行缩短 `max_length` 或降档模型。
- **loss 不下降**:检查数据是否清洗过、学习率是否合适(LoRA 常用 1e-4 ~ 2e-4)。
- **微调后「学傻了」**:减少 epoch、降低学习率,或在数据中混入通用问答防止灾难性遗忘。

## 📄 License

MIT,见 [LICENSE](LICENSE)。示例数据为教学用途原创编写。
