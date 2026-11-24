# LUMO V1.1 性能基准

## 已测结果

ARM64 工作区使用同一个本机配置和相同的 12 个内置 CSV，每个配置对每个文件测 30 次。两组交替测量，起始位姿为配置的中立位。两组都读取并校验文件字节，重新生成衔接轨迹并验证结果一致。优化组使用预热的内存缓存。结果只覆盖软件轨迹准备，不含关节状态读取和实体运动。

| 指标 | 无缓存 | 热缓存 | 降幅 |
| --- | ---: | ---: | ---: |
| 轨迹准备 P50 | 7.792 ms | 1.429 ms | 81.7% |
| 轨迹准备 P95 | 77.195 ms | 5.470 ms | 92.9% |
| 轨迹准备平均值 | 15.382 ms | 1.938 ms | 87.4% |
| `idle` 单项 P50 | 77.768 ms | 5.745 ms | 92.6% |

工作区未枚举到音频设备，未启用真实语音云测或实体运动。录音、ASR、LLM、TTS、语音首响与灯体首动的现场对比均待实测。71 项自动化测试通过。实时 ASR 另通过本机回环 WebSocket 模拟服务，ARM64 原生 WebRTC VAD 通过静音 PCM 冒烟测试。40% 首响改善仍是现场工程目标。

## A/B 测试

配置真实 ASR 和 TTS 后，由开发者主动运行以下命令来测量云端调用。所有组使用同一 LLM 配置、同一设备、同一网络及相同的语料。A 是当前代码的非流式对照组，保留公共的观测与调度改进。A–C 使用同一非流式 TTS。D 和 E 使用同一流式音色和模型。尽可能采用两种 TTS 模式都支持的音色，并记录可用音色的差异。

```bash
uv run -m lelamp.demo ai --dry-run --device 2 \
  --no-vad --no-streaming-asr --no-streaming-tts --no-reuse-audio --no-motion-cache \
  --benchmark --benchmark-label A --benchmark-output perf-a.jsonl

uv run --extra vad -m lelamp.demo ai --dry-run --device 2 \
  --vad --no-streaming-asr --no-streaming-tts --no-reuse-audio --no-motion-cache \
  --benchmark --benchmark-label B --benchmark-output perf-b.jsonl

uv run --extra vad -m lelamp.demo ai --dry-run --device 2 \
  --vad --streaming-asr --no-streaming-tts --no-reuse-audio --no-motion-cache \
  --benchmark --benchmark-label C --benchmark-output perf-c.jsonl

uv run --extra vad -m lelamp.demo ai --dry-run --device 2 \
  --vad --streaming-asr --streaming-tts --no-reuse-audio --no-motion-cache \
  --benchmark --benchmark-label D --benchmark-output perf-d.jsonl

uv run --extra vad -m lelamp.demo ai --dry-run --device 2 \
  --vad --streaming-asr --streaming-tts --reuse-audio --no-motion-cache \
  --benchmark --benchmark-label E --benchmark-output perf-e.jsonl

uv run -m lelamp.behavior.performance --baseline A \
  perf-a.jsonl perf-b.jsonl perf-c.jsonl perf-d.jsonl perf-e.jsonl
```

每组至少收集 30 次有效交互，同时保留失败记录。每次启动的第一轮标为 `cold`，之后标为 `warm`。要获得可比较的冷启动分布，需要为每轮新启进程并追加到同一个输出文件。稳定轮次之间等待语音播放完成，再发起下一轮。等最后一个任务完成后输入 `q` 退出以写入 JSONL。退出时仍在执行的任务会被取消并计入失败。音频半双工直接挡下的按键没有启动新一轮录音。

实体动作对比需要另开已授权的实机批次。所有组使用同一适配、授权和硬件设置，并在 E 组开启 `--motion-cache`。用同一现场音视频记录用户末音、扬声器首音和点亮标记或可观察的灯体首动，计算用户说完至实际响应的时间。固定录音路径无法自动确定用户何时说完，相关软件字段为空。跨模式的物理响应对比应使用现场记录。

| 指标 | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| 录音耗时 | 待测 | 待测 | 待测 | 待测 | 待测 |
| ASR 最终返回 | 待测 | 待测 | 待测 | 待测 | 待测 |
| LLM 请求耗时 | 待测 | 待测 | 待测 | 待测 | 待测 |
| TTS 首个 PCM | 待测 | 待测 | 待测 | 待测 | 待测 |
| 用户末音至首次舵机指令和实际首动 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 用户末音至首次提交音频和实际首音 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 端到端首次响应与失败率 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 计时口径

每行 JSONL 包含交互 ID、调度及任务结果、相对单调时钟的事件、各阶段耗时和缓冲区计数。报告按配置与冷暖阶段输出样本数、P50、P95、平均值和失败率，同时计算相对基线的变化百分比。缺少相应事件的指标以空值记录，从该项分位数样本中排除。

| 指标 | 时间区间 |
| --- | --- |
| `recording_ms` | 开始采集至采集结束 |
| `endpoint_latency_ms` | 从最后一个判为有声的帧至判定端点 |
| `asr_final_latency_ms` | 采集结束至最终转写到达。另有 `asr_total_ms` 记录 ASR 开始至最终转写 |
| `llm_total_ms` | LLM 调用开始至返回完整结果 |
| `dispatch_latency_ms` | LLM 返回至行为入队，包含 JSON 与能力校验 |
| `motion_queue_ms`、`motion_preparation_ms` | 提交至动作任务实际开始，以及开始至轨迹准备完毕 |
| `tts_queue_ms`、`tts_ttfa_ms` | 提交至音频任务开始，以及请求至首个有效 PCM |
| `first_motion_command_latency_ms` | 最后一个有声帧至首次成功发送舵机指令 |
| `first_audio_output_latency_ms` | 最后一个有声帧至首个 PCM 块提交给音频设备 |
| `end_to_end_ms` | 最后一个有声帧至最早可测的输出提交 |
| `audio_after_capture_ms`、`motion_after_capture_ms` | 从采集结束起算的首段音频与首条舵机指令，便于跨录音模式检查内部处理 |

首次舵机指令可能就是保持在起始位姿。音频提交后，设备仍可能有自己的播放缓冲。物理首动和首音以现场信号为准。

## 回归与现场语料

```bash
uv run -m unittest discover -s tests -v
uv run -m lelamp.intelligence.evaluate --live --verbose
```

真实模型评估继续使用仓库现有用例并输出意图准确率、动作选择准确率、JSON 有效率和误触发率。现场语音测试要保存人工参考文本，按同一分词规范计算 WER。中文可同时报告字符错误率。评估 TTS 时由听者按相同准则评价中英文发音、语调与情绪表达，并核对动作和台词时序。

| 场景 | 语料或注入 | 核对 |
| --- | --- | --- |
| 否定与短暂停顿 | “请先别点头，……把灯打开。” / “Don't nod, … turn on the light.” | 否定词和后半句完整，只有请求的灯光动作 |
| 组合指令与快语速 | “Turn on the light and shake your head.” | 两项指令完整并符合授权 |
| 长句与较长停顿 | 用接近配置上限的长句，并在句中逐步增加停顿 | 记录首次出现截断的停顿，调整端点阈值后复测 |
| 知识和情感 | “Why is the sky blue?” / “I had a difficult day.” | 内容正确、语气合适，附带动作符合授权 |
| 噪声 | 静默、舵机运转声、风扇声、这些声音叠加正常说话 | 记录误触发和漏检 |
| 失败与取消 | ASR 中断、仅发部分结果、TTS 中断、用户 Ctrl+C | 无迟到动作，无重复或迟到播放 |
