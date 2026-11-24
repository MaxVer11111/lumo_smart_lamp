# LUMO V1.1 低延迟语音链路

原链路依次等待固定录音、完整 WAV 上传及识别、完整 TTS 合成及下载。V1.1 允许录音和 ASR 上传并行，并在收到足够 PCM 后启动播放。行为分发继续以完整 LLM JSON 通过协议和能力校验为前提。LLM 客户端、模型配置、System Prompt、能力注入和单轮无状态请求沿用 V1.0。

```text
麦克风 ── PCM 帧 ── 本地 VAD 或固定时长 ── Qwen 实时 ASR
                     └─ 内存 WAV ── Bailian HTTP ASR
                                             ↓
                                  完整最终转写 → LLM
                                              ↓
                              JSON → Command Dispatcher
                                              ↓
                                     Behavior Manager
                            ┌─────────────────┼─────────────────┐
                          运动线程          灯光线程          音频线程
                            ↓                 ↓                 ↓
                       MotionProfile         LED       TTS → 有界 PCM → 扬声器
```

## 运行

以下命令从仓库根目录执行。沿用现有 `.env` 的 LLM 设置，按 [配置示例](../.env.example) 补充所用语音服务的设置。按回车开始一轮录音。

```bash
uv run -m lelamp.demo devices
uv run -m lelamp.demo ai --dry-run --device 0
uv run --extra vad -m lelamp.demo ai --dry-run --device 0 --vad
uv run --extra vad -m lelamp.demo ai --dry-run --device 0 \
  --vad --streaming-asr --streaming-tts
uv run --extra vad -m lelamp.demo ai --dry-run --device 2 \
  --vad --streaming-asr --streaming-tts --reuse-audio
```

`--no-vad`、`--no-streaming-asr`、`--no-streaming-tts`、`--no-reuse-audio` 和 `--no-motion-cache` 覆盖环境变量，方便对照测试。可单独开启流式 ASR 并配合固定录音。固定录音使用 `--duration`，VAD 使用自己的讲话、静默和等待阈值。

实体运动仍需明确指定 `--enable-motion --id … --port … --profile …`，提供当前本机授权并完成启动确认。授权后可独立开启 `--motion-cache`。`--dry-run` 使用模拟运动，仍可调用配置的云服务和实际语音设备。校准与控制命令见 [V1.0 集成指南](llm_integration.md)。

## 语音设置

| 项目 | 设置 |
| --- | --- |
| VAD | `VAD_ENABLED=true` 或 `--vad`。默认使用 20 ms WebRTC 帧，连续 3 帧触发开始，保留约 300 ms 预录音，至少 200 ms 有声帧，连续 400 ms 静默结束。无有效语音或达到上限仍无完整端点时取消本轮。 |
| ASR | `ASR_PROVIDER=dashscope_realtime` 或 `--streaming-asr`。默认模型为 `qwen3-asr-flash-realtime`，使用 `DASHSCOPE_API_KEY` 和 `ASR_REALTIME_URL`。地址为空时取 `ASR_BASE_URL` 的主机，未配置则取中国内地默认主机。用 `ASR_LANGUAGE=en`、`zh` 或空值设置语言。 |
| 流式 TTS | `TTS_STREAMING_ENABLED=true` 或 `--streaming-tts`。配置 `TTS_STREAMING_VOICE` 为所选模型支持的音色，默认模型为 `qwen3-tts-flash-realtime`。使用 `TTS_API_KEY`，缺省时使用 `DASHSCOPE_API_KEY`。可为 TTS 单独设置 `TTS_STREAMING_URL`；缺省时依次取 `TTS_BASE_URL`、`ASR_BASE_URL` 的主机和中国内地默认主机。 |
| 原有 TTS | `TTS_ENABLED=true` 并配置 `TTS_PROVIDER=openai`、`dashscope` 或 `mock`。关闭流式 TTS 后使用此组设置。 |
| 设备复用 | `AUDIO_REUSE_ENABLED=true` 或 `--reuse-audio`。轮次之间保留已经停下的输入设备句柄与流式输出设备句柄。设备出错后释放，下次重新打开。 |

在百炼控制台核对同一地域的 API Key、模型和主机。实时 WebSocket 使用 `wss://<host>/api-ws/v1/realtime`。具体事件定义见 [Qwen 实时 ASR 协议](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-interaction-process) 和 [Qwen 实时 TTS 协议](https://www.alibabacloud.com/help/en/model-studio/interactive-process-of-qwen-tts-realtime-synthesis)。

WebRTC VAD 的起点和终点由音频帧估计。现场检查低音量、句中停顿、风扇及舵机噪声。若否定或后半句被切断，增大 `VAD_SILENCE_MS` 后用相同语料复测。半双工期间，TTS 排队、合成或播放会阻止新的普通录音。

如果录满等待时间仍提示无有效语音，日志会列出输入设备、最强 PCM16 电平、VAD 模式和最佳候选的有声时长。若电平接近零，先用 `uv run -m lelamp.demo devices` 核对输入设备和麦克风增益。若有可见电平但最佳候选持续过短，可以在原启动命令前加 `VAD_MODE=1 VAD_SILENCE_MS=800`。模式 1 是新默认值，模式 3 对非语音的过滤最严格。

## 并行、资源与停止

- ASR 每轮使用一个 WebSocket 会话，分批发送 16 kHz 单声道 PCM16。局部转写可用 `--debug-ai` 查看。每轮使用独立连接并只提交一次。客户端提交后，只有收齐最终转写和 `session.finished` 才进入 LLM。如果提交确认和最终结果都带有对话项 ID，二者必须一致。缺少确认事件或 ID 时，仍需满足完整的会话结束条件。连接建立阶段的暂时性 I/O 失败可重试一次。途中断线、溢出、超时或取消会丢弃本轮，下次录音建立新会话。
- Qwen TTS 接收经验证的完整文本，以 24 kHz 单声道 PCM16 播放，默认缓冲至少 100 ms 后启动。网络消费落后时降速读取，播放欠载时补静音并重新积累缓冲。合成失败会保留终端回复，避免自动重播一段可能已输出的语音。
- 实时服务在 `session.finish` 后结束会话。连接在同一轮任务内复用，下一轮使用独立的会话。DashScope 非流式 TTS 复用持久的 HTTP 合成客户端和签名下载连接池，退出时关闭。签名下载使用 TLS 且独立于合成鉴权，不跟随重定向。
- 运动、灯光、音频沿用各自的执行线程。队列提交保留显式动作优先级，较新的同优先级语音可替代过时语音，较新自主动作可替代排队中的旧自主动作。舵机依然只有一个写入线程。
- 动作缓存复用已验证的相对录制数据。缓存键包含 CSV 哈希、MotionProfile 和校准指纹、有效限位、Scale 与 FPS。每次物理播放仍检查当前动作授权、CSV 哈希、实际校准和关节位置，并重新生成和验证前后衔接。缓存只存在于当前进程。
- 本地 `/stop` 在终端输入状态锁定停止，Ctrl+C 可在录音和云请求期间触发本地取消和退出。两者均由本机处理。语音停止仍依赖 ASR 与 LLM。句中停顿、断网、丢帧或坏 JSON 的测试见 [基准与回归](performance_benchmark.md)。
