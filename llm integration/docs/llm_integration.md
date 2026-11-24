# LUMO LLM Integration V1.0

## 架构与代码位置

现有的 `main.py` 和 `smooth_animation.py` 通过 LiveKit Realtime Agent 使用舵机和 RGB 服务。独立演示 `lelamp.demo voice` 使用 `record_audio()` 采集 16 kHz 单声道 PCM16 WAV，交由 `BailianASR` 转写，再用 `match_command()` 匹配有限的英文口令。舵机控制由 follower、`MotionProfile`、`DemoPlayer` 和 CSV 轨迹适配模块提供。

新版 `chat` 和 `ai` 走独立的 LLM 路径。文本或 ASR 返回的有效转写作为唯一用户消息进入 Chat Completions。系统消息包含人格提示、当前可用能力和必要状态，例如灯光设定亮度。每次请求都重新构建两条消息。模型只返回高层行为，底层执行先经过本地验证。

```text
键盘 / 麦克风 → BailianASR（仅语音）→ LLMClient → Behavior Event JSON
                                                       ↓
                                                CommandDispatcher
                                                       ↓
                                                BehaviorManager
                                     ┌─────────────────┼─────────────────┐
                                     ↓                 ↓                 ↓
                               DemoPlayer        LightingController      TTS
```

- `lelamp/intelligence/` 包含 [System Prompt](../lelamp/intelligence/system_prompt.md)、模型客户端、严格事件协议和评估工具。
- `lelamp/behavior/` 包含能力目录、调度器、分发器和现有回放层的适配。
- `lelamp/audio/`、`lelamp/lighting/` 和 `lelamp/ai_runtime.py` 分别提供语音合成、灯光接口及完整交互入口。
- 新代码和修改后的演示模块集中在仓库的 `llm integration/` 下。原路径通过相对符号链接继续供 Python、uv 和 unittest 使用。以下命令从仓库根目录执行。迁移 checkout 时请保留仓库中的符号链接。

## 配置

在尚未创建本机 `.env` 时执行 `cp .env.example .env`。已有文件请补齐变量。进程环境优先。三个服务的模型端点和密钥独立配置，真实 `.env` 和 `.env.local` 已被 Git 忽略。

| 服务 | 设置 |
| --- | --- |
| LLM | `LLM_ENABLED=true`，填写 `LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL`。可选择 OpenAI、DeepSeek、Qwen、Ollama 或其他兼容端点。使用服务提供的 Chat Completions 基地址和模型 ID。对无需鉴权的本地端点可用占位值，例如 `LLM_API_KEY=local`。 |
| LLM 兼容 | 默认 `LLM_RESPONSE_FORMAT=text`，由 prompt 请求 JSON。服务支持 JSON Mode 时可设为 `json_object`。`LLM_TOKEN_PARAMETER` 可设为 `max_tokens`、`max_completion_tokens` 或空值。服务要求省略采样温度时将 `LLM_TEMPERATURE` 设为空。客户端非流式请求，默认超时 15 秒，单轮无自动重试。 |
| ASR | 使用相同地域和工作空间的 `DASHSCOPE_API_KEY` 和 `ASR_BASE_URL`。默认 `ASR_MODEL=qwen3-asr-flash`，保留旧版英文 ASR 参数。中文文本可用同一 LLM 入口，中文录音识别尚未验证。 |
| TTS | 默认关闭，回复输出到终端。可通过 `TTS_ENABLED=true`、`TTS_PROVIDER=mock` 练习半双工调度。使用 `openai` provider 时填入独立的 `TTS_API_KEY`、`TTS_BASE_URL`、`TTS_MODEL`、`TTS_VOICE`，并可设置 `TTS_DEVICE`、`TTS_TIMEOUT`。OpenAI 兼容端点须返回 24 kHz 单声道 PCM16。百炼 Qwen-Audio-TTS 使用 `TTS_PROVIDER=dashscope`，`TTS_BASE_URL` 填同地域业务空间的 HTTPS `/api/v1` 基地址。项目请求 24 kHz WAV 并交给现有播放器。[Qwen-Audio-TTS HTTP API](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-http-api) |
| 灯光 | 预演使用 Mock。实体模式只有 `LIGHTING_ENABLED=true` 且通过集成工厂提供经验证的控制器才可用。当前标准 CLI 没有实体 LED 后端。 |

## 运行

```bash
uv run -m lelamp.demo chat --dry-run
uv run -m lelamp.demo devices
uv run -m lelamp.demo ai --dry-run --device 2
```

文本模式显示模型回复、通过协议校验的 JSON 及调度状态。语音模式按回车录音，默认 4 秒，可用 `--duration` 调整。运动在独立线程运行，可以继续按键录音。TTS 排队、合成或播放时会阻止开始新的普通录音。终端的 `/stop`、`/resume` 和退出命令仍然可用。

新版提示词要求普通回复选择一个已获自主授权的表情动作，例如知识问答配轻点头。明确动作指令只播放请求的动作。停止、保持安静或不动、动作指令含糊或当前无合适动作时允许只回复文字。实体模式从本机授权读取 `allow_autonomous`。

实机启动需要明确的本机授权，见下节。

```bash
uv run -m lelamp.demo ai --id mylamp --port /dev/ttyACM0 \
  --profile mylamp-demo.json --device 2 --enable-motion
```
```bash
uv run -m lelamp.demo ai --enable-motion \
  --id mylamp --port /dev/ttyACM0 \
  --profile mylamp-demo.json --device 2 --scale 0.25
```
`--debug-ai` 会在日志显示转写和原始事件。默认日志记录交互 ID、各服务耗时、意图、情绪、验证结果及行为生命周期。默认在终端显示回复，程序不写入录音、转写或对话日志文件。启用云端服务会将音频或文本发送给相应服务。诊断输出会屏蔽当前配置的密钥。

### 百炼 ASR 返回 HTTP 400

运行 `uv run -m lelamp.demo voice --dry-run --device 0`，按回车说一句英文，单独复现 ASR 请求。程序会显示百炼返回的错误码、Request ID 和经过脱敏的错误说明（服务返回这些字段时）。若显示模型或端点不匹配，在百炼控制台打开该模型的 API 调用示例，对照 `ASR_MODEL`、`ASR_BASE_URL` 以及同业务空间的 Model Studio API Key。北京业务空间的 SDK Base URL 以 `/compatible-mode/v1` 结尾，由 SDK 追加 `/chat/completions`。[百炼 ASR API](https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference) · [百炼错误码](https://help.aliyun.com/zh/model-studio/error-code/)

## 本机动作授权

先按[原有校准和轨迹验证指南](../../docs/demo.md)完成校准。逐关节验证限位和最大速度，支撑灯体并清空工作空间，按真实灯体、当前配置、目标录制文件和幅度做单动作检查。已验证的记录可写入本机 MotionProfile 的 `ai_motion` 字段。仓库内没有默认已授权动作。

以下示例展示如何计算校验值。计算哈希只是让本机授权与输入文件绑定，动作安全验证由操作者完成。

```bash
uv run python -c 'from lelamp.behavior.capabilities import profile_fingerprint; from lelamp.demo_motion import MotionProfile; print(profile_fingerprint(MotionProfile.load("mylamp-demo.json")))'
sha256sum "llm integration/lelamp/motion/nod.csv"
```

保留 MotionProfile 原有字段，仅在同一 JSON 对象中追加类似的授权。

```json
{
  "ai_motion": {
    "profile_sha256": "<MotionProfile hash above>",
    "actions": {
      "nod": {
        "recording_sha256": "<recording file hash above>",
        "max_scale": 0.25,
        "allow_autonomous": false
      }
    }
  }
}
```

`--scale` 不得超过授权上限。实机授权默认只接受显式指令，符合条件的录制可由操作者设置 `allow_autonomous: true` 允许情感交互触发。程序打开串口之前预检配置指纹和 CSV 安全适配。每次播放前再读取真实关节位置、检查 follower 校准及 CSV 字节指纹。原始录制以相对位移适配并受现有位置和速度限制。正常动作包含回到本机待机姿态的轨迹。停止会取消后续帧。

内置白名单是 `nod`、`headshake`、`curious`、`excited`、`happy_wiggle`、`sad`、`scanning`、`shock`、`shy`、`wake_up`、`idle`。预演可使用仓库存在的白名单动作。实体运行只发布当前本机授权的录制。`scanning` 是预设环顾动作。`idle` 和 `wake_up` 都需要显式请求，每次只播放一次。LLM 预检和实机播放都读取 `llm integration/lelamp/motion/`。内部测试录制 `first_test.csv` 不进入 LLM 白名单。

扩展 `turn_left`、`turn_right`、`raise_arm` 或 `lower_head` 时，先用现有录制工具制作对应 CSV，用 `play --source recording --recordings-dir "llm integration/lelamp/motion" --dry-run` 检查适配后的轨迹，再完成本机安全验证。与内置动作使用相同的授权流程。目前这四个录制没有随仓库提供。

## 行为协议与调度

```json
{
  "intent": "combined_control",
  "user_emotion": "neutral",
  "speech": "Sure!",
  "actions": [
    {"type": "motion", "name": "nod"},
    {"type": "lighting", "operation": "set_brightness", "value": 50}
  ]
}
```

| 意图 | 允许的动作 |
| --- | --- |
| `motion_control` | 一个显式授权的 `motion` |
| `light_control` | 一个 `lighting` |
| `emotion_expression` | 零个动作，或各至多一个允许自主触发的 `motion` 和 `lighting` |
| `conversation`、`no_action` | 空列表 |
| `combined_control` | 一个显式授权的 `motion` 和一个 `lighting` |
| `stop` | 仅 `{"type":"stop"}` |

四个顶层字段均必填。`speech` 可为空字符串，最多 1000 个字符。情绪枚举见 [schemas.py](../lelamp/intelligence/schemas.py)。灯光 `set_brightness` 接受 0–100，`increase_brightness`、`decrease_brightness` 接受 1–100 个百分点，结果限制在 0–100。`turn_on`、`turn_off`、`reading_mode`、`night_mode` 无 `value`。关灯保留设定亮度，读取状态同时给出开关状态。未知字段、非有限数值、越界值、冲突动作和未获授权的能力会拒绝整条事件。

P0 是停止，P1 是显式控制，P2 是情感表达，P3 用于最低优先级回应。调度器默认每个执行通道最多积压两个待执行任务。高优先级能取消同通道的低优先级任务。单一运动 worker 在旧任务停止写入后才执行下一个任务。灯光和音频有独立通道。终端显示行为的 `accepted` 和后续 `success`、`failed`、`cancelled`、`unsupported` 或 `rejected`。安排行为后，预先合成的回复应使用意向表达，终端结果来自执行器。TTS 失败时仍可阅读终端回复。

## 停止和故障

- 在输入提示处输入 `/stop`，立即锁住本地调度并清除待执行行为。现场检查安全后由操作者输入 `/resume`，恢复只接受新的交互。进行录音或等待云端时可使用 Ctrl+C，它会退出会话、取消行为，并等待运动线程退出再关闭串口。
- 经 LLM 验证的语音 stop 也会立即请求取消并清空队列。它受云端延迟及半双工约束。停止会废弃此前签发的交互，迟到结果无法重新启动运动。
- 舵机通信故障会锁住运动通道直至重启。保留扭矩时末帧已经发出的目标可能仍在执行，软件停止不保证机械瞬时刹车。需要切断电源时，先支撑灯体，使用现场独立硬件断电装置。本次没有安装或验证实体急停开关。
- 程序结束时只有操作者支撑灯体并明确键入 `RELEASE` 才请求卸力。回车、EOF 或关闭提示时中断会关闭串口并维持现有扭矩状态。本地停止流程已用 Mock 舵机验证，实机停止距离和停止时延仍待现场验证。

## 自动化测试与真实模型评估

```bash
uv run -m unittest discover -s tests -v
uv run -m lelamp.intelligence.evaluate --live --verbose
```

开发验收运行 `uv run -m unittest discover -s tests -v`，共 56 项通过（30 项原有测试、26 项新增测试）。自动化测试使用 Mock 服务及机器人，包含数据协议、授权、回放、取消竞态、并行调度与完整入口回归。真实模型评估只有显式传入 `--live` 才发送 API 请求，可能产生费用。评估使用 62 条独立的[中英文用例](../lelamp/intelligence/eval_cases.jsonl)，覆盖改述、情感、对话、组合指令、否定、引用、缺失上下文、危险字段要求和不可用硬件。

评估报告给出意图准确率（目标 90%）、动作选择准确率、JSON 协议有效率、无动作样例的误触发率、平均 LLM 耗时和平均端到端耗时。动作按允许的动作集合和指定参数比对。端到端从构建交互计时到 Mock 行为完成，适用于文本链路。误触发率统计模型向无动作样例提出动作的比例，包括被能力校验阻止的可解析事件。API 错误在准确率和有效率中计为失败。真实模型意图准确率、百炼端到端现场语音、真实 TTS 发音和实体 LED 或动作安全均待验证。

已实现的扩展接口可注入兼容语音合成客户端或提供可取消的 `speak()`。灯光可经 `lighting_factory` 初始化带验证输出回调的控制器，模式亮度和渐变时长在本地构造参数配置。先完成输出范围、故障和设备关闭验证，再把后端装入具体机器的运行入口。
