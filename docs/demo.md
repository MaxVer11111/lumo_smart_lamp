# 三个动作与英文口令演示

自然语言对话和动作控制见 [LUMO LLM 集成指南](../llm%20integration/docs/llm_integration.md)。

本入口独立于 `main.py`，启动时不会自动播放 wake_up，不依赖 LiveKit、RGB 或扬声器。
请先完成校准，再开启舵机运动。未校准时可以列出动作、检查麦克风和做语音预演。

## 1. 校准 follower

编号已完成时不需要重新运行 setup_motors。先核对五个关节 ID：
base_yaw=1、base_pitch=2、elbow_pitch=3、wrist_roll=4、wrist_pitch=5。

```bash
uv run lerobot-find-port
uv run -m lelamp.calibrate --id mylamp --port /dev/ttyACM0 --follower-only
```

支撑灯体，按提示将各关节放到活动范围中间，再逐个手动采集允许的范围。
不要强推机械限位。已有校准文件但需要重新标定时，在提示处输入 `c`。
使用同一用户运行校准和演示，避免 sudo 导致校准文件保存在另一用户目录。

以下命令只读取当前位置和校准指纹，不配置或启用扭矩：

```bash
uv run -m lelamp.demo inspect --id mylamp --port /dev/ttyACM0
```

校准中点不是演示待机姿态；待机姿态需在本机另行选择。所有位置、幅度和速度均使用
校准后的 `-100..100` 归一化坐标，**不是角度**。

## 2. 创建本机运动配置并逐关节验证

将灯体放到稳定、无干涉的待机姿态并支撑好，然后运行：

```bash
uv run -m lelamp.demo profile --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json
```

命令读取待机姿态，依次询问每个关节的软件下限、上限、最大单位/秒及三个动作幅度。
上下限应处于已采集的机械范围内，并留出余量。使用已检查的保守范围、速度和幅度；
它们无法由校准文件自动推断。命令只读取位置并保存配置，不移动舵机；已有文件不会覆盖。
配置绑定机器人名称和校准指纹，重新校准后需重新验证并创建配置。

先预览和导出 CSV，再用小偏移逐个验证关节方向与活动空间：

```bash
uv run -m lelamp.demo list
uv run -m lelamp.demo play --profile mylamp-demo.json --action nod --dry-run --export /tmp/nod.csv
uv run -m lelamp.demo jog --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --joint wrist_pitch.pos --delta 1
```

`--delta 1` 表示 1 个归一化单位；请结合本机情况调整。jog 平滑移动该关节后返回起点。
依次检查五个关节。若动作方向不符合预期，先修正装配/校准和配置再继续。
若当前位置在软件范围外，程序拒绝启用运动，需人工调整到范围内。

```bash
uv run -m lelamp.demo play --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --action nod
uv run -m lelamp.demo play --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --action headshake
uv run -m lelamp.demo play --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --action greet
uv run -m lelamp.demo play --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --source recording --action nod
```

点头使用 wrist_pitch，摇头和打招呼使用 base_yaw，各摆动两次；幅度分别配置。
从实测当前位置平滑进入待机姿态，再播放，最后回到待机姿态。
每个动作连续测试 5 次，检查碰撞、突跳、方向及待机位置。

Ctrl+C 中断后不继续发轨迹，也不自动回位。程序保留扭矩，提示支撑灯体后按回车，
然后释放扭矩并关闭串口；输入关闭或再次中断也会执行清理。电源断开时灯体可能下落，
始终准备好支撑。语音识别不承担紧急停止功能。

## 3. 百炼英文识别

复制 `.env.example` 为 `.env`，配置同一地域的 Key 与实际服务地址。
不需要 OpenAI Key 或 LiveKit Key。默认 `qwen3-asr-flash`，使用 OpenAI SDK 的
`chat.completions` 音频输入，不调用 `/audio/transcriptions`。
[官方接口说明](https://help.aliyun.com/en/model-studio/qwen-asr-api-reference)

```bash
uv run -m lelamp.demo devices
uv run -m lelamp.demo voice --dry-run --device 0
```

先检查上一步列出的麦克风索引，再设置 `--device`；省略则使用系统默认输入设备。
按回车录制 4 秒（`--duration` 可调，最长 30 秒），提交 16 kHz 单声道 PCM16 WAV，
显示转写文本、匹配动作和识别耗时。`--dry-run` 不打开舵机串口。

语音默认使用适配后的内置 CSV 动作：

| 英文口令 | 录制动作 |
| --- | --- |
| nod / nod your head | nod.csv |
| shake your head / head shake | headshake.csv |
| hello / say hello | happy_wiggle.csv（打招呼） |
| curious / be curious | curious.csv |
| excited / be excited | excited.csv |
| happy / be happy / happy wiggle | happy_wiggle.csv |
| sad / be sad | sad.csv |
| scan / scanning / look around | scanning.csv |
| shock / shocked / be shocked | shock.csv |
| shy / be shy | shy.csv |
| wake up | wake_up.csv |
| idle / be idle | idle.csv |

`idle` 仅播放一次，不循环。`wake up` 只有收到口令才播放，不会在启动时自动播放。
本机测试文件 `first_test.csv` 不加入语音口令表。
可不连接硬件、不配置云端，查看完整口令列表：

```bash
uv run -m lelamp.demo commands
```

忽略大小写、多余空白和句末 `.!?`；仅完整匹配。不执行未知词、否定句或多指令句。
空结果、网络超时、鉴权及音频错误不触发动作，可以重新按回车尝试。
音频以 Base64 发送至配置的云端服务，程序不保存录音。

### HTTP 401 鉴权失败

若已经发出 HTTP 请求但返回 `401 Unauthorized`，说明当前云端凭据未通过验证。
检查项目根目录 `.env` 中的 `DASHSCOPE_API_KEY` 是否为百炼控制台创建且未失效的
模型 API Key，`ASR_BASE_URL` 是否属于该 Key 对应的地域和工作空间。
不要将 OpenAI、阿里云 AccessKey 或 Coding Plan/Token Plan 专属密钥与通用模型端点混用。
[百炼官方错误说明](https://help.aliyun.com/zh/model-studio/error-code-start)

程序使用已有进程环境变量优先于 `.env`。如果终端曾导出旧配置，使用
`unset DASHSCOPE_API_KEY ASR_BASE_URL ASR_MODEL` 清除终端覆盖值，再从项目目录重新启动。
不要在终端回显密钥，也不要把密钥贴入聊天。鉴权失败后程序结束本次会话，不反复录音，
不触发动作；修改配置后需重启。

预演正常后再启用真实运动：

```bash
uv run -m lelamp.demo voice --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --device 2
```

动作完成后才接受下一次录音，不积压命令。播放/串口错误结束本次运行并清理连接。
输入 `q` 回车退出。每种动作测试 10 次，总计至少 27 次正确触发；另测 10 条无关语句，
应完全不触发。记录日志中的识别耗时和动作耗时，并测试断网和 Ctrl+C。

## 录制与开发验证

### 舵机提示 There is no status packet

`Failed to write 'Lock' on id_=2` 表示该次寄存器写入未收到 2 号舵机（base_pitch）应答，
不是动作限位错误。LeRobot 的关闭扭矩操作也会写 `Lock=0`，因此启动或清理阶段均可能出现。
启动扭矩切换和模式/PID 写入使用最多 3 次额外重试；配置未成功完成时不重新上扭矩。
若清理通信也失败，程序会提示扭矩释放未确认，不能假设灯体已断力。

支撑灯体，退出占用同一串口的其他控制程序；断电后检查 2 号舵机附近的总线插头，
确认独立舵机电源正常，再通电。先单独验证读取：

```bash
uv run -m lelamp.demo inspect --id mylamp --port /dev/ttyACM0
```

inspect 只连接并读取位置和校准，不启用扭矩。若读取仍丢包，应先解决供电/接线/端口占用，
不要反复播放动作、扩大限位或重新编号。语音识别可以继续用 `voice --dry-run` 单独验证。

`lelamp.record` 现在使用 follower 的校准目录，不需要另做 leader 校准。
录制期间关闭扭矩，需支撑灯体；与演示不能同时占用同一串口。CSV 可通过下述适配模式播放。
保留原始回放入口用于旧工作流，新的位置/速度配置限制只由 `lelamp.demo` 执行。

### 复用原 CSV 的动作节奏

用 `--source recording` 选择仓库录制，`--action` 指定 CSV 文件名（不带扩展名）：

```bash
uv run -m lelamp.demo list --source recording
uv run -m lelamp.demo play --profile mylamp-demo.json --source recording --action nod --dry-run --export /tmp/adapted-nod.csv
uv run -m lelamp.demo play --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --source recording --action nod
```

适配使用“本机待机姿态 + 原录制相对首帧的变化 × 缩放系数”。五个关节使用同一个
系数以保留相互配合，不复用原录制的绝对位置。不变更本机限位、速度或校准配置。

`--scale` 默认 `0.25`，是请求幅度的上限；程序根据各关节位置和原始速度进一步缩幅，
保留 20% 可用位置余量和 10% 速度余量。不会为了满足限速而把原片段拉长。
位置可行范围不保证整个灯体无碰撞，首次仍需人工逐个动作验证。

程序读取原时间戳，线性重采样到 30 FPS，保留停顿和节奏，末尾取整误差小于一帧。
动作前后分别平滑进入、回到待机姿态，因此总时长还包含这两段过渡；启动实际位置
离待机姿态较远时会增加进入时间。预演显示 `actual_scale`、原片段时长和导出总时长。
缩放后运动幅度可能很小，先检查导出的轨迹和实机效果；不要直接提高软件限速。

语音也可以选择适配录制：

```bash
uv run -m lelamp.demo voice --profile mylamp-demo.json --source recording --dry-run --device 0
uv run -m lelamp.demo voice --id mylamp --port /dev/ttyACM0 --profile mylamp-demo.json --source recording --device 0
```

语音模式省略 `--source` 时默认使用适配录制，并在日志中显示实际 CSV 名称。
使用 `--source generated` 可切回原先三个生成动作，此时只接受点头、摇头和打招呼口令。
`play` 和 `list` 模式仍默认使用生成动作，播放或列举 CSV 时需指定 `--source recording`。

```bash
uv run -m unittest discover -s tests -v
```

自动测试使用模拟机器人和云端响应，不移动真实硬件，也不调用付费云端接口。
