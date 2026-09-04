"""终端对话 demo：在网页终端里直接和传感器-LLM 模型对话。

Run: ~/.conda/envs/minimind-o/bin/python demo/chat.py
命令：回车空行=重新提问当前样本 | random 随机样本 | pick <id片段> 选样本
      toggle <wifi|depth|lidar|mmwave|rgb> 开关传感器 | show 信号概览
      gold 显示真实标签 | quit 退出；其他输入 = 作为问题提问
"""
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from framework.llm_sft.demo import DEFAULT_QUESTION, make_engine
from framework.models.alignment import MODALITIES

MOD_LABELS = {"wifi": "WiFi CSI", "depth": "Depth 深度图", "lidar": "LiDAR 点云",
              "mmwave": "mmWave 雷达", "rgb": "RGB 骨架"}


class ChatSession:
    def __init__(self, engine):
        self.engine = engine
        self.avail = {m: True for m in MODALITIES}
        self.sample = None

    def banner(self) -> list:
        ids = self.engine.sample_ids()
        self.set_sample(random.choice(ids))
        return ["=" * 62,
                "🛰 SensorBench 伪 token 对话（27 类动作封闭集；自由提问仅供演示）",
                "命令: random / pick <id片段> / toggle <模态> / show / gold / quit",
                "其他任何输入 = 作为问题向模型提问", "=" * 62]

    def set_sample(self, sid: str) -> str:
        matches = [i for i in self.engine.sample_ids() if sid in i]
        if not matches:
            return f"❌ 找不到样本 id 片段: {sid}"
        self.sample = self.engine.get_sample(matches[0])
        return (f"样本: {self.sample.id}  (真实: "
                f"{self.engine.class_map.get(self.sample.label, '?')})")

    def status_line(self) -> str:
        mods = " ".join(
            f"{MOD_LABELS[m]}{'✓' if self.avail[m] else '✗'}" for m in MODALITIES)
        return f"[传感器] {mods}"

    def handle(self, line: str) -> list:
        line = line.strip()
        if not line:
            return self.ask(DEFAULT_QUESTION)
        low = line.lower()
        if low == "quit":
            return ["👋 bye"]
        if low == "random":
            return [self.set_sample(random.choice(self.engine.sample_ids())),
                    self.status_line()]
        if low.startswith("pick "):
            return [self.set_sample(line[5:].strip()), self.status_line()]
        if low.startswith("toggle "):
            m = line[7:].strip().lower()
            if m not in MODALITIES:
                return [f"❌ 未知模态 {m}，可选: {'/'.join(MODALITIES)}"]
            self.avail[m] = not self.avail[m]
            n_off = sum(1 for v in self.avail.values() if not v)
            note = "（该配置模型未训练过，回答仅观察退化）" if n_off else ""
            return [f"{'✓ 开' if self.avail[m] else '✗ 关'} {MOD_LABELS[m]} {note}",
                    self.status_line()]
        if low == "show":
            return self.show()
        if low == "gold":
            return [f"真实标签: {self.engine.class_map.get(self.sample.label, '?')}"
                    f" ({self.sample.label})"]
        return self.ask(line)

    def ask(self, question: str) -> list:
        res = self.engine.answer(self.sample, avail=self.avail, question=question)
        off = [m for m in MODALITIES if not self.avail[m]]
        out = [f"🤖 {res['text']}",
               f"   判定: {res['class_name'] or '(未匹配任何类别)'}"
               + (f"  ｜ Top3: {res['top3']}" if res["top3"] else "")]
        if off:
            out.append(f"   ⚠ 已关闭: {','.join(off)}")
        return out

    def show(self) -> list:
        out = ["信号概览（帧数 × 形状，mean±std）:"]
        for m in MODALITIES:
            d = self.sample.modalities[m].data
            tag = "✓" if self.avail[m] else "✗(关)"
            out.append(f"  {tag} {MOD_LABELS[m]:<10} {str(d.shape):<18} "
                       f"{d.mean():+.3f}±{d.std():.3f}")
        return out


def main():
    print("加载模型（首次约 1 分钟）...", flush=True)
    engine = make_engine(_ROOT)
    print(f"就绪：{len(engine.sample_ids())} 个验证样本\n", flush=True)
    ses = ChatSession(engine)
    for line in ses.banner():
        print(line, flush=True)
    while True:
        try:
            line = input("\n❓ 你: ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 bye")
            break
        out = ses.handle(line)
        for l in out:
            print(l, flush=True)
        if any("bye" in l for l in out):
            break


if __name__ == "__main__":
    main()
