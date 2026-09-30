"""把 E02 的真实数量、类别和长度整理成图与博客正文。"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def main():
    folder = ROOT / "experiments/E02"
    statistics = json.loads((folder / "statistics.json").read_text(encoding="utf-8"))
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    figures = folder / "figures"; figures.mkdir(exist_ok=True)
    categories = statistics["selected_categories"]["train"]
    grouped = Counter()
    labels = {"no_tool_schema": "没有工具定义", "duplicate_exact": "规范化后完全重复", "argument_schema": "参数不符合 schema",
              "incomplete_conversation": "轨迹结束不完整", "invalid_or_duplicate_tool_name": "工具名缺失或重复"}
    for reason, count in statistics["rejected"].items():
        if reason in labels: grouped[labels[reason]] += count
        elif reason == "SyntaxError" or any(x in reason for x in ["Unterminated", "Expecting", "Invalid", "JSONDecode"]): grouped["调用串无法安全解析"] += count
        else: grouped["其他格式或衔接问题"] += count
    for name in ["lengths", "categories", "rejections"]:
        fig, axis = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        if name == "lengths":
            lengths = {int(k): v for k, v in statistics["length_histogram_256"].items()}
            axis.bar(list(lengths), list(lengths.values()), width=230, align="edge", color="#2563EB")
            axis.set_xlabel("完整消息与工具模板长度（token），组距 256", fontproperties=font)
            axis.set_ylabel("通过结构检查的记录数", fontproperties=font)
            axis.set_title(f"去除精确重复后的长度分布，n={sum(lengths.values()):,}", fontproperties=font)
        elif name == "categories":
            pairs = [("单次调用", "single_call"), ("多轮调用", "multi_turn"), ("不调用工具", "no_call"), ("并行调用", "parallel")]
            values = [categories[k] for _, k in pairs]
            axis.bar(range(4), values, color=["#059669", "#2563EB", "#6B7280", "#D97706"])
            axis.set_xticks(range(4), [k for k, _ in pairs], fontproperties=font)
            axis.set_ylabel("独立任务组中的选中记录数", fontproperties=font)
            axis.set_title("冻结的 10k 训练集包含哪些任务？", fontproperties=font)
            axis.set_ylim(0, max(values) * 1.18)
            for i, value in enumerate(values): axis.text(i, value + 80, f"{value:,}", ha="center")
        else:
            pairs = sorted(grouped.items(), key=lambda item: item[1])
            axis.barh(range(len(pairs)), [v for _, v in pairs], color="#6B7280")
            axis.set_yticks(range(len(pairs)), [k for k, _ in pairs], fontproperties=font)
            axis.set_xlabel("被拒绝的原始记录数", fontproperties=font)
            axis.set_title(f"清洗拒绝原因，合计 {sum(grouped.values()):,} 条", fontproperties=font)
        axis.grid(axis="x" if name == "rejections" else "y", color="#E5E7EB", linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ["top", "right"]: axis.spines[side].set_visible(False)
        path = figures / f"{name}.png"; fig.savefig(path, dpi=300); plt.close(fig)
        write_json(figures / f"{name}.source.json", {"run_id": "E02-R02", "source": "../statistics.json", "source_sha256": sha256(folder / "statistics.json"), "image_sha256": sha256(path)})
    (folder / "notes.md").write_text(f"""# E02：先把数据看明白，再谈 10k 微调

清洗后已经有足够的任务组组成 1k、5k、10k 嵌套训练集，dev 和 test 各保留 500 条。三者按任务组划分，实际集合交集为零。不过，抽查也发现弱标注：能解析、参数类型正确，不等于调用内容一定可靠。

## Hermes 不够 10k，怎么办？

Hermes 下载的三个工具文件合计只有 8,995 条原始记录，两个 1,893 条文件又分别是单轮和多轮表示，不能简单相加当作独立任务。于是补充了 Glaive v2 的 112,960 条。两份来源均为 Apache 2.0 授权，并固定了具体版本。xLAM 需要额外登录和接受访问条款，本轮没有采用。

总计读取 {sum(statistics['raw_sources'].values()):,} 条；通过结构检查且去除完全重复后，留下 {statistics['accepted_records']:,} 条。按工具名、参数签名和规范化首问分组后，得到 {statistics['independent_groups']:,} 个任务组。这是本项目的分组口径，不代表已经排除所有改写或预训练污染。

## 清洗没有替模型补答案

Glaive 有些调用混用了 JSON 和 Python 字符串引号。转换时只做安全反序列化，把字符串参数还原成对象，再检查工具名、必需参数和参数类型；解析失败就拒绝，不猜测缺失内容。公开工具返回是数据里的标注，没有在本机调用那些真实外部服务。

第一次抽查发现三条记录把说明文字和调用拆成了连续的 assistant 消息。第二次运行将它们合并为一个完整回复，调用和参数保持不变。原运行和抽查发现都保留，后续使用第二次运行的冻结数据。

![图 E02-1：E02-R02，121,955 条原始记录的拒绝统计；每条只计一个首要拒绝原因。](figures/rejections.png)

拒绝最多的是没有工具定义的记录。这些记录不属于当前的工具训练表示，不能拿来补足数量。调用串损坏、参数与 schema 不符、轨迹不完整和重复，也分别记账。

## 长度和类别，会影响训练看到什么

![图 E02-2：E02-R02，通过结构检查并去除精确重复后的 62,386 条记录；长度包含完整工具定义。](figures/lengths.png)

多数记录落在 256—512 token。训练只选择能够在 2048 token 内完整表示的记录；先划分任务组，再在训练组中选完整记录，不把调用截一半。dev 和 test 允许保留到 8192 token，数据冻结后不因模型表现而缩小评测分母。

![图 E02-3：E02-R02，10,000 条训练记录，每组只取一条；单次、多轮、并行和不调用工具分开统计。](figures/categories.png)

训练集以单次调用为主，仍保留多轮、并行和不调用工具的样本。增加数据量的结果要结合这个分布看：只看 1k 或 10k 的名字，无法知道模型究竟多学了哪些任务。

## 50 条抽查，暴露了什么？

抽查时逐条对照了任务、工具选择、参数来源和返回衔接。44 条没有发现明显错配，2 条有参数来源问题，另外 4 条带有占位或时间、实体方面的疑点。这是实际抽查结果，不是全数据语义准确率。

例如，用户只说想听轻快音乐，标注却直接填入了两个未提供的 URL；另一个年龄查询填了用户没给出的“当前日期”。还有图像参数只是占位字符串，不能算作真实视觉工具执行。这些问题说明基础清洗还不够，E12 会在等量、类别匹配的条件下进一步比较质量筛选。

复习时可以这样解释泄漏：不同 id 的记录，如果只是同一问题换了数字或引号内容，仍可能属于同一个任务模板。先按组划分，能减少这种变体跨集合；近似匹配只是启发式检查，不能保证发现所有同义改写。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
