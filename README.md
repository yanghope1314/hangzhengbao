# Gong-e Hang-jian (工e航鉴)

**Trade Background Verification for International Settlements Using Vessel Physical Co-occurrence Networks**

> Entry for the 17th "ICBC Cup" National College Student FinTech Innovation Competition
> Dalian Maritime University · Houbo Yang, Feiyang Pan · Advisor: Zijian Li
> Track: Financial Security Services

*[中文说明见下方 ↓](#中文说明)*

---

## What This Is

When settling international trade, banks must verify that the **shipping background behind a transaction is genuine**. Today's practice relies on documents self-reported by the trading parties — bills of lading, invoices, contracts — and checks only their formal consistency ("three flows": funds, goods, documents). Regulators have explicitly stated this framework **lacks an executable, traceable, cross-verifiable standard for determining authenticity**.

This project uses **real physical movement data from vessel AIS** as an independent evidence source. It builds a heterogeneous "company–vessel–port" graph, identifies **physical co-occurrence patterns of circular trading between related parties**, and converts them into structured decision variables that can be embedded directly into a bank's approval model.

**In one line**: we do not compete with data vendors — we sit downstream of them. We don't produce data; we change how data is used.

---

## 🚀 Quick Start

### Step 1 — Run the demo (zero downloads, ~10 seconds)

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

**The data the demo needs (1.8 MB) is committed to this repository — nothing to download.**

In the browser you will see the declaration-verification flow, the co-occurrence network, and actual AIS tracks. Switch documents in the left panel and compare `D000` (normal) against `D000F0` (injected fraud) — that contrast is the clearest.

### Step 2 — Reproduce all experiments

```bash
python run_all.py
```

> ⚠️ Full reproduction requires a **16 GB dataset**. If it is missing, `src/setup_data.py` will **download it automatically** (resumable, with MD5 verification). **You do not need this step to view the demo.**

---

## 📁 Repository Layout

```
hangzhengbao/
├── app/streamlit_app.py      ← demo UI (entry point)
├── run_all.py                ← one-command full pipeline (entry point)
├── START_DEMO.bat            ← double-click launcher for Windows
│
├── demo_data/                ← ★ demo snapshot (1.8 MB, committed)
│   ├── README.txt                what this is and where it came from
│   ├── rule_engine_results.csv   verification results for 22 declarations
│   ├── cooccurrence_edges.csv    vessel co-occurrence network
│   ├── vessel_features.csv       voyage feature table
│   └── tracks.csv                AIS tracks for the demo vessels
│
├── src/                      ← 11 functional modules
│   ├── setup_data.py             download / verify / extract data
│   ├── data_loader.py            loading and field validation
│   ├── eda.py                    exploratory analysis and plots
│   ├── feature_engineering.py    AIS base feature engineering
│   ├── network_analysis.py    ★  co-occurrence network construction (core anchor)
│   ├── network_viz.py            network visualisation
│   ├── rule_engine.py            declaration verification rule engine
│   ├── anomaly_model.py          anomaly detection (ablation control)
│   ├── vertical_fl_sim.py        vertical federated learning simulation
│   ├── multimodal_fusion.py      multimodal evidence fusion
│   └── make_snapshot.py          build the demo snapshot
│
├── experiments/              ← 3 experiment scripts + results/
└── requirements.txt          ← dependencies, all versions pinned
```

---

## 🔬 Core Method: Physical Co-occurrence Networks

Every AIS point is binned by (time window × spatial cell); vessels appearing in the same cell form a pairwise co-occurrence edge.

**Key methodological finding (counter-intuitive, but reproducible):**

> **The stronger the co-occurrence, the more likely it reflects routine operations — not anomalies.**
>
> Permanent, high-intensity co-occurrence (present in 12/12 months, thousands of windows) = **infrastructure-like relationships** (harbour tugs, sightseeing ferries).
> **Event-driven, medium-intensity co-occurrence is what carries signal.**

On that basis we introduce a **within-vessel-type z-score**. Baseline co-occurrence intensity differs enormously between vessel-type pairs (median 0 between fishing vessels, 8 between tugs) — without normalisation, tugs occupy the top of the ranking permanently.

---

## 📊 Data Source

| Item | Detail |
|---|---|
| **Dataset** | **HawaiiCoast_GT** |
| Publisher | **Sandia National Laboratories** (US Department of Energy / NNSA) |
| Location | Zenodo DOI `10.5281/zenodo.8253611` |
| Licence | **CC BY 4.0** (free to use and redistribute with attribution) |
| Scale | 3.0 GB compressed / 16 GB extracted; **88,749,176 AIS points / 2,622 vessels / 208 labelled tracks / 154 real incidents** |
| Period & region | 2017–2020, coastal Hawaii |
| Upstream source | MarineCadastre (NOAA / BOEM) |

**The dataset is not committed to this repository** (16 GB). Running `python src/setup_data.py` downloads and MD5-verifies it automatically.

### Why not Chinese data?

We checked candidate sources one by one:

- **China MSA AIS platform (`ais.msa.gov.cn`)**: verified by direct access — requires login, **supports single-vessel queries only, offers no bulk data interface**, and scraping carries clear compliance risk.
- **The more important finding**: no AIS dataset was found that is simultaneously **free, clearly licensed, directly downloadable, and covers Chinese coastal waters**.

**The project therefore takes the compliant path deliberately**: algorithm validation is performed on an openly licensed international benchmark. That dataset is honestly described as *"an open international benchmark for algorithm validation"* — **we do not imply it is Chinese data**.

---

## 🔌 Portability

**All paths are relative** (resolved from each script's own location to the project root), so the project runs **from any directory**, with no dependency on a specific drive letter or working directory.

The **16 GB dataset is not committed** — the demo data is extracted into a 1.8 MB snapshot, and scripts download the full set when needed.

---

## ⚙️ Environment

- Python 3.11+
- Dependencies: see `requirements.txt` (**all versions pinned**)
- Install: `pip install -r requirements.txt`

---

## ⚠️ A Deliberate Honesty Statement

The report **proactively discloses six limitations**. The most important one:

> The ground truth used for validation consists of **maritime safety incidents** (towing, rescue, collision) — **not trade-fraud incidents**.
> The headline number (top-50 lift 369×) shows that *"the co-occurrence network method can surface anomalous vessel interaction"*.
> It does **not** directly establish *"it can identify circular-trade fraud"* — the bridge between the two (the multimodal fusion layer) is not yet fully built.

This is the largest evidence gap in the current approach. We state it plainly rather than downplaying it.

---
---

# 中文说明

**基于船舶物理共现网络的国际结算贸易背景核验系统**

> 第十七届"工行杯"全国大学生金融科技创新大赛参赛作品
> 大连海事大学 · 杨厚博、潘飞扬 · 指导教师：李梓健
> 参赛方向：金融安全服务

---

## 这是什么

银行在做国际结算时，需要核实一笔贸易的**运输背景是否真实**。现有做法依赖企业自报单据（提单、发票、合同）的"三流"形式合规，监管已明确指出这套体系**缺乏可交叉验证的"真实性"判定标准**。

本项目用**船舶 AIS 的真实物理运动数据**作为独立证据源，构建"企业—船舶—港口"异构图，识别**关联方循环贸易的物理共现模式**，并把它转化为可直接嵌入银行审批模型的结构化决策变量。

**一句话定位**：不做数据商的竞争对手，做它们的下游 —— 不生产数据，改变数据的用法。

---

## 🚀 三步上手

### 第一步：看演示界面（**零下载，约 10 秒**）

```bash
python -m streamlit run app/streamlit_app.py
```

> 首次运行前先装依赖：`pip install -r requirements.txt`

**演示所需数据（1.8 MB）已随仓库提交，不需要下载任何东西。**

浏览器打开后可以看到：申报单据核验流程、共现网络、AIS 实际轨迹。
左侧切换单据，选 `D000`（正常）与 `D000F0`（注入欺诈）对比最直观。

### 第二步：复现全部实验

```bash
python run_all.py
```

> ⚠️ 全量复现需要 **16 GB 数据集**。数据缺失时 `src/setup_data.py` 会**自动下载**
> （支持断点续传 + md5 校验）。**看演示不需要这一步。**

---

## 📁 目录结构

```
hangzhengbao/
├── app/streamlit_app.py      ← 演示界面（入口）
├── run_all.py                ← 一键复现全流程（入口）
├── START_DEMO.bat            ← Windows 双击启动演示
│
├── demo_data/                ← ★ 演示快照（1.8 MB，随仓库提交）
│   ├── README.txt                说明这是什么、从哪来
│   ├── rule_engine_results.csv   22 张单据的核验结果
│   ├── cooccurrence_edges.csv    船舶共现网络
│   ├── vessel_features.csv       航次特征表
│   └── tracks.csv                演示涉及船舶的 AIS 轨迹
│
├── src/                      ← 11 个功能模块
│   ├── setup_data.py             数据下载 / 校验 / 解压
│   ├── data_loader.py            数据加载与字段核实
│   ├── eda.py                    探索性分析与图表
│   ├── feature_engineering.py    AIS 基础特征工程
│   ├── network_analysis.py    ★  共现网络构建（主锚点）
│   ├── network_viz.py            网络可视化
│   ├── rule_engine.py            申报单据核验规则引擎
│   ├── anomaly_model.py          异常检测（消融对照）
│   ├── vertical_fl_sim.py        纵向联邦学习模拟
│   ├── multimodal_fusion.py      多模态证据融合
│   └── make_snapshot.py          生成演示快照
│
├── experiments/              ← 3 个实验脚本 + results/（结果图与关键 CSV）
└── requirements.txt          ← 依赖清单，版本已全部锁定
```

---

## 🔬 核心方法：物理共现网络

对每个 AIS 点位按（时间窗 × 空间格）分箱，同一单元内出现的船舶两两构成共现关系。

**关键的方法论发现（反直觉但可复现）**：

> **共现越强，越可能是常规运营，而不是异常。**
>
> 永久性的高强度共现（12/12 月出现、数千个窗口）= **基础设施式关系**（港作拖轮、观光渡轮）；
> **事件性的中等强度共现才是信号**。

据此提出**船型对内 z-score** 评分：不同船型对的共现强度基线差异巨大（渔船之间中位数 0，拖轮之间中位数 8），不归一化则拖轮永远占据榜首。

---

## 📊 数据来源

| 项 | 内容 |
|---|---|
| **数据集** | **HawaiiCoast_GT** |
| 发布方 | **Sandia National Laboratories**（美国能源部国家核安全局下属国家实验室） |
| 地址 | Zenodo DOI `10.5281/zenodo.8253611` |
| 许可证 | **CC BY 4.0**（署名即可自由使用与再分发） |
| 规模 | 3.0 GB 压缩 / 16 GB 解压；**88,749,176 个 AIS 点 / 2,622 艘船 / 208 条标注轨迹 / 154 起真实事件** |
| 时间与区域 | 2017–2020 年，夏威夷沿海 |
| 原始来源 | MarineCadastre（NOAA / BOEM） |

**数据集不随仓库提交**（16 GB）。运行 `python src/setup_data.py` 会自动下载并校验 md5。

### 为什么不用中国数据？

本研究逐一核查过候选数据源：

- **中国海事局 AIS 信息服务平台（`ais.msa.gov.cn`）**：经实际访问核实，需登录、
  **仅支持按单船查询、不提供批量数据接口**，爬取存在明确合规风险。
- **更重要的核查结论**：未能找到任何"**免费 + 许可明确 + 可直接下载 + 覆盖中国近海**"
  的 AIS 数据集。

**因此本项目主动选择合规路径**：使用许可明确的国际公开基准数据集完成算法有效性验证。
该数据集被诚实定位为"**算法验证用的国际公开基准数据集**"，**不暗示是中国数据**。

---

## 🔌 可移植性

**全部路径均为相对路径**（基于脚本自身位置推算项目根），可在**任意位置解压运行**，
不依赖特定盘符或工作目录。

**16 GB 数据集不随仓库提交** —— 演示所需数据已抽取为 1.8 MB 快照；
需要全量数据时，脚本会自动下载。

---

## ⚙️ 环境

- Python 3.11+
- 依赖：见 `requirements.txt`（**版本全部锁定**）
- 安装：`pip install -r requirements.txt`

---

## ⚠️ 一个主动的诚实声明

本项目在报告中**主动披露了六条局限**，其中最关键的一条：

> 验证所用的真值是**海事安全事件**（拖带、救援、碰撞），而非贸易欺诈事件。
> 核心数字（top-50 lift 369×）证明的是"**共现网络方法能找到异常船舶互动**"，
> **不能直接等同于"能识别循环贸易欺诈"** —— 两者之间的桥梁（多模态融合层）
> 目前尚未完全打通。

这是本方案当前最大的证据缺口，我们选择如实写明，而不是淡化处理。
