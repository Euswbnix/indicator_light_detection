# 仪表盘指示灯检测与分类

给定一张用户拍摄的仪表盘照片，框选所有亮起的指示灯并输出类别。

## 架构：两阶段（检测 + 分类）

```
输入图 ──▶ Stage1: YOLOv8s-P2 单类检测 ──▶ 切片合并/DIoU-NMS ──▶ 裁剪 patch
                                                                    │
最终输出 ◀── 后处理(双闪等) ◀── 拒绝兜底 ◀── Stage2: MobileViT-XXS 分类
```

- **Stage1 — YOLOv8s-P2**（`models/yolov8s-p2.yaml`）：单类“有没有灯”。加 P2(1/4) 头抓
  15–40px 小灯；输入 1280；低 conf 阈值宁多勿漏，误检交 Stage2 兜底。
- **Stage2 — MobileViT-XXS**（`models/classifier.py`，自实现，仅依赖 torch）：128×128，
  分 123 类灯 + `not_a_light`。CNN+Transformer 混合，对全局颜色敏感，解决“同形不同色”。

## 关键设计

| 难点 | 方案 | 代码 |
|---|---|---|
| 小灯 + 大图标多尺度 | P2 头 + 切片推理 + 全图兜底 | `inference/tiling.py` |
| 一排小灯被并框 | DIoU-NMS（看中心点距离） | `inference/nms.py` |
| 光照/过曝导致颜色失真 | 分维度增强：V/S 放开，H 只±10°，白平衡模拟 | `data/augment.py` |
| 82 个零样本类 | 知识库图标合成（LCD/LED 双渲染轨） | `data/synth.py` |
| 检测器误检（时钟/温度/反光） | `not_a_light` 类 + 置信度阈值 + 难负样本挖掘 | `train/mine_hard_negatives.py` |
| 同灯一大一小 | 同类，后处理去重 | `inference/pipeline.py` |

## 数据现状

- `done/`：已标注 289 张（文件名 = 灯号列表），779 实例，覆盖 56 类。
- `仪表照片-重庆申通202507/`：492 张新照片（已统一为 .jpg），待标注。
- `仪表灯知识库/`：123 个图标 PNG + 类别表。长尾严重，67 类零样本 → 靠合成补。

## 目录

```
src/ilds/
  config.py            全局配置（路径/超参/阈值）
  taxonomy.py          类别体系（123+not_a_light、近似组、双闪定义）
  classes.json         从知识库导出的静态类表（运行时不依赖 xlrd）
  data/                parse_labels / augment / synth / 数据集
  models/              yolov8s-p2.yaml / classifier.py(MobileViT-XXS)
  inference/           tiling / nms / pipeline（端到端）
  train/               训练与难负样本挖掘脚本
```

## 用法

```bash
PY=~/anaconda3/bin/python
$PY -m src.ilds.taxonomy            # 查看类别体系
$PY -m src.ilds.models.classifier   # 验证分类器前向 & 参数量
$PY -m src.ilds.data.synth          # 生成合成样本预览
```
