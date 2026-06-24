# 多模態流感健康監測與早期預警系統

## Multimodal Influenza Health Monitoring & Early Warning System

**Intel Cup 2025 · Multi-AI Agent Layer v2.0.0**

---

## 專案概述

本系統透過**人臉影片、咳嗽音訊、生理信號**三種資料來源，各自經由專用 AI 模型分析後，將結果融合並分類為**健康 (0)、亞健康 (1)、不健康 (2)**。

在此基礎上，**Multi-AI Agent Layer（多 AI 代理層）** 即時監控資料流——偵測趨勢、標記異常、評估 11 條醫療決策規則，並生成自然語言的健康建議（可選配 GPT-4 或 Claude 等 LLM 增強臨床推理）。所有結果顯示在 **Web 儀表板** 上，包含即時圖表、AI 建議與警報指示。

> **一句話總結：** 感測器資料 → AI 模型分類健康狀態 → AI 代理推理分析 → 儀表板顯示狀況與建議。

---

## 系統架構

```
                       輸入資料
    ┌─────────────────────┼──────────────────────────┐
    ▼                     ▼                          ▼
┌────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  視覺層    │    │   音訊層     │    │     生理信號層       │
│  (Vision)  │    │   (Audio)    │    │  (Physiological)    │
│            │    │              │    │                     │
│ 人臉影片   │    │ 咳嗽聲音     │    │ 4 通道時間序列       │
│ (UBFC)     │    │ (COUGHVID)   │    │ (1250 步長)          │
│            │    │              │    │ (BIDMC PPG+ECG)      │
│ Swin-Tiny  │    │ AST (自訂)   │    │ iTransformer         │
│ 768 維特徵 │    │ 128 維 CLS   │    │ 128 維池化           │
└─────┬──────┘    └──────┬───────┘    └──────────┬──────────┘
      │ 特徵向量          │ 特徵向量             │ 特徵向量
      └──────────────────┼─────────────────────┘
                         ▼
             ┌──────────────────────────┐
             │        融合層            │
             │     (Fusion Layer)       │
             │                          │
             │ MultimodalFusionEncoder  │
             │ 4-token Transformer      │
             │ (CLS + 視覺 + 音訊       │
             │  + 生理信號)              │
             │                          │
             │ d_model=256, 4 層        │
             │ 8 注意力頭, GELU         │
             │                          │
             │ 輸出: 256 維 CLS 嵌入    │
             │    + 3 分類預測           │
             └────────────┬─────────────┘
                          │ predictions.csv (92 筆樣本)
                          ▼
┌══════════════════════════════════════════════════════════════════┐
║              AI 代理層  (FastAPI :8000)                          ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │          單一 AI 代理 (v1 — 始終運行)                      │ ║
║  │                                                            │ ║
║  │  POST /api/v1/tick  { prediction, subject_id,             │ ║
║  │                        feature_vector (256 維) }           │ ║
║  │       │                                                    │ ║
║  │       ├─ 從融合嵌入向量計算生命徵象代理值                    │ ║
║  │       ├─ TrendAnalyzer：滾動緩衝 (20 筆)                  │ ║
║  │       │   → "惡化" / "改善" / "穩定"                       │ ║
║  │       │   → HR/SpO₂/RR 斜率 (numpy.polyfit)               │ ║
║  │       ├─ DecisionEngine：11 條優先級規則                  │ ║
║  │       │   → 匹配規則 + 嚴重等級 + 可能病症                  │ ║
║  │       └─ AdviceGenerator：結構化建議                       │ ║
║  │                                                            │ ║
║  │  持久化至 PostgreSQL (4 張表)                              │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                           │                                      ║
║                           ▼                                      ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │           多 AI 代理擴展 (v2 — 疊加功能)                    │ ║
║  │                                                            │ ║
║  │  ┌──────────────────┐  ┌────────────────────────────────┐  │ ║
║  │  │   MCP 伺服器     │  │        3 項技能                │  │ ║
║  │  │                  │  │                                │  │ ║
║  │  │ Memory：LRU + PG │  │ 🔴 異常檢測器                  │  │ ║
║  │  │ Control：Fan-out │  │    滾動 z-score (閾值 2.5)     │  │ ║
║  │  │   + Fan-in       │  │    + 持續性異常偵測            │  │ ║
║  │  │ Planning：DAG    │  │                                │  │ ║
║  │  │   工作流          │  │ 📈 進階趨勢分析器              │  │ ║
║  │  └──────────────────┘  │    4 個窗口 (5/10/30/60)       │  │ ║
║  │                        │    + 線性+指數平滑預測          │  │ ║
║  │  ┌──────────────────┐  │    (向前 5 步)                 │  │ ║
║  │  │   協調器         │  │                                │  │ ║
║  │  │   (Coordinator)  │  │ 🤖 LLM 建議生成器              │  │ ║
║  │  │                  │  │    OpenAI / Claude / 本地      │  │ ║
║  │  │ 並行執行 v1 代理 │  │    臨床推理提示詞              │  │ ║
║  │  │ + 3 項技能 +     │  │    保留結構化欄位              │  │ ║
║  │  │ 外部代理          │  │                                │  │ ║
║  │  └──────────────────┘  └────────────────────────────────┘  │ ║
║  │                                                            │ ║
║  │  21 個 REST 端點 · PostgreSQL：共 9 張表                  │ ║
║  │  92 個測試 · 向後兼容 v1                                   │ ║
║  └────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════╝
                          │  HTTP REST + Socket.IO
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│               儀表板與警報層 (Dashboard & Alert)                  │
│                                                                  │
│  Flask 後端 (:5000)                  React 前端 (:3000)          │
│  ┌────────────────────────┐         ┌─────────────────────────┐ │
│  │ HealthSimulator        │         │ 🩺 健康狀態儀表          │ │
│  │ (重播 predictions.csv  │ Socket  │ 📈 生理趨勢圖            │ │
│  │  每 2 秒一筆)           │◄───────►│ 🫁 咳嗽波形圖            │ │
│  │                        │         │ 🧠 AI 代理建議面板       │ │
│  │ AlertManager           │         │ ⚠️ 警報狀態面板          │ │
│  │ (LED·蜂鳴器·Telegram)  │         │ 🔬 特徵視覺化 (PCA)      │ │
│  └────────────────────────┘         └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 第一層：模態模型 + 融合

三個獨立的 AI 模型，各自使用不同的公開資料集訓練。不同資料集之間**沒有共享的受試者**——融合層使用標籤匹配配對來創建訓練三元組。

### 視覺層 — Swin-Tiny 用於 rPPG

| 屬性 | 值 |
|----------|-------|
| **模型** | Swin-Tiny (2800 萬參數, `microsoft/swin-tiny-patch4-window7-224`) |
| **輸入** | RGB 人臉影片幀, 224×224 |
| **檢測內容** | 遠程光電容積描記 (rPPG) — 從臉部顏色變化推算心率 |
| **資料集** | UBFC rPPG (42 位受試者, ~607 筆樣本) |
| **輸出** | 768 維特徵向量 + 3 分類 logits |

### 音訊層 — 音訊頻譜 Transformer (AST)

| 屬性 | 值 |
|----------|-------|
| **模型** | 自訂輕量 AST (3 層 Transformer, d_model=128) |
| **輸入** | Mel 頻譜圖 (128 頻帶 × 192 時間幀) |
| **檢測內容** | 咳嗽聲音模式、呼吸音訊特徵 |
| **資料集** | COUGHVID (~2,800 筆樣本) |
| **輸出** | 128 維 CLS 嵌入 + 3 分類 logits |

### 生理信號層 — iTransformer

| 屬性 | 值 |
|----------|-------|
| **模型** | iTransformerClassifier（跨通道注意力，而非跨時間步） |
| **輸入** | 4 通道時間序列 (1250 步): PPG, ECG, HR, SpO₂ 衍生信號 |
| **建模內容** | 心血管與呼吸模式 |
| **資料集** | BIDMC PPG & Respiration (53 位受試者) |
| **輸出** | 128 維池化嵌入 + 3 分類 logits |

### 融合層 — MultimodalFusionEncoder

| 屬性 | 值 |
|----------|-------|
| **輸入** | 拼接特徵: 視覺(768) + 音訊(128) + 生理(128) = **1024 維** |
| **架構** | 4-token 序列: [CLS, 視覺投影, 音訊投影, 生理投影], 每個 256 維（經 Linear 投影後） |
| **Transformer** | 4 層, 8 注意力頭, d_ff=512, GELU, pre-norm |
| **輸出** | **256 維 CLS 嵌入** + 3 分類預測 |
| **訓練** | 5 組實驗, Focal Loss (γ=2.0), 最佳: Exp 2 (77.2% 準確率, 0.775 加權 F1) |
| **輸出檔案** | `predictions.csv` — 92 筆樣本，每筆含 256 維特徵向量 |

---

## 第二層：單一 AI 代理 (v1)

### 為何需要此層

融合層僅輸出一個原始數字 (0/1/2)，臨床上無法直接使用。AI 代理回答以下問題：

- **趨勢**：病人正在惡化、改善，還是維持穩定？
- **診斷**：符合 11 種臨床模式中的哪一種？
- **行動**：臨床醫師現在該做什麼？
- **歷史**：5、10、30 分鐘前的狀態是什麼？

### 運作原理 — `process_tick()` 管線

```
POST /api/v1/tick
{
  "prediction": 2,
  "subject_id": "subject14",
  "feature_vector": [0.12, -0.45, 0.78, ...]  // 256 維融合嵌入
}

                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 步驟 1：生命徵象代理值計算                                │
│   將 256 維向量均分為三等份：                             │
│   HR  = 75 + mean(前1/3)  × 10  →  95 bpm              │
│   SpO₂ = 97 − |mean(中1/3)| × 5  →  93%               │
│   RR  = 0.85 + mean(後1/3) × 0.2 → 0.72 s             │
├─────────────────────────────────────────────────────────┤
│ 步驟 2：TrendAnalyzer（趨勢分析）                        │
│   滾動 deque（最多 20 筆）                                │
│   unhealthy_ratio ≥ 0.3 → "惡化"                        │
│   healthy_ratio ≥ 0.7   → "改善"                        │
│   否則                  → "穩定"                        │
│   numpy.polyfit 斜率：HR=+5.2, SpO₂=-3.1, RR=-0.04     │
├─────────────────────────────────────────────────────────┤
│ 步驟 3：DecisionEngine（11 條規則，優先級匹配）          │
│   匹配規則：rule_002 "嚴重呼吸窘迫"                      │
│   條件：prediction=2 + 惡化 + SpO₂↓≥3%                   │
│   嚴重等級：高                                           │
├─────────────────────────────────────────────────────────┤
│ 步驟 4：AdviceGenerator（建議生成）                      │
│   可能病症："可能的呼吸道感染 / 肺炎"                     │
│   建議："建議立即進行呼吸評估..."                         │
│   行動：[通知醫師, 檢查血氧, 呼吸評估]                    │
├─────────────────────────────────────────────────────────┤
│ 步驟 5：去重                                            │
│   與上次規則相同？→ 回傳 null（未變更）                   │
├─────────────────────────────────────────────────────────┤
│ 步驟 6：PostgreSQL 持久化                                │
│   → observations, advice_log, trend_snapshots           │
└─────────────────────────────────────────────────────────┘
```

### API 回應範例（儀表板接收的資料）

```json
{
  "matched_rule_id": "rule_002",
  "matched_rule_name": "severe_respiratory_distress",
  "severity": "high",
  "possible_condition": "可能的呼吸道感染 / 肺炎",
  "advice": "建議立即進行呼吸評估。不健康分類伴隨血氧飽和度下降（≥3%）及惡化趨勢，可能為肺炎、支氣管炎或 COVID-19。請立即使用脈搏血氧儀檢查 SpO₂。若 SpO₂ 低於 92% 請尋求緊急醫療。",
  "actions": ["通知醫師", "檢查血氧", "呼吸評估"],
  "context": {
    "trend": "degrading",
    "unhealthy_ratio": 0.45,
    "healthy_ratio": 0.10,
    "hr_slope": 5.2,
    "spo2_slope": -3.1,
    "rr_slope": -0.04
  },
  "timestamp": "2026-06-24T00:16:32.154Z"
}
```

---

## 第三層：多 AI 代理 (v2)

### v2 在 v1 基礎上新增了什麼

單一代理在規則化建議方面表現良好，但存在以下缺口：

| v1 的缺口 | v2 的解決方案 |
|---------|-------------|
| 無法偵測突然的生命徵象飆升 | **異常檢測器**：滾動窗口 z-score，標記臨界異常值 |
| 只有一個趨勢窗口 (10 筆) | **進階趨勢分析器**：4 個窗口 (5/10/30/60) + 5 步預測 |
| 建議文字為固定模板 | **LLM 建議生成器**：以 GPT-4/Claude 臨床推理增強 |
| 單體架構——無法新增代理 | **MCP 伺服器**：透過 API 註冊外部代理，並行 Fan-out |
| 無跨代理協調 | **代理協調器**：運行所有組件，聚合結果 |

### MCP 伺服器（記憶 · 控制 · 規劃）

| 組件 | 功能 | 範例 |
|-----------|-------------|---------|
| **Memory Store** | 共享鍵值存儲，支援 TTL + PostgreSQL 備份 | `"patient_14:last_critical_hr" = 142`（3600 秒後過期） |
| **Controller** | 代理註冊表、並行 Fan-out 請求至多個代理、Fan-in 聚合（多數決/平均/全部） | 同時發送 tick 至 HealthAgent + AnomalyDetector + TrendAnalyzer |
| **Planner** | 將目標分解為子任務 DAG、拓撲排序、工作流 Session | "監測健康並偵測異常" → 3 個子任務 → 按依賴順序執行 |

### 三項技能

**1. 異常檢測器**——捕捉規則引擎遺漏的異常：
- 每個指標的滾動 z-score（HR, SpO₂, RR, prediction），窗口=30
- |z| > 2.5 → 警告，|z| > 3.5 → 危急
- 持續性偵測：當歷史為健康時，連續 3+ 次不健康預測
- 範例：*"HR z-score = +3.8（危急）。觀測值 142 bpm vs 預期值 82 bpm。"*

**2. 進階趨勢分析器**——多尺度視角：
- 4 個同步滾動窗口：5, 10, 30, 60 筆觀測
- 每個尺度的趨勢分類
- 線性回歸 + 指數平滑預測（向前 5 步）
- 跨尺度洞察：*"短期惡化 vs 長期改善——可能為暫時性。"*

**3. LLM 建議生成器**——臨床推理層（可選，需主動開啟）：
- 系統提示詞將 LLM 約束為臨床決策支援助理
- 保留結構化欄位（嚴重等級、行動、規則 ID）
- 僅增強 `advice` 文字欄位
- 支援：OpenAI (gpt-4o)、Anthropic Claude、本地 (Ollama)
- 未配置時：原樣輸出模板建議（零成本）

### 代理協調器 (Agent Coordinator)

每次 tick 執行此管線：

```
process_tick_multi():
  1. v1 HealthAgent.process_tick()         → 單一代理建議（始終執行）
  2. AnomalyDetector.update()              → 異常事件列表
  3. AdvancedTrendAnalyzer.update()        → 多尺度趨勢 + 預測
  4. LLMAdviceGenerator.enrich()           → 增強建議文字（若已配置）
  5. MCP Controller.fan_out()              → 分派至外部代理 (HTTP)
  6. MCP Controller.fan_in()               → 聚合：共識嚴重等級
  7. 持久化至 DB                            → anomaly_events + skill_executions
```

---

## 第四層：儀表板

### 資料如何到達儀表板

**HealthSimulator** 是一個背景執行緒，以 2 秒間隔重播 `predictions.csv`（92 筆融合樣本）。它**不使用真實感測器**——這是用於展示的模擬重播。每次 tick：

1. 從 predictions.csv 讀取一行
2. 通過 `POST /api/v1/tick` 發送至 AI 代理
3. 接收 AdviceResponse
4. 通過 Socket.IO 向發送 `agent_advice` 事件
5. 同時發送包含原始預測資料的 `health_update`

### 儀表板顯示內容

| 組件 | 顯示內容 | 資料來源 |
|-----------|-------------|-------------|
| **健康狀態儀表** | 環形圖：健康 / 亞健康 / 不健康 計數 | REST `/api/health_state` |
| **AI 代理建議面板** | 嚴重等級徽章 (🔴🟡🟢)、病症名稱、建議文字、行動標籤、趨勢指標、可展開的上下文（生命徵象斜率、比例） | REST `/api/agent_advice` + Socket.IO `agent_advice` |
| **生理趨勢圖** | 多線圖：HR, SpO₂, RR interval 隨時間變化 | REST `/api/physio_trend` |
| **咳嗽波形圖** | 呼吸模式視覺化 | REST `/api/cough_curve` |
| **疾病分類面板** | 混淆矩陣、各類別 precision/recall/F1、準確率 | REST `/api/disease_classification` |
| **特徵視覺化** | 256 維融合嵌入的 PCA/t-SNE 散點圖 | REST `/api/feature_viz` |
| **警報狀態** | 警報記錄、LED（紅燈閃爍）、蜂鳴器（嗶聲）、不健康時 Telegram 通知 | Socket.IO `alert_triggered` |
| **實驗選擇器** | 切換 5 個已訓練的融合實驗 | REST `/api/experiments` |

---

## 端到端：一筆預測的完整旅程

```
1. 融合層產生 predictions.csv 的一行：
   filename: "v:UBFC2/subject14/...|a:COUGHVID/uuid|p:bidmc19_..."
   prediction: 2 (不健康)
   label: 2 (真實標籤)
   feature_vector: "[0.12, -0.45, 0.78, ...]"  (256 個浮點數，JSON 字串)

2. 健康模擬器讀取該行，發送至 AI 代理：
   POST http://localhost:8000/api/v1/tick
   { prediction: 2, subject_id: "subject14", feature_vector: [0.12, -0.45, ...] }

3. AI 代理處理：
   a. 從嵌入向量推算生命徵象：HR=95, SpO₂=93%, RR=0.72s
   b. 趨勢："惡化"（HR↑, SpO₂↓）
   c. 匹配規則：rule_002 → 高嚴重等級 → "可能為肺炎"
   d. 異常檢測器：HR z-score +3.8 → 危急警報
   e. 趨勢分析器：短期惡化，長期穩定
   f. LLM（若啟用）：以臨床上下文增強建議
   回傳：AdviceResponse JSON

4. Flask 後端接收回應 → Socket.IO "agent_advice" 事件

5. React 前端渲染 AgentSuggestionsPanel：
   🔴 高 — 規則：severe_respiratory_distress
   可能的呼吸道感染 / 肺炎
   "建議立即進行呼吸評估..."
   [通知醫師] [檢查血氧] [呼吸評估]
   ▼ 上下文：趨勢=惡化, HR↑5.2, SpO₂↓3.1
```

---

## 單一代理 vs 多代理

| | 單一代理 (v1) | 多代理 (v2) |
|---|---|---|
| **核心功能** | 基於規則的健康建議 | 同上 + 異常檢測 + 多尺度趨勢 + LLM 增強 |
| **趨勢窗口** | 1 個 (10 筆) | 4 個 (5/10/30/60) + 5 步預測 |
| **異常檢測** | ❌ | ✅ z-score + 持續性警報 |
| **建議來源** | 固定模板 | 模板 + 可選 AI 增強臨床推理 |
| **外部代理** | ❌ | ✅ 通過 MCP 註冊，HTTP Fan-out |
| **工作流規劃** | ❌ | ✅ 任務分解 + 依賴 DAG |
| **API 端點** | 11 | 21（保留所有 v1 + 新增 10 個） |
| **資料庫表** | 4 | 9（保留所有 v1 + 新增 5 張） |
| **測試** | 83 | 92 |
| **儀表板相容性** | ✅ | ✅（向後兼容——相同 /tick 回應格式） |

**多代理包裝了單一代理。** 它完成 v1 的所有功能，並在此基礎上增加更多。儀表板呼叫相同的 `/api/v1/tick` 並獲得相同的回應格式——無需修改儀表板。

---

## 如何運行（3 個終端機）

```bash
# ═══════════════════════════════════════════════════════════════════
# 終端機 1 — AI 代理後端 (FastAPI :8000)
#   文檔：http://localhost:8000/docs
#   健康檢查：http://localhost:8000/api/v1/health
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer
python run.py

# ═══════════════════════════════════════════════════════════════════
# 終端機 2 — 儀表板後端 (Flask + SocketIO :5000)
#   API：http://localhost:5000/api/health_state
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer\"intel multimodal (AI_Agent_Single_layer)"\"intel multimodal (dashboard_and_alert_layer)"\dashboard_and_alert_layer
$env:AGENT_API_URL="http://localhost:8000/api/v1"
python run.py --no-agent

# ═══════════════════════════════════════════════════════════════════
# 終端機 3 — 儀表板前端 (React :3000)
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer\"intel multimodal (AI_Agent_Single_layer)"\"intel multimodal (dashboard_and_alert_layer)"\dashboard_and_alert_layer\dashboard\frontend
npm install    # 首次運行
npm start
```

開啟 **http://localhost:3000** ——儀表板顯示來自模擬器的即時健康資料串流，側邊欄顯示 AI 代理建議。

### 快速測試

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"2.0.0","db_connected":false}

curl -X POST http://localhost:8000/api/v1/tick \
  -H "Content-Type: application/json" \
  -d '{"prediction":2,"subject_id":"demo","feature_vector":[0.12,-0.45,0.78]}'
# → 結構化建議 JSON

curl http://localhost:8000/api/v1/multi/agents
# → ["health_agent","anomaly_detector","advanced_trend_analyzer","llm_advice_generator"]

curl http://localhost:8000/api/v1/mcp/status
# → {"memory_entries":0,"control_active":true,"planning_queue_size":0,...}
```

---

## API 參考（21 個端點）

### 單一代理 — `/api/v1`

| Method | Path | 用途 |
|--------|------|---------|
| POST | /tick | 提交健康觀測 → 獲取建議 |
| POST | /reset | 清除所有代理狀態 |
| GET | /advice/current | 最新建議 |
| GET | /advice/history?n=20 | 近期建議記錄 |
| GET | /trends/current | 當前趨勢摘要 |
| GET | /trends/history?window=100 | 歷史趨勢快照 |
| GET | /rules | 列出所有 11 條決策規則 |
| POST | /rules | 新增自訂規則 |
| DELETE | /rules/{rule_id} | 刪除規則 |
| GET | /status | 代理心跳 |
| GET | /health | 健康檢查 + 資料庫連線 |

### 多代理 — `/api/v1/multi`

| Method | Path | 用途 |
|--------|------|---------|
| GET | /multi/advice | 所有代理的聚合建議 |
| GET | /multi/trends | 多尺度趨勢 + 預測 |
| GET | /multi/anomalies?n=20 | 近期異常事件 |
| POST | /multi/skills | 按需執行技能 |
| GET | /multi/agents | 已註冊代理目錄 |

### MCP 伺服器 — `/api/v1/mcp`

| Method | Path | 用途 |
|--------|------|---------|
| GET | /mcp/status | MCP 伺服器狀態 |
| POST | /mcp/agents | 註冊外部代理 |
| DELETE | /mcp/agents/{agent_id} | 註銷代理 |
| POST | /mcp/workflow | 啟動規劃的工作流 |
| GET | /mcp/workflow/{session_id} | 檢查工作流進度 |

---

## 測試

```bash
cd Multi_AI_Agent_layer
pytest tests/ -v
# 92 passed — 使用 SQLite（無需 PostgreSQL）
```

---

## 團隊

| 成員 | 角色 |
|--------|------|
| **Justin** | 硬體整合、DK-2500 邊緣部署、感測器 |
| **Sunny** | 視覺/音訊模型、融合 Transformer、儀表板前端、Multi AI Agent Layer |
| **Baileys** | 生理信號建模、資料處理、邊緣部署 |

## GitHub 倉庫

| Layer | GitHub |
|-------|--------|
| Single AI Agent (v1) | [github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer) |
| Multi AI Agent (v2) | [github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Multi-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Multi-AI-Agent-Layer) |
