# 历史 AQHI 原始数据下载

## 内容与来源

- 数据集：香港环保署「过去空气质素健康指数记录（简体中文）」。
- 官方目录：https://data.gov.hk/sc-data/dataset/hk-epd-airteam-past-record-of-air-quality-health-index-sc
- 本次目录列出范围：2013 年 12 月至 2026 年 6 月，共 151 个月份 CSV。
- 官方说明记录始于 2013 年 12 月 30 日，因此首月不是完整月份。
- `raw/aqhi_history/YYYY/YYYYMM_ChS.csv`：未经修改的官方月度原始文件，按年份整理。
- `download_records/时间戳/download_manifest.csv`：每个文件的下载状态、来源链接、大小、SHA-256 校验值。
- 同目录 `manifest.json`：下载汇总；`official_catalog.html`：下载时的官方目录快照；如成功获取，`hk_aqhisc_sc.pdf` 为官方数据字典。

## 使用注意

1. 最终下载数量与成功情况以 manifest.json 为准；目录列出不代表所有下载都成功。
2. 旧样本 `raw/202606_ChS.csv` 保留不动。批量读取时只扫描 `raw/aqhi_history`，避免六月重复导入。
3. 原有 `processed` 文件和 `01_data_check.ipynb` 仍对应单月样本；本次下载不会自动更新它们。
4. 这些 CSV 尚未完成跨月份清洗或合并。必须区分小时记录、每日摘要、缺失符号与替代标记，并检查历年站点变化。
5. 下载所有历史文件不代表最终模型必须使用全部年份。训练、验证、测试范围需要在质量检查后按时间确定。
6. 本次新增的是历史 AQHI 数据，不是污染物浓度、气象或户外场地数据；原有 RSS 快照也没有刷新。
