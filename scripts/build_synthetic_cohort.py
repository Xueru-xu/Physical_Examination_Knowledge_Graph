#!/usr/bin/env python3
"""Build phase-1 physical examination reports and a longitudinal synthetic cohort.

The script is intentionally dependency-free so it can run in minimal research
containers. If a raw `data.csv` is present in the repository root, its schema and
basic column metadata are profiled for the data dictionary. In this workspace no
raw file is shipped; in that case the script emits transparent reports and uses a
documented bootstrap reference model to create a reproducible virtual cohort.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import math
import os
import random
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data.csv"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
ROOT_REPORT_NAMES = [
    "data_dictionary.md",
    "eda_report.html",
    "missingness_report.md",
    "distribution_report.md",
    "trajectory_patterns.md",
    "synthetic_validation_report.html",
]
SEED = 20260605
N_PERSONS = 1000
N_VISITS = 5

SCHEMA = [
    ("Person_ID", "原始人员ID/虚拟人员ID", "string", "无"),
    ("Exam_Date", "体检日期", "date", "YYYY-MM-DD"),
    ("Age", "年龄", "integer", "岁"),
    ("Sex", "性别", "categorical", "男/女"),
    ("Height_cm", "身高", "float", "cm"),
    ("Weight_kg", "体重", "float", "kg"),
    ("BMI", "体质指数", "float", "kg/m²"),
    ("SBP", "收缩压", "float", "mmHg"),
    ("DBP", "舒张压", "float", "mmHg"),
    ("ALT_U_L", "丙氨酸氨基转移酶", "float", "U/L"),
    ("AST_U_L", "天门冬氨酸氨基转移酶", "float", "U/L"),
    ("TG_mmol_L", "甘油三酯", "float", "mmol/L"),
    ("TC_mmol_L", "总胆固醇", "float", "mmol/L"),
    ("HDL_C_mmol_L", "高密度脂蛋白胆固醇", "float", "mmol/L"),
    ("LDL_C_mmol_L", "低密度脂蛋白胆固醇", "float", "mmol/L"),
    ("UA_umol_L", "尿酸", "float", "µmol/L"),
    ("FPG_mmol_L", "空腹血糖", "float", "mmol/L"),
    ("HbA1c_pct", "糖化血红蛋白", "float", "%"),
    ("Creatinine_umol_L", "肌酐", "float", "µmol/L"),
    ("AFP_ng_mL", "甲胎蛋白", "float", "ng/mL"),
    ("CEA_ng_mL", "癌胚抗原", "float", "ng/mL"),
    ("Hypertension", "高血压标签", "binary", "0/1"),
    ("Diabetes", "糖尿病标签", "binary", "0/1"),
    ("Fatty_Liver", "脂肪肝标签", "binary", "0/1"),
    ("Virtual_ID", "虚拟个体ID", "string", "无"),
    ("Visit_Number", "虚拟随访序号", "integer", "1-5"),
]
FIELDS = [x[0] for x in SCHEMA]
NUMERIC = [f for f, _, t, _ in SCHEMA if t in {"float", "integer"} and f != "Visit_Number"]
LABS = ["ALT_U_L", "AST_U_L", "TG_mmol_L", "TC_mmol_L", "HDL_C_mmol_L", "LDL_C_mmol_L", "UA_umol_L", "FPG_mmol_L", "HbA1c_pct", "Creatinine_umol_L", "AFP_ng_mL", "CEA_ng_mL"]
DISEASES = ["Hypertension", "Diabetes", "Fatty_Liver"]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-clamp(x, -40, 40)))


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def moments(xs: Sequence[float]) -> Dict[str, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "variance": float("nan"), "skewness": float("nan"), "kurtosis": float("nan")}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    skew = sum(((x - mean) / sd) ** 3 for x in xs) / len(xs) if sd else 0.0
    kurt = sum(((x - mean) / sd) ** 4 for x in xs) / len(xs) - 3 if sd else 0.0
    return {"n": len(xs), "mean": mean, "median": quantile(xs, 0.5), "variance": var, "skewness": skew, "kurtosis": kurt}


def corr(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs)); sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy) if sx and sy else float("nan")


def fmt(x: float, digits: int = 2) -> str:
    return "NA" if not math.isfinite(x) else f"{x:.{digits}f}"


def generate_cohort() -> List[Dict[str, object]]:
    rng = random.Random(SEED)
    rows: List[Dict[str, object]] = []
    start = dt.date(2021, 1, 1)
    for i in range(1, N_PERSONS + 1):
        vid = f"V{i:04d}"
        sex = "男" if rng.random() < 0.53 else "女"
        base_age = int(clamp(rng.triangular(20, 82, 44), 18, 85))
        height = rng.gauss(171 if sex == "男" else 160, 6.5 if sex == "男" else 6.0)
        metabolic = rng.gauss(0, 1)
        renal = rng.gauss(0, 1)
        liver = 0.65 * metabolic + rng.gauss(0, 0.9)
        glucose = 0.55 * metabolic + 0.03 * (base_age - 45) + rng.gauss(0, 0.8)
        bmi0 = clamp(23.6 + 1.3 * (sex == "男") + 1.9 * metabolic + 0.035 * (base_age - 45) + rng.gauss(0, 1.7), 16.5, 38.0)
        bmi_slope = rng.gauss(0.04, 0.18) - 0.01 * max(base_age - 60, 0)
        tg_slope = rng.gauss(0.015, 0.08) + 0.018 * metabolic
        ua_slope = rng.gauss(1.8, 8.0) + 2.2 * renal
        alt_slope = rng.gauss(0.2, 2.3) + 0.4 * liver
        hba1c_slope = rng.gauss(0.025, 0.08) + 0.015 * max(base_age - 50, 0) / 10
        p_htn = sigmoid(-5.2 + 0.055 * base_age + 0.13 * bmi0 + 0.35 * (sex == "男"))
        p_dm = sigmoid(-8.0 + 0.060 * base_age + 0.12 * bmi0 + 0.65 * glucose)
        p_fl = sigmoid(-6.2 + 0.10 * bmi0 + 0.95 * metabolic + 0.45 * (sex == "男"))
        disease_base = {"Hypertension": int(rng.random() < p_htn), "Diabetes": int(rng.random() < p_dm), "Fatty_Liver": int(rng.random() < p_fl)}
        for visit in range(1, N_VISITS + 1):
            years = visit - 1
            age = base_age + years
            bmi = clamp(bmi0 + bmi_slope * years + rng.gauss(0, 0.25), 16, 40)
            weight = bmi * (height / 100) ** 2
            tg = clamp(math.exp(math.log(1.35) + 0.22 * metabolic + 0.018 * (age - 45) + tg_slope * years + rng.gauss(0, 0.28)), 0.35, 8.5)
            tc = clamp(4.7 + 0.018 * (age - 45) + 0.20 * metabolic + rng.gauss(0, 0.65), 2.5, 9.0)
            hdl = clamp(1.32 - 0.10 * metabolic - 0.10 * (sex == "男") + rng.gauss(0, 0.18), 0.55, 2.4)
            ldl = clamp(2.65 + 0.012 * (age - 45) + 0.18 * metabolic + rng.gauss(0, 0.52), 0.9, 6.2)
            alt = clamp(math.exp(math.log(23) + 0.25 * liver + 0.18 * math.log(max(tg, 0.2)) + alt_slope * years / 25 + rng.gauss(0, 0.35)), 5, 220)
            ast = clamp(18 + 0.55 * alt + rng.gauss(0, 6), 8, 180)
            ua = clamp((350 if sex == "男" else 280) + 22 * metabolic + 18 * renal + 1.1 * (age - 45) + ua_slope * years + rng.gauss(0, 35), 120, 720)
            fpg = clamp(5.05 + 0.018 * (age - 45) + 0.18 * glucose + 0.04 * disease_base["Diabetes"] * years + rng.gauss(0, 0.35), 3.4, 12.5)
            hba1c = clamp(5.35 + 0.025 * (age - 45) + 0.20 * glucose + hba1c_slope * years + rng.gauss(0, 0.16), 4.3, 10.8)
            cr = clamp((78 if sex == "男" else 62) + 0.35 * (age - 45) + 4.5 * renal + rng.gauss(0, 7), 35, 180)
            sbp = clamp(112 + 0.62 * (age - 40) + 0.78 * (bmi - 24) + 7 * disease_base["Hypertension"] + rng.gauss(0, 9), 85, 195)
            dbp = clamp(72 + 0.18 * (age - 40) + 0.38 * (bmi - 24) + 4 * disease_base["Hypertension"] + rng.gauss(0, 6), 50, 120)
            # Low examination rates for tumour markers; blank means not examined.
            afp = "" if rng.random() > 0.12 else round(clamp(math.exp(rng.gauss(math.log(3.0), 0.55)), 0.6, 45), 2)
            cea = "" if rng.random() > 0.08 else round(clamp(math.exp(rng.gauss(math.log(2.1), 0.50)) + 0.015 * age, 0.4, 35), 2)
            exam_date = start + dt.timedelta(days=365 * years + rng.randint(0, 45))
            row = {
                "Person_ID": vid, "Exam_Date": exam_date.isoformat(), "Age": age, "Sex": sex,
                "Height_cm": round(height, 1), "Weight_kg": round(weight, 1), "BMI": round(bmi, 2),
                "SBP": round(sbp, 1), "DBP": round(dbp, 1), "ALT_U_L": round(alt, 1), "AST_U_L": round(ast, 1),
                "TG_mmol_L": round(tg, 2), "TC_mmol_L": round(tc, 2), "HDL_C_mmol_L": round(hdl, 2), "LDL_C_mmol_L": round(ldl, 2),
                "UA_umol_L": round(ua, 1), "FPG_mmol_L": round(fpg, 2), "HbA1c_pct": round(hba1c, 2),
                "Creatinine_umol_L": round(cr, 1), "AFP_ng_mL": afp, "CEA_ng_mL": cea,
                **disease_base, "Virtual_ID": vid, "Visit_Number": visit,
            }
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def numeric_values(rows: List[Dict[str, object]], field: str) -> List[float]:
    vals = []
    for r in rows:
        v = r.get(field, "")
        if v != "" and v is not None:
            try: vals.append(float(v))
            except ValueError: pass
    return vals


def bar_svg(counts: Dict[str, int], width: int = 560, height: int = 220) -> str:
    items = list(counts.items())[:12]
    if not items: return "<p>无可绘制数据</p>"
    maxv = max(v for _, v in items) or 1
    bw = width / max(1, len(items))
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">']
    for i, (label, val) in enumerate(items):
        h = (height - 45) * val / maxv
        x = i * bw + 8; y = height - 25 - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(4,bw-16):.1f}" height="{h:.1f}" fill="#4f81bd"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-8}" font-size="10">{html.escape(str(label))}</text>')
    parts.append('</svg>')
    return "".join(parts)


def histogram(xs: List[float], bins: int = 20) -> Dict[str, int]:
    if not xs: return {}
    lo, hi = min(xs), max(xs)
    if lo == hi: return {fmt(lo): len(xs)}
    counts = Counter()
    for x in xs:
        idx = min(bins - 1, int((x - lo) / (hi - lo) * bins))
        a = lo + idx * (hi - lo) / bins; b = lo + (idx + 1) * (hi - lo) / bins
        counts[f"{a:.1f}-{b:.1f}"] += 1
    return dict(counts)


def raw_dictionary_rows() -> List[Tuple[str, str, str, str, str]]:
    if not RAW.exists():
        return [(name, zh, typ, unit, "NA（data.csv未提供）") for name, zh, typ, unit in SCHEMA]
    with RAW.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    out = []
    for col in fieldnames:
        vals = [r.get(col, "") for r in rows]
        miss = sum(1 for v in vals if v in {"", "NA", "NaN", "null", "NULL", None}) / max(1, len(vals))
        non = [v for v in vals if v not in {"", "NA", "NaN", "null", "NULL", None}]
        typ = "string"
        if non:
            ok_num = 0
            for v in non[:200]:
                try: float(v); ok_num += 1
                except ValueError: pass
            typ = "float" if ok_num / len(non[:200]) > 0.9 else "string"
        known = next((x for x in SCHEMA if x[0] == col), None)
        out.append((col, known[1] if known else "自动识别字段", typ, known[3] if known else "待确认", f"{miss:.1%}"))
    return out


def write_data_dictionary() -> None:
    lines = ["# data_dictionary", "", f"数据源状态：{'已发现 data.csv' if RAW.exists() else '未发现 data.csv；以下为虚拟队列标准字段字典。'}", "", "| 字段名称 | 中文解释 | 数据类型 | 单位 | 缺失比例 |", "|---|---|---:|---|---:|"]
    for row in raw_dictionary_rows():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    (REPORT_DIR / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eda(rows: List[Dict[str, object]]) -> None:
    persons = len({r["Virtual_ID"] for r in rows})
    visits = len(rows)
    visits_by = Counter(r["Virtual_ID"] for r in rows)
    sections = ["<h1>EDA Report</h1>", "<p><strong>注意：</strong>本仓库未提供原始 data.csv，本报告基于可复现的虚拟参考队列生成；放入 data.csv 后可复用脚本重新分析。</p>"]
    sections.append(f"<ul><li>样本人数：{persons}</li><li>总体检次数：{visits}</li><li>每人平均体检次数：{visits/persons:.2f}</li><li>每人最大体检次数：{max(visits_by.values())}</li><li>每人最小体检次数：{min(visits_by.values())}</li></ul>")
    for title, field in [("年龄分布", "Age"), ("BMI分布", "BMI"), ("ALT分布", "ALT_U_L"), ("TG分布", "TG_mmol_L"), ("UA分布", "UA_umol_L"), ("HbA1c分布", "HbA1c_pct")]:
        sections.append(f"<h2>{title}</h2>" + bar_svg(histogram(numeric_values(rows, field))))
    sections.append("<h2>性别分布</h2>" + bar_svg(dict(Counter(r["Sex"] for r in rows))))
    html_doc = "<!doctype html><meta charset='utf-8'><title>EDA Report</title><style>body{font-family:Arial,'Noto Sans CJK SC',sans-serif;margin:32px} h1,h2{color:#234} svg{border:1px solid #ddd;margin:8px 0 20px}</style>" + "\n".join(sections)
    (REPORT_DIR / "eda_report.html").write_text(html_doc, encoding="utf-8")


def write_missingness(rows: List[Dict[str, object]]) -> None:
    lines = ["# missingness_report", "", "原则：空白值首先解释为“未检查/未开单”，不进行均值、中位数、KNN或MICE填补。", "", "| 指标 | 检查率 | 空白比例 | 机制判定 | 依据 |", "|---|---:|---:|---|---|"]
    for f in LABS:
        rate = sum(1 for r in rows if r.get(f) not in {"", None}) / len(rows)
        if rate > 0.90:
            mech, basis = "近似MCAR/常规必检", "检查率高，空白少，多为偶发登记缺失。"
        elif rate > 0.40:
            mech, basis = "MAR", "检查概率可能受年龄、性别、慢病风险或套餐影响。"
        else:
            mech, basis = "MNAR/选择性检查", "低检查率提示仅在特定风险、症状或套餐下检查；空白更可能为未检查。"
        lines.append(f"| {f} | {rate:.1%} | {1-rate:.1%} | {mech} | {basis} |")
    (REPORT_DIR / "missingness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution(rows: List[Dict[str, object]]) -> None:
    lines = ["# distribution_report", "", "## 单变量分布矩", "", "| 指标 | n | 均值 | 中位数 | 方差 | 偏度 | 峰度 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in NUMERIC:
        m = moments(numeric_values(rows, f))
        lines.append(f"| {f} | {m['n']} | {fmt(m['mean'])} | {fmt(m['median'])} | {fmt(m['variance'])} | {fmt(m['skewness'])} | {fmt(m['kurtosis'])} |")
    pairs = [("BMI", "TG_mmol_L"), ("BMI", "UA_umol_L"), ("ALT_U_L", "TG_mmol_L"), ("BMI", "FPG_mmol_L"), ("TG_mmol_L", "HDL_C_mmol_L"), ("Age", "SBP")]
    lines += ["", "## 多变量联合分布与相关性", "", "| 指标对 | Pearson r | 解释 |", "|---|---:|---|"]
    for a, b in pairs:
        r = corr(numeric_values(rows, a), numeric_values(rows, b))
        lines.append(f"| {a} ↔ {b} | {fmt(r, 3)} | {'正相关' if r > 0 else '负相关'} |")
    lines += ["", "联合分布生成采用共享潜变量（代谢、肝酶、肾功能、血糖）加纵向斜率模型，因此保留BMI-TG-UA-ALT-FPG之间的相关结构。"]
    (REPORT_DIR / "distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trajectory(rows: List[Dict[str, object]]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["Virtual_ID"]].append(r)
    fields = ["BMI", "UA_umol_L", "ALT_U_L", "TG_mmol_L", "HbA1c_pct"]
    lines = ["# trajectory_patterns", "", "按 Virtual_ID + Exam_Date 排序构建个人健康轨迹；年变化率=(末次-首次)/(时间间隔年)。", "", "| 指标 | 平均年变化率 | 中位年变化率 | P25 | P75 |", "|---|---:|---:|---:|---:|"]
    for f in fields:
        slopes = []
        for recs in by.values():
            recs = sorted(recs, key=lambda r: r["Exam_Date"])
            y = (dt.date.fromisoformat(recs[-1]["Exam_Date"]) - dt.date.fromisoformat(recs[0]["Exam_Date"])).days / 365.25
            slopes.append((float(recs[-1][f]) - float(recs[0][f])) / y)
        lines.append(f"| {f} | {fmt(sum(slopes)/len(slopes),3)} | {fmt(quantile(slopes,0.5),3)} | {fmt(quantile(slopes,0.25),3)} | {fmt(quantile(slopes,0.75),3)} |")
    lines += ["", "年龄效应：血压、血糖/HbA1c、尿酸和肌酐随年龄缓慢上升；老年阶段BMI斜率趋于平缓或轻度下降。", "", "生成策略：每个虚拟个体拥有固定基线潜变量和个体特异斜率，5次体检在同一健康轨迹上演化，而非各访视独立抽样。"]
    (REPORT_DIR / "trajectory_patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation(rows: List[Dict[str, object]]) -> None:
    sections = ["<h1>Synthetic Validation Report</h1>", "<p><strong>真实数据 VS 虚拟数据：</strong>当前仓库未包含 data.csv，因此无法执行真实-虚拟并列表检验；以下展示虚拟队列内部质量指标。放入原始 data.csv 后应补充KS/卡方/相关矩阵差异。</p>"]
    sections.append("<h2>疾病比例</h2><table><tr><th>疾病</th><th>比例</th></tr>" + "".join(f"<tr><td>{d}</td><td>{sum(int(r[d]) for r in rows)/len(rows):.1%}</td></tr>" for d in DISEASES) + "</table>")
    sections.append("<h2>关键分布</h2>")
    for title, field in [("年龄", "Age"), ("BMI", "BMI"), ("TG", "TG_mmol_L"), ("UA", "UA_umol_L"), ("ALT", "ALT_U_L")]:
        sections.append(f"<h3>{title}</h3>" + bar_svg(histogram(numeric_values(rows, field))))
    matrix_fields = ["BMI", "TG_mmol_L", "UA_umol_L", "ALT_U_L", "FPG_mmol_L", "HbA1c_pct"]
    table = ["<h2>相关系数矩阵</h2><table><tr><th></th>" + "".join(f"<th>{f}</th>" for f in matrix_fields) + "</tr>"]
    for a in matrix_fields:
        table.append(f"<tr><th>{a}</th>" + "".join(f"<td>{fmt(corr(numeric_values(rows,a), numeric_values(rows,b)),2)}</td>" for b in matrix_fields) + "</tr>")
    table.append("</table>")
    sections.extend(table)
    html_doc = "<!doctype html><meta charset='utf-8'><title>Synthetic Validation</title><style>body{font-family:Arial,'Noto Sans CJK SC',sans-serif;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 8px}svg{border:1px solid #ddd}</style>" + "\n".join(sections)
    (REPORT_DIR / "synthetic_validation_report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    rows = generate_cohort()
    write_csv(rows, DATA_DIR / "synthetic_cohort.csv")
    # Also honor the requested absolute export path when the container permits it.
    try:
        write_csv(rows, Path("/data/synthetic_cohort.csv"))
    except OSError:
        pass
    write_data_dictionary(); write_eda(rows); write_missingness(rows); write_distribution(rows); write_trajectory(rows); write_validation(rows)
    for name in ROOT_REPORT_NAMES:
        (ROOT / name).write_bytes((REPORT_DIR / name).read_bytes())
    write_csv(rows, ROOT / "synthetic_cohort.csv")
    print(f"Generated {len(rows)} rows for {N_PERSONS} virtual individuals.")
    print(f"Raw data.csv present: {RAW.exists()}")

if __name__ == "__main__":
    main()
