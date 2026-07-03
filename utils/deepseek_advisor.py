"""
DeepSeek 裂缝修补建议生成模块
读取裂缝检测 JSON 报告，调用 DeepSeek API 生成结构化维修建议
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime

import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def get_api_key():
    """从环境变量获取 DeepSeek API Key"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY，请先配置环境变量后再运行。\n"
            "  Linux/Mac: export DEEPSEEK_API_KEY=your_key_here\n"
            "  Windows:   set DEEPSEEK_API_KEY=your_key_here\n"
            "或在代码运行前通过 os.environ 设置。"
        )
    return key


def load_report(report_path):
    """加载裂缝检测 JSON 报告"""
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"报告文件不存在: {report_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_detection_context(report):
    """
    从报告中提取 DeepSeek 所需的检测上下文

    兼容两种报告格式：
      1. 单报告格式：直接包含 source_type / summary / cracks（video / image_sequence / 单图 image）
      2. 批量汇总格式：包含 per_image 列表（image 模式批量推理输出）
    """
    # 处理汇总报告格式（pipeline 生成的 summary_xxx.json，包含 all_cracks）
    if "all_cracks" in report:
        all_cracks = report.get("all_cracks", [])
        total_raw = report.get("total_raw_detections", 0)
        total_unique = report.get("total_unique_cracks", 0)
        confs = [c.get("confidence", 0) for c in all_cracks if c.get("confidence")]
        context = {
            "source_type": "image_sequence",
            "wall_id": report.get("wall_id"),
            "image_or_video_id": report.get("title", ""),
            "summary": {
                "raw_detection_count": total_raw,
                "unique_crack_count": total_unique,
                "duplicate_removed_count": total_raw - total_unique,
                "overall_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
            },
            "cracks": all_cracks,
            "limitations": [],
        }
        return context

    # 处理批量汇总格式（image 模式的 batch 输出）
    if "per_image" in report and isinstance(report.get("per_image"), list):
        per_image = report["per_image"]
        if not per_image:
            return _empty_context()
        # 汇总所有图像的统计
        total_raw = 0
        total_unique = 0
        all_cracks = []
        all_lims = set()
        source_type = per_image[0].get("source_type", "image")
        for img_report in per_image:
            s = img_report.get("summary", {})
            total_raw += s.get("raw_detection_count", 0)
            total_unique += s.get("unique_crack_count", 0)
            all_cracks.extend(img_report.get("cracks", []))
            for lim in img_report.get("limitations", []):
                all_lims.add(lim)
        summary = {
            "raw_detection_count": total_raw,
            "unique_crack_count": total_unique,
            "duplicate_removed_count": total_raw - total_unique,
            "overall_confidence": (
                round(sum(c.get("confidence", 0) for c in all_cracks) / len(all_cracks), 4)
                if all_cracks else 0
            ),
        }
        cracks = all_cracks
        limitations = list(all_lims)
        image_id = report.get("source", "")
        wall_id = None
    else:
        # 单报告格式
        source_type = report.get("source_type", "unknown")
        summary = report.get("summary", {})
        cracks = report.get("cracks", [])
        limitations = report.get("limitations", [])
        image_id = report.get("image_or_video_id", "")
        wall_id = report.get("wall_id")

    # 裂缝简况
    crack_list = []
    for c in cracks:
        crack_list.append({
            "crack_id": c.get("crack_id"),
            "track_id": c.get("track_id"),
            "frame_index": c.get("frame_index"),
            "confidence": c.get("confidence"),
            "area_px": c.get("area_px"),
            "length_px_est": c.get("length_px_est"),
            "orientation_angle": c.get("orientation_angle"),
            "is_duplicate": c.get("is_duplicate", False),
            "matched_with": c.get("matched_with"),
        })

    return {
        "source_type": source_type,
        "wall_id": wall_id,
        "image_or_video_id": image_id,
        "summary": {
            "raw_detection_count": summary.get("raw_detection_count", 0),
            "unique_crack_count": summary.get("unique_crack_count", 0),
            "duplicate_removed_count": summary.get("duplicate_removed_count", 0),
            "overall_confidence": summary.get("overall_confidence", 0),
        },
        "cracks": crack_list,
        "limitations": limitations,
    }


def _empty_context():
    """返回空的检测上下文"""
    return {
        "source_type": "unknown",
        "wall_id": None,
        "image_or_video_id": "",
        "summary": {
            "raw_detection_count": 0,
            "unique_crack_count": 0,
            "duplicate_removed_count": 0,
            "overall_confidence": 0,
        },
        "cracks": [],
        "limitations": [],
    }


def build_system_prompt():
    """构建 DeepSeek 系统角色提示"""
    return (
        "你是一名墙面裂缝检测与维修建议助手。"
        "你的任务是根据 yolov8n-seg-cracks-joints 模型的裂缝检测结果，给出专业、可操作的维修建议。\n\n"
        "请严格遵循以下规则：\n"
        "1. 不要虚构裂缝的真实宽度、真实长度或深度；如果输入只有像素长度和像素面积，"
        "必须明确说明「缺少尺度标定，无法得到真实毫米级尺寸」。\n"
        "2. 不要直接判断建筑结构是否安全；如果裂缝数量较多、长度较长或重复出现明显，"
        "应建议人工复核或专业检测机构介入。\n"
        "3. 建议应具体、可操作。\n"
        "4. 输出必须是合法的 JSON 格式，不要包含 markdown 代码块标记。\n"
        "5. JSON 必须使用以下英文字段名（不要使用中文键名）：\n\n"
        '{\n'
        '  "overall_assessment": "检测概况（一段话总结）",\n'
        '  "risk_level": "风险等级，取值为：轻微/中等/较严重/需人工复核",\n'
        '  "possible_causes": ["可能原因1", "可能原因2"],\n'
        '  "repair_plan": [\n'
        '    {"step": 1, "title": "步骤标题", "description": "详细描述"}\n'
        '  ],\n'
        '  "materials": ["建议材料1", "建议材料2"],\n'
        '  "need_manual_review": true或false,\n'
        '  "manual_review_reason": "需要人工复核的原因（如不需要则写无需）",\n'
        '  "limitations": ["当前检测的局限性1", "局限性2"]\n'
        '}\n'
    )


def build_user_prompt(context):
    """根据检测上下文构建用户提示"""
    source_type = context["source_type"]
    summary = context["summary"]
    cracks = context["cracks"]
    wall_id = context.get("wall_id")
    limitations = context.get("limitations", [])

    # 场景描述
    scene_desc = {
        "image": "单张图像检测",
        "video": f"视频连续帧跟踪检测（含 track_id）",
        "image_sequence": f"同一墙面多角度图像跨图去重检测",
    }.get(source_type, source_type)

    prompt_lines = [
        f"以下是一次墙面裂缝检测的结果汇总：",
        f"",
        f"## 检测场景",
        f"- 模式: {scene_desc}",
    ]
    if wall_id:
        prompt_lines.append(f"- 墙面标识 (wall_id): {wall_id}")
    prompt_lines.extend([
        f"- 图像/视频标识: {context['image_or_video_id']}",
        f"",
        f"## 检测统计",
        f"- 原始检测总数 (raw): {summary['raw_detection_count']}",
        f"- 去重后唯一裂缝数 (unique): {summary['unique_crack_count']}",
        f"- 去除的重复检测数: {summary['duplicate_removed_count']}",
        f"- 整体平均置信度: {summary['overall_confidence']:.2%}",
        f"",
        f"## 检测到的裂缝详情",
    ])

    if cracks:
        for c in cracks:
            dup_info = ""
            if c.get("is_duplicate"):
                matched = c.get("matched_with", [])
                if matched:
                    dup_info = f" (跨图重复，匹配到 {len(matched)} 张图像)"
            prompt_lines.append(
                f"  - {c['crack_id']}: 置信度={c['confidence']:.2%}, "
                f"像素面积={c['area_px']}px², 像素长度≈{c['length_px_est']:.0f}px, "
                f"方向角≈{c['orientation_angle']:.0f}°{dup_info}"
            )
    else:
        prompt_lines.append("  （未检测到裂缝）")

    prompt_lines.extend([
        f"",
        f"## 当前检测局限性",
    ])
    if limitations:
        for lim in limitations:
            prompt_lines.append(f"  - {lim}")
    else:
        prompt_lines.append("  （无）")

    prompt_lines.extend([
        f"",
        f"请根据以上信息，输出 JSON 格式的维修建议。",
        f"注意：像素长度和面积无法直接换算为真实物理尺寸，请勿虚构毫米级数据。",
    ])

    return "\n".join(prompt_lines)


def call_deepseek(api_key, system_prompt, user_prompt, temperature=0.3, max_tokens=4096):
    """调用 DeepSeek API 并返回响应文本"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API 返回错误 (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek 响应格式异常: {json.dumps(data, ensure_ascii=False)[:500]}")


def parse_advice_json(raw_text):
    """从 DeepSeek 响应中解析 JSON，带容错"""
    # 尝试直接解析
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试寻找第一个 { 到最后一个 }
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 解析失败，返回包含原始文本的兜底结构
    return {
        "overall_assessment": raw_text.strip(),
        "risk_level": "需人工复核",
        "parse_error": True,
        "raw_response": raw_text,
    }


def normalize_advice(parsed, source_report, context):
    """将 DeepSeek 输出标准化为统一 schema，填充缺失字段。支持中英文键名回退。"""
    summary = context.get("summary", {})

    # 回退映射：中文键名 → 英文字段
    cn_map = {
        "overall_assessment": ["检测概况", "overall_assessment"],
        "risk_level": ["风险等级", "risk_level"],
        "possible_causes": ["可能原因", "possible_causes"],
        "repair_plan": ["修补方案", "repair_plan", "施工步骤"],
        "materials": ["建议材料", "materials", "推荐材料"],
        "need_manual_review": ["need_manual_review", "需要人工复核"],
        "manual_review_reason": ["人工复核原因", "manual_review_reason", "注意事项"],
        "limitations": ["limitations", "局限性", "当前局限"],
    }

    def get_field(key, default):
        """优先取英文字段，找不到则回退中文别名"""
        if key in parsed and parsed[key]:
            return parsed[key]
        for alias in cn_map.get(key, []):
            if alias in parsed and parsed[alias]:
                return parsed[alias]
        return default

    # 修复 repair_plan：如果返回的是字符串列表而非对象列表，转换之
    repair = get_field("repair_plan", [])
    if repair and isinstance(repair, list) and len(repair) > 0:
        if isinstance(repair[0], str):
            repair = [
                {"step": i + 1, "title": item[:40], "description": item}
                for i, item in enumerate(repair)
            ]

    return {
        "source_report": source_report,
        "generated_at": datetime.now().isoformat(),
        "model_used": DEEPSEEK_MODEL,
        "overall_assessment": get_field("overall_assessment", ""),
        "risk_level": get_field("risk_level", "需人工复核"),
        "detected_summary": {
            "raw_detection_count": summary.get("raw_detection_count", 0),
            "unique_crack_count": summary.get("unique_crack_count", 0),
            "duplicate_removed_count": summary.get("duplicate_removed_count", 0),
        },
        "possible_causes": get_field("possible_causes", []),
        "repair_plan": repair,
        "materials": get_field("materials", []),
        "need_manual_review": get_field("need_manual_review", True),
        "manual_review_reason": get_field("manual_review_reason", ""),
        "limitations": get_field("limitations", [
            "像素长度和面积无法直接换算为真实物理尺寸（缺少尺度标定）",
            "裂缝深度无法从 2D 图像中测量",
            "结构安全性需要专业人员现场评估",
        ]),
    }


def generate_advice(report_path, output_dir="outputs/advice", dry_run=False):
    """
    主入口：读取检测报告 → 调用 DeepSeek → 保存维修建议

    Args:
        report_path: 裂缝检测 JSON 报告路径
        output_dir: 建议输出目录
        dry_run: 仅构建 prompt 不调用 API（调试用）

    Returns:
        advice: dict 维修建议
        output_path: str 保存路径
    """
    # 1. 加载报告
    report = load_report(report_path)
    context = extract_detection_context(report)

    # 2. 构建 prompt
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(context)

    # 3. 调用 DeepSeek（或 dry run）
    if dry_run:
        raw_response = json.dumps({
            "overall_assessment": "[DRY RUN] 未实际调用 DeepSeek API",
            "risk_level": "需人工复核",
            "possible_causes": ["请设置 DEEPSEEK_API_KEY 后运行以获取完整建议"],
            "repair_plan": [],
            "materials": [],
            "need_manual_review": True,
            "manual_review_reason": "当前为 dry run 模式，未调用 DeepSeek API",
            "limitations": ["缺少尺度标定"],
        }, ensure_ascii=False)
        print("[DRY RUN] 跳过 DeepSeek API 调用")
    else:
        api_key = get_api_key()
        print("正在调用 DeepSeek API ...")
        raw_response = call_deepseek(api_key, system_prompt, user_prompt)

    # 4. 解析响应
    parsed = parse_advice_json(raw_response)
    advice = normalize_advice(parsed, str(report_path), context)

    # 5. 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_stem = Path(report_path).stem
    filename = f"advice_{report_stem}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(advice, f, indent=2, ensure_ascii=False)

    print(f"维修建议已保存到: {filepath}")
    return advice, str(filepath)
