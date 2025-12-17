# RNTR.py
"""
全自动肺部CT病灶检测系统 (RT-DETR)
- 支持中文路径 & .rf. 文件名
- 自动训练 + 批量预测 + 评估 + 可视化
- 运行即全自动执行，无需命令行参数
"""

import os
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import torch
import warnings

# ==================== 【关键修复】====================
torch.use_deterministic_algorithms(False)  # 允许非确定性算法
warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # 忽略相关警告
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 稳定 CUDA 操作
# =====================================================
# ==================== 【用户配置区】====================
ROOT_DIR = r"D:\2016\2222\lungdis(DST2016)"      # 数据根目录
RESULTS_DIR = r"D:\2016\2222\lungdis(DST2016)"    # 结果输出目录

MODEL_NAME = "rtdetr-l.pt"
IMG_SIZE = 1024
BATCH_SIZE = 4
EPOCHS = 50
DEVICE = 0  # -1 for CPU

CLASS_NAMES = [
    "atelectasis", "cardiomegaly", "consolidation", "edema", "effusion",
    "emphysema", "fibrosis", "hernia", "infiltration", "mass",
    "nodule", "pleural_thickening", "pneumonia", "pneumothorax"
]
NUM_CLASSES = len(CLASS_NAMES)

# =====================================================

def ensure_path(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())

def find_label_file(image_path):
    img_path = Path(image_path)
    stem = img_path.stem
    label_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / f"{stem}.txt"
    return label_path if label_path.exists() else None

def prepare_yolo_dataset(images_dir, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_files.extend(images_dir.rglob(ext))

    copied = 0
    for img_path in image_files:
        label_path = find_label_file(img_path)
        if not label_path:
            continue

        new_img_name = f"{copied:06d}.jpg"
        new_img_path = out_images / new_img_name
        new_label_path = out_labels / f"{copied:06d}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(str(new_img_path), img)
        shutil.copy(label_path, new_label_path)
        copied += 1

    print(f"✅ 已处理 {copied} 张图像到 {output_dir}")
    return str(output_dir)

import yaml

import yaml

def create_data_yaml(train_dir):
    """
    创建 YOLO 格式的 data.yaml 文件并返回其路径
    """
    data_dict = {
        'path': str(Path(train_dir).parent.parent),  # 指向 lungdis(DST2016) 根目录
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASS_NAMES  # Ultralytics 支持直接 list
    }

    yaml_path = os.path.join(RESULTS_DIR, "data.yaml")
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ data.yaml 已保存至: {yaml_path}")
    return yaml_path


def check_dataset_integrity():
    from pathlib import Path
    root = Path(ROOT_DIR)
    for split in ['train', 'val']:
        img_dir = root / 'images' / split
        lbl_dir = root / 'labels' / split

        # 获取所有图像
        imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            imgs.extend(img_dir.rglob(ext))

        # 检查对应标签
        valid_imgs = []
        for img in imgs:
            label_path = lbl_dir / f"{img.stem}.txt"
            if label_path.exists():
                # 检查标签文件是否非空
                if os.path.getsize(label_path) > 0:
                    valid_imgs.append(img)

        print(f"\n🔍 {split.upper()} 集诊断:")
        print(f"   • 扫描目录: {img_dir}")
        print(f"   • 找到图像: {len(imgs)}")
        print(f"   • 有效图像 (有非空标签): {len(valid_imgs)}")
        print(f"   • 标签示例: {list(lbl_dir.glob('*.txt'))[:3]}")

    # 检查类别数量
    print(f"\n🏷️  配置类别数: {NUM_CLASSES} ({', '.join(CLASS_NAMES[:3])}...)")


# 在 train_and_validate() 开头调用

def train_and_validate():
    check_dataset_integrity()

    print("🚀 准备训练数据...")
    train_clean = ensure_path(os.path.join(RESULTS_DIR, "dataset", "train"))
    val_clean = ensure_path(os.path.join(RESULTS_DIR, "dataset", "val"))

    shutil.rmtree(train_clean, ignore_errors=True)
    shutil.rmtree(val_clean, ignore_errors=True)

    prepare_yolo_dataset(os.path.join(ROOT_DIR, "images", "train"), train_clean)
    prepare_yolo_dataset(os.path.join(ROOT_DIR, "images", "val"), val_clean)

    print("🧠 加载 RT-DETR 模型...")
    model = YOLO(MODEL_NAME)
    model.model.yaml['nc'] = NUM_CLASSES  # 覆盖模型配置

    model.model.names = CLASS_NAMES  # 设置类别名
    # ✅ 使用原始数据路径
    data_yaml = create_data_yaml(os.path.join(ROOT_DIR, "images", "train"))

    model.train(
        data=data_yaml,
        imgsz=IMG_SIZE,
        epochs=4,
        batch=BATCH_SIZE,
        name="lung_ct_rtdetr",
        device=DEVICE,
        project=RESULTS_DIR,
        exist_ok=True,
        # ✅ 关键调整：
        optimizer='adamw',
        lr0=0.0001,  # 从 0.001 降低 10 倍
        lrf=0.01,  # 末尾学习率 = lr0 * lrf
        augment=True,
        mosaic=0.0,  # 医学图像禁用 mosaic（会破坏病灶）
        mixup=0.0,  # 禁用 mixup
        flipud=0.0,  # 禁用上下翻转（CT 有方向性）
        fliplr=0.5,  # 仅保留左右翻转
        workers=0,  # Windows 必须为 0
        patience=15,
        close_mosaic=0  # 不关闭 mosaic（因为已禁用）
    )

    print("🔍 验证模型...")
    metrics = model.val(data=data_yaml, imgsz=IMG_SIZE,workers=0)  # 验证也用同一个 YAML
    return metrics




def batch_predict_and_save_csv(test_images_dir):
    best_pt = os.path.join(RESULTS_DIR, "lung_ct_rtdetr", "weights", "best.pt")
    model = YOLO(best_pt)

    test_dir = Path(test_images_dir)
    save_dir = Path(RESULTS_DIR) / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_paths.extend(test_dir.rglob(ext))

    results_list = []
    print(f"🎯 批量预测 {len(image_paths)} 张测试图像...")

    for img_path in image_paths:
        try:
            results = model(str(img_path), imgsz=IMG_SIZE, conf=0.01)  # 低阈值保留更多结果用于评估
            boxes = results[0].boxes

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                results_list.append({
                    "image": img_path.name,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # 保存可视化结果（高阈值）
            annotated = results[0].plot(conf_thres=0.25)
            cv2.imwrite(str(save_dir / img_path.name), annotated)

        except Exception as e:
            print(f"⚠️ 跳过 {img_path}: {e}")

    # 保存 CSV
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📊 预测结果已保存至: {csv_path}")
    return df

def compute_roc_data(test_images_dir, pred_df):
    """
    基于预测结果和真实标签，计算 ROC 所需的 y_true 和 y_score
    返回: dict {class_id: (y_true, y_scores)}
    """
    from collections import defaultdict

    # 按图像分组预测结果
    pred_by_image = defaultdict(list)
    for _, row in pred_df.iterrows():
        pred_by_image[row["image"]].append(row)

    all_y_true = defaultdict(list)
    all_y_scores = defaultdict(list)

    test_images_dir = Path(test_images_dir)
    test_labels_dir = test_images_dir.parent.parent / "labels" / test_images_dir.name

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_files.extend(test_images_dir.rglob(ext))

    for img_path in image_files:
        label_path = find_label_file(img_path)
        if not label_path or not label_path.exists():
            continue

        # 读取真实标签 [class_id, cx, cy, w, h]（归一化）
        with open(label_path, 'r', encoding='utf-8') as f:
            gt_boxes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2) * IMG_SIZE
                y1 = (cy - h / 2) * IMG_SIZE
                x2 = (cx + w / 2) * IMG_SIZE
                y2 = (cy + h / 2) * IMG_SIZE
                gt_boxes.append((cls_id, np.array([x1, y1, x2, y2])))

        # 获取该图的预测
        preds = pred_by_image.get(img_path.name, [])
        pred_boxes = []
        for p in preds:
            pred_boxes.append((
                p["class_id"],
                np.array([p["x1"], p["y1"], p["x2"], p["y2"]]),
                p["confidence"]
            ))

        # 对每个类别独立处理
        for cls_id in range(NUM_CLASSES):
            # 当前类别的 GT
            gt_cls = [box for cid, box in gt_boxes if cid == cls_id]
            # 当前类别的预测
            pred_cls = [(box, conf) for cid, box, conf in pred_boxes if cid == cls_id]

            # 标记所有预测为负例（初始）
            y_true_cls = [0] * len(pred_cls)
            y_score_cls = [conf for _, conf in pred_cls]

            # 如果有 GT，尝试匹配（IoU >= 0.5）
            matched_gt = set()
            if gt_cls:
                for i, (pred_box, conf) in enumerate(pred_cls):
                    best_iou = 0
                    best_j = -1
                    for j, gt_box in enumerate(gt_cls):
                        if j in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= 0.5 and best_j != -1:
                        y_true_cls[i] = 1
                        matched_gt.add(best_j)

            if y_true_cls:  # 避免空列表
                all_y_true[cls_id].extend(y_true_cls)
                all_y_scores[cls_id].extend(y_score_cls)

    return dict(all_y_true), dict(all_y_scores)

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def generate_evaluation_report(metrics, pred_df):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid")

    # 计算 ROC/PR 所需数据
    test_images_dir = os.path.join(ROOT_DIR, "images", "test")
    y_true_dict, y_score_dict = compute_roc_data(test_images_dir, pred_df)

    # 创建多页 PDF（使用 matplotlib 的 PdfPages）
    from matplotlib.backends.backend_pdf import PdfPages
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.pdf")

    with PdfPages(report_path) as pdf:
        # =============== 第一页：基础指标 ===============
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RT-DETR 肺部CT病灶检测评估报告 - 第1页", fontsize=16)

        # 1. mAP
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        axes[0, 0].bar(["mAP@0.5", "mAP@0.5:0.95"], [map50, map5095], color=['skyblue', 'salmon'])
        axes[0, 0].set_title("整体检测性能")
        for i, v in enumerate([map50, map5095]):
            axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')

        # 2. 各类别 AP@0.5
        ap_per_class = metrics.box.ap[:, 0]
        axes[0, 1].barh(CLASS_NAMES, ap_per_class, color='lightgreen')
        axes[0, 1].set_title("各类别 AP@0.5")

        # 3. 置信度分布
        axes[1, 0].hist(pred_df["confidence"], bins=30, color='orange', alpha=0.7)
        axes[1, 0].set_title("预测置信度分布")

        # 4. 各类别预测数量
        class_counts = pred_df["class_name"].value_counts()
        axes[1, 1].bar(class_counts.index, class_counts.values, color='purple', alpha=0.7)
        axes[1, 1].set_title("各类别预测数量")
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()

        # =============== 第二页：ROC 曲线 ===============
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle("RT-DETR 肺部CT病灶检测评估报告 - 第2页：ROC 曲线", fontsize=16)

        macro_auc_roc = 0
        valid_classes_roc = 0

        for cls_id in range(NUM_CLASSES):
            if cls_id in y_true_dict and len(y_true_dict[cls_id]) > 0:
                y_true = y_true_dict[cls_id]
                y_score = y_score_dict[cls_id]
                if len(np.unique(y_true)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                macro_auc_roc += roc_auc
                valid_classes_roc += 1
                ax.plot(fpr, tpr, lw=1.5, alpha=0.8,
                        label=f'{CLASS_NAMES[cls_id]} (AUC={roc_auc:.2f})')

        if valid_classes_roc > 0:
            macro_auc_roc /= valid_classes_roc
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'各类别 ROC 曲线 (宏平均 AUC = {macro_auc_roc:.3f})')
            ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()

        # =============== 第三页：PR 曲线 + F1 ===============
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("RT-DETR 肺部CT病灶检测评估报告 - 第3页：PR曲线与F1-score", fontsize=16)

        # PR 曲线
        macro_ap_pr = 0
        valid_classes_pr = 0
        f1_scores = []

        for cls_id in range(NUM_CLASSES):
            if cls_id in y_true_dict and len(y_true_dict[cls_id]) > 0:
                y_true = y_true_dict[cls_id]
                y_score = y_score_dict[cls_id]
                if len(np.unique(y_true)) < 2:
                    f1_scores.append(0)
                    continue

                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                ap = auc(recall, precision)
                macro_ap_pr += ap
                valid_classes_pr += 1

                # 计算 F1 并找最大值
                f1_vals = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_vals)
                f1_scores.append(best_f1)

                ax1.plot(recall, precision, lw=1.5, alpha=0.8,
                         label=f'{CLASS_NAMES[cls_id]} (AP={ap:.2f})')
            else:
                f1_scores.append(0)

        if valid_classes_pr > 0:
            macro_ap_pr /= valid_classes_pr
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
            ax1.set_title(f'各类别 PR 曲线 (宏平均 AP = {macro_ap_pr:.3f})')
            ax1.legend(loc="lower left", fontsize=7)

        # F1-score 柱状图
        ax2.barh(CLASS_NAMES, f1_scores, color='coral')
        ax2.set_title("各类别最佳 F1-score")
        ax2.set_xlabel("F1-score")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()

    print(f"📈 评估报告（含ROC、PR、F1）已保存至: {report_path}")

def main():
    print("="*60)
    print("🤖 全自动 RT-DETR 肺部CT病灶检测系统启动！")
    print("="*60)

    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"

    # Step 1: 训练 + 验证
    metrics = train_and_validate()

    # Step 2: 批量预测测试集
    test_images_dir = os.path.join(ROOT_DIR, "images", "test")
    pred_df = batch_predict_and_save_csv(test_images_dir)

    # Step 3: 生成评估报告
    generate_evaluation_report(metrics, pred_df)

    print("\n🎉 全流程完成！")
    print(f"📁 结果目录: {RESULTS_DIR}")
    print("   ├── predictions/       # 带检测框的图像")
    print("   ├── results.csv        # 所有预测结果（CSV）")
    print("   └── evaluation_report.pdf  # 评估图表")

if __name__ == "__main__":
    main()