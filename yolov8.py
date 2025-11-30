import torch
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO


# 支持中文路径读图（保持原样，但确保返回 3 通道）
def imread_chinese_path(image_path: str) -> np.ndarray:
    image_path = str(image_path)
    if image_path.lower().endswith('.dcm'):
        import pydicom
        ds = pydicom.dcmread(image_path)
        img = ds.pixel_array
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转为 3 通道
        return img
    else:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # ← 关键：强制读为 BGR 3 通道！
        if img is None:
            raise ValueError(f"无法解码图像: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class ThreeChannelYOLO:
    def __init__(self, model_size='s', num_classes=3):
        self.model_size = model_size
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 直接加载预训练模型，不修改结构！
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("✅ 使用原生三通道 YOLOv8 模型")

    def train(self, data_yaml, epochs=100, imgsz=640, batch=8, name='yolov8_3channel'):
        print("🚀 开始训练三通道 YOLOv8...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            device=self.device,
            hsv_v=0.3,
            fliplr=0.5,
            mosaic=0.5,
            copy_paste=0.0,
            patience=20
        )
        print("✅ 训练完成！")
        return results

    def predict(self, image_path, conf=0.25, save_dir=None):
        image_path = Path(image_path)
        if image_path.is_file():
            image_files = [image_path]
        elif image_path.is_dir():
            supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm')
            image_files = [f for f in image_path.rglob('*') if f.suffix.lower() in supported_ext]
            if not image_files:
                raise ValueError(f"❌ 在 {image_path} 中未找到图像！")
            print(f"📁 找到 {len(image_files)} 张图像，开始批量预测...")
        else:
            raise FileNotFoundError(f"路径不存在: {image_path}")

        if save_dir is None:
            save_dir = image_path.parent / f"{image_path.name}_predictions"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            try:
                img = imread_chinese_path(str(img_path))  # 确保是 (H, W, 3)
                if img.shape[:2] != (640, 640):  # 或你训练用的 imgsz
                    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

                # 保存临时文件（3 通道）
                temp_path = os.path.join(tempfile.gettempdir(), "temp_pred_input.jpg")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                cv2.imwrite(temp_path, img)

                # 预测（YOLO 自动读 3 通道）
                results = self.model.predict(
                    source=temp_path,
                    conf=conf,
                    imgsz=640,
                    device=self.device,
                    save=False,
                    show=False
                )

                # 保存结果
                save_path = save_dir / f"{img_path.stem}_pred.jpg"
                result_img = results[0].plot()  # plot() 会自动在原图上画框
                cv2.imencode('.jpg', result_img)[1].tofile(str(save_path))
                print(f"✅ {img_path.name} → {save_path.name}")

            except Exception as e:
                print(f"⚠️ 跳过 {img_path}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        print(f"🎉 预测完成！结果保存至: {save_dir}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # ✅ YAML 路径（指向你的数据集配置）
    DATASET_YAML = r"D:\肺部疾病诊断检测数据集(DST2016)\2222\lungdis(DST2016)\data.yaml"

    # 测试路径（可以是文件夹）
    TEST_PATH = r"D:\肺部疾病诊断检测数据集(DST2016)\2222\lungdis(DST2016)\images\test"
    SAVE_DIR = r"D:\肺部疾病诊断检测数据集(DST2016)\2222\lungdis(DST2016)\新建文件夹"

    if not os.path.isfile(DATASET_YAML):
        raise FileNotFoundError(f"YAML 文件不存在: {DATASET_YAML}")

    # 初始化三通道模型（不再修改网络结构！）
    model = ThreeChannelYOLO(model_size='m', num_classes=3)

    # 【训练】取消注释即可
    # model.train(data_yaml=DATASET_YAML, epochs=50, imgsz=640, batch=8)

    # 【预测】
    model.predict(image_path=TEST_PATH, conf=0.3, save_dir=SAVE_DIR)