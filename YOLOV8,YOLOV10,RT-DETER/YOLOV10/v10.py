import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import platform
import shutil
import tempfile
import yaml
import glob
from tqdm import tqdm
from ultralytics import YOLO
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# 基础配置
# =====================
BASE_DIR = r'D:\肺部疾病诊断检测数据集(DST2016)\2222\lungdis(DST2016)'
WORK_DIR = r'D:\肺部疾病诊断检测数据集(DST2016)\2222\lungdis(DST2016)\runs'
DATASET_ROOT = BASE_DIR

# ========================
# 📍 硬编码路径配置区域 - 直接在这里修改你的路径
# ========================

# 🔑 核心路径配置 (修改这里的路径为你自己的)
CONFIG = {
    # ===== 数据集根目录 (支持中文路径) =====
    'dataset_root': DATASET_ROOT,

    # ===== YAML配置文件路径 =====
    'yaml_path': f'{BASE_DIR}\\data.yaml',

    # ===== 数据集划分配置 (完整目录结构) =====
    'dataset_splits': {
        # 训练集目录
        'train': {
            'images': f'{BASE_DIR}\\images\\train',  # 训练集图像目录 (完整绝对路径)
            'labels': f'{BASE_DIR}\\labels\\train'  # 训练集标签目录 (完整绝对路径)
        },
        # 验证集目录
        'val': {
            'images': f'{BASE_DIR}\\images\\valid',  # 验证集图像目录
            'labels': f'{BASE_DIR}\\labels\\valid'  # 验证集标签目录
        },
        # 测试集/预测集目录 (可选)
        'test': {
            'images': f'{BASE_DIR}\\images\\test',  # 测试集图像目录
            'labels': f'{BASE_DIR}\\labels\\test'  # 测试集标签目录 (预测时可不提供)
        }
    },

    # ===== 类别配置 (已移至YAML文件，此处仅作参考) =====
    # 实际类别配置将从YAML文件中读取，避免重复定义

    # ===== 训练参数配置 =====
    'train_params': {
        'epochs': 5,  # 训练轮数
        'batch_size': 16,  # 批次大小
        'imgsz': 640,  # 输入图像尺寸
        'save_dir': WORK_DIR,
        'experiment_name': 'yolov10_lung_disease'
    },

    # ===== 预测/检测参数配置 =====
    'predict_params': {
        'source': f'{BASE_DIR}\\images\\test',  # 预测源 - 与测试集图像目录统一
        'save_dir': f'{WORK_DIR}\\detect',
        'experiment_name': 'results',
        'conf_threshold': 0.25,  # 置信度阈值
        'iou_threshold': 0.45  # IOU阈值
    }
}


# ========================
# 0. 全局配置与路径处理 (关键修复)
# ========================

def ensure_unicode_path(path):
    """确保路径是Unicode字符串，处理中文路径问题"""
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        return str(path)
    return str(path)


def safe_imread(image_path):
    """安全读取图像，支持中文路径和特殊格式"""
    # 确保路径是Unicode
    image_path = ensure_unicode_path(image_path)

    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 尝试直接读取 (适用于Linux/Mac)
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception as e:
        print(f"直接读取失败: {str(e)}")

    # Windows特殊处理：复制到临时英文路径
    if platform.system() == 'Windows':
        print("🔄 Windows系统检测，使用临时路径处理中文路径...")
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            # 生成临时文件名 (保持扩展名)
            ext = os.path.splitext(image_path)[1]
            temp_path = os.path.join(temp_dir, f"temp{ext}")
            # 复制文件 (使用二进制模式)
            with open(image_path, 'rb') as src, open(temp_path, 'wb') as dst:
                dst.write(src.read())
            # 读取临时文件
            img = cv2.imread(temp_path)
            # 清理临时文件
            os.unlink(temp_path)
            os.rmdir(temp_dir)
            if img is not None:
                return img
        except Exception as e:
            print(f"Windows临时路径处理失败: {str(e)}")

    # 最后尝试：使用标准imread
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img
    except Exception as e:
        print(f"标准读取方式失败: {str(e)}")

    raise ValueError(f"无法读取图像: {image_path}. 请检查文件格式和权限。")


def get_image_files(directory, include_roboflow=True):
    """ 获取目录中所有图像文件，支持特殊格式
    参数:
        directory: 目录路径
        include_roboflow: 是否包含Roboflow格式文件 (xxx.jpg.rf.xxx.jpg)
    返回:
        文件路径列表
    """
    directory = ensure_unicode_path(directory)

    # 支持的扩展名
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

    # 添加Roboflow格式支持
    if include_roboflow:
        # 匹配 xxx.jpg.rf.xxx.jpg 和类似格式
        extensions.extend(['*.jpg.*.jpg', '*.jpeg.*.jpg', '*.png.*.jpg'])

    image_files = []
    for ext in extensions:
        # 使用glob查找文件
        pattern = os.path.join(directory, ext)
        files = glob.glob(pattern)
        image_files.extend(files)

        # 递归查找子目录
        for root, _, _ in os.walk(directory):
            if root == directory:
                continue
            pattern = os.path.join(root, ext)
            files = glob.glob(pattern)
            image_files.extend(files)

    # 去重并排序
    image_files = list(set(image_files))
    image_files.sort()

    print(f"📁 在 {directory} 中找到 {len(image_files)} 个图像文件")
    if image_files:
        print("🔍 样本文件:")
        for i, f in enumerate(image_files[:3]):
            print(f" {i + 1}. {os.path.basename(f)}")
        if len(image_files) > 3:
            print(f" ... 还有 {len(image_files) - 3} 个文件")

    return image_files


def get_label_files(directory):
    """ 获取目录中所有标签文件 (.txt)
    参数:
        directory: 目录路径
    返回:
        文件路径列表
    """
    directory = ensure_unicode_path(directory)

    # 支持的标签扩展名
    extensions = ['*.txt']

    label_files = []
    for ext in extensions:
        # 使用glob查找文件
        pattern = os.path.join(directory, ext)
        files = glob.glob(pattern)
        label_files.extend(files)

        # 递归查找子目录
        for root, _, _ in os.walk(directory):
            if root == directory:
                continue
            pattern = os.path.join(root, ext)
            files = glob.glob(pattern)
            label_files.extend(files)

    # 去重并排序
    label_files = list(set(label_files))
    label_files.sort()

    print(f"📁 在 {directory} 中找到 {len(label_files)} 个标签文件")
    if label_files:
        print("🔍 样本标签文件:")
        for i, f in enumerate(label_files[:3]):
            print(f" {i + 1}. {os.path.basename(f)}")
        if len(label_files) > 3:
            print(f" ... 还有 {len(label_files) - 3} 个文件")

    return label_files


# ========================
# 1. YOLOv10 核心类 (增强版)
# ========================

class YOLOv10Detector:
    """ YOLOv10目标检测器封装类，增强支持中文路径和特殊格式 """

    def __init__(self, model_size='n', checkpoint=None, data_yaml=None):
        """ 初始化YOLOv10检测器
        参数:
            model_size: 模型尺寸 'n', 's', 'm', 'b', 'l', 'x'
            checkpoint: 预训练权重路径，None则使用官方预训练权重
            data_yaml: YAML配置文件路径 (可选)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_yaml = data_yaml  # 存储YAML配置路径

        print(f"🚀 设备: {self.device.upper()}")
        print(f"🌐 系统: {platform.system()}, Python: {platform.python_version()}")

        # 检查OpenCV版本
        print(f"🖼️ OpenCV版本: {cv2.__version__}")

        # 模型尺寸映射
        size_map = {
            'n': 'yolov10n.pt',  # 最小最快
            's': 'yolov10s.pt',  # 平衡速度和精度
            'm': 'yolov10m.pt',  # 中等
            'b': 'yolov10b.pt',  # 基础
            'l': 'yolov10l.pt',  # 大型
            'x': 'yolov10x.pt'  # 最大最准
        }
        model_name = size_map.get(model_size.lower(), 'yolov10s.pt')
        self.model_size = model_size

        # 加载模型
        if checkpoint and Path(checkpoint).exists():
            print(f"📦 加载自定义权重: {checkpoint}")
            checkpoint = ensure_unicode_path(checkpoint)
            self.model = YOLO(checkpoint)
            self.is_custom_model = True
        else:
            print(f"🌐 加载官方预训练权重: {model_name}")
            # 检查权重文件是否存在，不存在则下载
            weights_path = Path('weights') / model_name
            weights_path.parent.mkdir(exist_ok=True)
            if not weights_path.exists():
                print(f"⬇️ 权重文件不存在，正在下载: {model_name}")
                # 从官方仓库下载权重
                import urllib.request
                url = f"https://github.com/THU-MIG/yolov10/releases/download/v1.0/{model_name}"
                try:
                    urllib.request.urlretrieve(url, str(weights_path))
                    print(f"✅ 权重下载成功: {weights_path}")
                except Exception as e:
                    print(f"❌ 权重下载失败: {str(e)}")
                    print("⚠️ 尝试使用内置模型加载方式")
                    self.model = YOLO(model_name)
            else:
                print(f"✅ 使用本地权重: {weights_path}")
                self.model = YOLO(str(weights_path))
            self.is_custom_model = False

        # 移动到设备
        self.model.to(self.device)

        # 医学影像专用预处理
        self.medical_transform = None
        self._setup_medical_transform()

        # 验证YAML配置
        if self.data_yaml and Path(self.data_yaml).exists():
            print(f"✅ YAML配置文件已设置: {self.data_yaml}")
            self._validate_yaml_config()
        elif self.data_yaml:
            print(f"⚠️ YAML配置文件不存在: {self.data_yaml}. 训练时需要提供有效的配置。")

        print(f"✅ YOLOv10-{model_size.upper()} 初始化完成!")

    def _validate_yaml_config(self):
        """验证YAML配置文件"""
        try:
            with open(self.data_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 检查必要字段
            required_fields = ['path', 'train', 'val', 'names']
            missing = [field for field in required_fields if field not in config]
            if missing:
                print(f"⚠️ YAML配置缺少必要字段: {missing}")
                return False

            # 验证路径是否存在
            base_path = Path(config['path'])

            # 验证训练集和验证集的images和labels目录
            for split in ['train', 'val']:
                if split in config:
                    # 验证images目录
                    img_dir = base_path / config[split]
                    if not img_dir.exists():
                        print(f"⚠️ {split} 图像目录不存在: {img_dir}")
                    else:
                        img_files = get_image_files(str(img_dir), include_roboflow=True)
                        print(f"✅ {split} 图像目录包含 {len(img_files)} 个图像文件")

                    # 验证labels目录 (YOLO约定：labels目录与images目录同级)
                    labels_dir = base_path / str(config[split]).replace('images', 'labels')
                    if not labels_dir.exists():
                        print(f"⚠️ {split} 标签目录不存在: {labels_dir}")
                        print(f"💡 YOLO约定：标签目录应为 {labels_dir}")
                    else:
                        label_files = get_label_files(str(labels_dir))
                        print(f"✅ {split} 标签目录包含 {len(label_files)} 个标签文件")

                    # 检查图像和标签文件数量是否匹配
                    if img_files and label_files:
                        if len(img_files) == len(label_files):
                            print(f"✅ {split} 图像和标签文件数量匹配: {len(img_files)}")
                        else:
                            print(
                                f"⚠️ {split} 图像和标签文件数量不匹配! 图像: {len(img_files)}, 标签: {len(label_files)}")

            # 验证测试集 (如果有)
            if 'test' in config:
                test_dir = base_path / config['test']
                if test_dir.exists():
                    test_files = get_image_files(str(test_dir), include_roboflow=True)
                    print(f"✅ 测试集目录包含 {len(test_files)} 个图像文件")

            # 验证类别名称
            if 'names' in config:
                print(f"✅ 类别数量: {len(config['names'])}")
                print(f"🎯 类别名称: {', '.join(config['names'])}")
            else:
                print("⚠️ 未找到类别名称配置")

            print("✅ YAML配置验证通过!")
            return True
        except Exception as e:
            print(f"❌ YAML配置验证失败: {str(e)}")
            return False

    def _setup_medical_transform(self):
        """设置医学影像专用预处理"""
        self.medical_transform = A.Compose([
            # 对比度受限自适应直方图均衡化 (CLAHE)
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            # 调整大小 (保持比例)
            A.Resize(640, 640, always_apply=True),
            # 归一化 (单通道医学影像统计)
            A.Normalize(mean=[0.5], std=[0.25], max_pixel_value=255.0),
            ToTensorV2()
        ])

    def train(self, epochs=100, batch_size=16, imgsz=640, patience=10,
              save_dir='runs/train', name='exp', exist_ok=True, resume=False, **kwargs):
        """ 训练YOLOv10模型
        参数:
            epochs: 训练轮数
            batch_size: 批次大小
            imgsz: 输入图像尺寸
            patience: 早停耐心值
            save_dir: 保存目录
            name: 实验名称
            exist_ok: 是否覆盖现有实验
            resume: 是否从上次中断处恢复
            **kwargs: 其他训练参数
        返回:
            训练结果
        """
        if not self.data_yaml or not Path(self.data_yaml).exists():
            raise ValueError("训练需要有效的YAML配置文件。请在初始化时设置data_yaml参数。")

        print("=" * 50)
        print("🎯 开始训练 YOLOv10")
        print("=" * 50)
        print(f"📊 配置:")
        print(f"  YAML配置: {self.data_yaml}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  图像尺寸: {imgsz}")
        print(f"  保存目录: {save_dir}/{name}")

        # 训练参数
        train_args = {
            'data': ensure_unicode_path(self.data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'project': save_dir,
            'name': name,
            'exist_ok': exist_ok,
            'patience': patience,
            'resume': resume,
            'verbose': True,
            'workers': 4,
            'cache': False,  # 禁用缓存，避免中文路径问题
            'close_mosaic': 10,  # 最后10轮关闭mosaic增强
            **kwargs
        }

        # 执行训练
        try:
            results = self.model.train(**train_args)
            print("✅ 训练完成!")
            return results
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")

            # 提供详细的错误诊断
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                print("🔍 诊断: 可能是路径问题。检查以下内容:")
                print(f" - YAML文件路径: {self.data_yaml}")
                print(f" - 训练集/验证集目录是否存在于YAML指定的位置")
                print(f" - 标签目录是否与图像目录结构匹配")
                print(f" - 确保所有路径不包含特殊字符或过长的路径名")

            if "cannot identify image file" in str(e).lower():
                print("🔍 诊断: 图像文件识别问题。检查以下内容:")
                print(f" - 确保所有图像文件完整且格式正确")
                print(f" - 对于Roboflow格式文件 (xxx.jpg.rf.xxx.jpg)，确保文件扩展名正确")
                print(f" - 确保每个图像都有对应的.txt标签文件")
                print(f" - 尝试将数据集移动到简单英文路径")

            raise

    def detect(self, source, conf_threshold=0.25, iou_threshold=0.45,
               save=False, save_path=None, visualize=True):
        """ 执行目标检测，增强中文路径支持 """
        # 确保source路径正确
        source = ensure_unicode_path(source)
        print(f"🔍 开始检测: {source}")

        # 特殊处理中文路径
        temp_source = None
        if platform.system() == 'Windows' and any('一' <= c <= '鿿' for c in source):
            print("🔄 Windows系统检测到中文路径，使用临时路径处理...")
            try:
                # 检查是文件还是目录
                if os.path.isfile(source):
                    # 创建临时目录
                    temp_dir = tempfile.mkdtemp()
                    # 生成临时文件名
                    ext = os.path.splitext(source)[1]
                    temp_source = os.path.join(temp_dir, f"temp{ext}")
                    # 复制文件
                    with open(source, 'rb') as src, open(temp_source, 'wb') as dst:
                        dst.write(src.read())
                elif os.path.isdir(source):
                    # 对于目录，直接使用原始路径，但提醒用户
                    print("⚠️ Windows系统上中文目录路径可能存在问题。建议将数据集移动到英文路径。")
                    temp_source = source
                else:
                    temp_source = source
            except Exception as e:
                print(f"⚠️ 临时路径创建失败: {str(e)}。尝试直接使用原始路径。")
                temp_source = source
        else:
            temp_source = source

        start_time = time.time()

        # 推理参数
        args = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'imgsz': 640,
            'device': self.device,
            'save': save,
            'project': 'runs/detect',
            'name': 'exp',
            'exist_ok': True,
            'half': True if self.device == 'cuda' else False,
            'show': visualize and not save
        }

        # 执行检测
        try:
            results = self.model.predict(source=temp_source, **args)
        except Exception as e:
            print(f"❌ 检测失败: {str(e)}")
            if temp_source != source:
                print(f"🔍 原始路径: {source}")
                print(f"🔍 临时路径: {temp_source}")
            raise

        # 处理结果
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f"✅ 检测完成! 耗时: {inference_time:.3f}s ({fps:.1f} FPS)")

        # 清理临时文件
        if temp_source != source and temp_source and os.path.exists(temp_source):
            try:
                if os.path.isfile(temp_source):
                    os.unlink(temp_source)
                elif os.path.isdir(temp_source) and 'temp' in temp_source.lower():
                    shutil.rmtree(temp_source)
            except Exception as e:
                print(f"⚠️ 临时文件清理失败: {str(e)}")

        # 处理结果保存
        if save and save_path:
            # 确保保存路径是Unicode
            save_path = ensure_unicode_path(save_path)
            # 确保目录存在
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

            # 处理结果文件
            result_dir = Path(args['project']) / args['name']
            if result_dir.exists():
                try:
                    # 获取最新结果文件
                    result_files = list(result_dir.glob('*.*'))
                    if result_files:
                        latest_file = max(result_files, key=os.path.getctime)
                        # 复制到目标位置
                        shutil.copy2(str(latest_file), save_path)
                        print(f"💾 结果已保存到: {save_path}")
                except Exception as e:
                    print(f"⚠️ 结果保存失败: {str(e)}")

        return results


# ========================
# 2. 数据集配置工具 (完整目录结构版)
# ========================

class DatasetConfigurator:
    """数据集配置工具，支持完整目录结构和中文路径"""

    def __init__(self, config):
        """ 初始化数据集配置
        参数:
            config: 包含所有路径配置的字典
        """
        self.config = config
        self.dataset_root = ensure_unicode_path(config['dataset_root'])

        # 构建完整路径
        self.paths = {}
        for split_name, split_config in config['dataset_splits'].items():
            self.paths[split_name] = {
                'images': split_config['images'] if split_config.get('images') else None,
                'labels': split_config['labels'] if split_config.get('labels') else None
            }

        # 验证目录结构
        self._validate_dataset_structure()

    def _validate_dataset_structure(self):
        """验证完整的数据集目录结构"""
        print("\n" + "=" * 50)
        print("🔍 验证数据集目录结构")
        print("=" * 50)

        # 验证根目录
        if not Path(self.dataset_root).exists():
            raise ValueError(f"❌ 数据集根目录不存在: {self.dataset_root}")
        print(f"✅ 数据集根目录: {self.dataset_root}")

        # 验证所有目录
        for split in ['train', 'val', 'test']:
            if split not in self.paths:
                continue

            print(f"\n📊 {split.upper()} 数据集验证:")

            # 验证images目录
            img_dir = self.paths[split]['images']
            if img_dir and Path(img_dir).exists():
                img_files = get_image_files(img_dir, include_roboflow=True)
                print(f"✅ {split} 图像目录: {img_dir}")
                print(f" 📁 图像文件数量: {len(img_files)}")
                # 显示部分样本
                if img_files:
                    print(" 🖼️ 样本图像:")
                    for i, f in enumerate(img_files[:3]):
                        print(f"  {i + 1}. {os.path.basename(f)}")
            elif img_dir:
                print(f"❌ {split} 图像目录不存在: {img_dir}")

            # 验证labels目录 (训练和验证需要，测试可选)
            label_dir = self.paths[split]['labels']
            if label_dir and Path(label_dir).exists():
                label_files = get_label_files(label_dir)
                print(f"✅ {split} 标签目录: {label_dir}")
                print(f" 📄 标签文件数量: {len(label_files)}")
                # 显示部分样本
                if label_files:
                    print(" 🏷️ 样本标签:")
                    for i, f in enumerate(label_files[:3]):
                        print(f"  {i + 1}. {os.path.basename(f)}")
            elif label_dir and split in ['train', 'val']:
                # 训练和验证必须有标签
                print(f"❌ {split} 标签目录不存在: {label_dir}")
            elif label_dir:
                print(f"⚠️ {split} 标签目录不存在 (预测时可不提供): {label_dir}")

    def create_yaml_config(self, override=True):
        """创建YAML配置文件，包含完整的目录结构"""
        yaml_path = Path(self.config['yaml_path'])
        if yaml_path.exists() and not override:
            print(f"⚠️ YAML配置文件已存在: {yaml_path}")
            print("🔄 使用现有配置文件")
            return str(yaml_path)

        # 确保输出目录存在
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建配置
        config = {
            'path': str(self.dataset_root),  # 数据集根目录
            'train': str(Path(self.config['dataset_splits']['train']['images']).relative_to(self.dataset_root)),
            # 训练集图像目录 (相对路径)
            'val': str(Path(self.config['dataset_splits']['val']['images']).relative_to(self.dataset_root)),
            # 验证集图像目录 (相对路径)
            'names': {
                0: "atelectasis",
                1: "cardiomegaly",
                2: "consolidation",
                3: "edema",
                4: "effusion",
                5: "emphysema",
                6: "fibrosis",
                7: "hernia",
                8: "infiltration",
                9: "mass",
                10: "nodule",
                11: "pleural_thickening",
                12: "pneumonia",
                13: "pneumothorax"
            },
            'nc': 14
        }

        # 添加测试集 (如果有)
        if 'test' in self.config['dataset_splits'] and self.config['dataset_splits']['test']['images']:
            config['test'] = str(Path(self.config['dataset_splits']['test']['images']).relative_to(self.dataset_root))

        # 保存YAML
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, sort_keys=False, allow_unicode=True)
            print(f"\n" + "=" * 50)
            print(f"✅ YAML配置文件已创建: {yaml_path}")
            print("=" * 50)
            print("📁 YAML配置内容:")
            print(f"  path: {config['path']}")
            print(f"  train: {config['train']}")
            print(f"  val: {config['val']}")
            if 'test' in config:
                print(f"  test: {config['test']}")
            print(f"  nc: {config['nc']}")
            print(f"  names: {config['names']}")
            return str(yaml_path)
        except Exception as e:
            print(f"❌ 保存YAML配置失败: {str(e)}")
            raise

    def verify_file_matching(self):
        """验证图像文件和标签文件是否匹配"""
        print("\n" + "=" * 50)
        print("🔍 验证文件匹配性")
        print("=" * 50)

        for split in ['train', 'val']:
            if split not in self.paths:
                continue

            img_dir = self.paths[split]['images']
            label_dir = self.paths[split]['labels']

            if not (img_dir and label_dir and Path(img_dir).exists() and Path(label_dir).exists()):
                continue

            print(f"\n📊 {split.upper()} 数据集文件匹配验证:")

            # 获取所有图像文件 (不包括路径，只取文件名)
            img_files = get_image_files(img_dir, include_roboflow=True)
            img_names = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}

            # 获取所有标签文件
            label_files = get_label_files(label_dir)
            label_names = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

            # 找出有图像但没有标签的文件
            missing_labels = [name for name in img_names if name not in label_names]

            # 找出有标签但没有图像的文件
            missing_images = [name for name in label_names if name not in img_names]

            print(f"✅ 总图像文件: {len(img_files)}")
            print(f"✅ 总标签文件: {len(label_files)}")
            print(f"✅ 匹配文件数: {len(img_names) - len(missing_labels)}")

            if missing_labels:
                print(f"❌ {len(missing_labels)} 个图像缺少标签文件:")
                for i, name in enumerate(missing_labels[:5]):
                    print(f"  {i + 1}. {name} (图像: {img_names[name]})")
                if len(missing_labels) > 5:
                    print(f"  ... 还有 {len(missing_labels) - 5} 个")

            if missing_images:
                print(f"❌ {len(missing_images)} 个标签缺少图像文件:")
                for i, name in enumerate(missing_images[:5]):
                    print(f"  {i + 1}. {name} (标签: {label_names[name]})")
                if len(missing_images) > 5:
                    print(f"  ... 还有 {len(missing_images) - 5} 个")

            if not missing_labels and not missing_images:
                print("✅ 所有图像和标签文件完美匹配!")


# ========================
# 3. 主程序 (完整目录结构版)
# ========================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 YOLOv10 训练与预测系统 (修正完整版)")
    print("=" * 60)

    # ========================
    # 1. 显示当前配置
    # ========================
    print("\n" + "=" * 50)
    print("📋 当前路径配置")
    print("=" * 50)
    print(f"📁 数据集根目录: {CONFIG['dataset_root']}")

    # 训练集配置
    print(f"\n🏋️ 训练集配置:")
    train_img_dir = CONFIG['dataset_splits']['train']['images']
    train_label_dir = CONFIG['dataset_splits']['train']['labels']
    print(f" 🖼️ 图像目录: {train_img_dir}")
    print(f" 🏷️ 标签目录: {train_label_dir}")

    # 验证集配置
    print(f"\n📊 验证集配置:")
    val_img_dir = CONFIG['dataset_splits']['val']['images']
    val_label_dir = CONFIG['dataset_splits']['val']['labels']
    print(f" 🖼️ 图像目录: {val_img_dir}")
    print(f" 🏷️ 标签目录: {val_label_dir}")

    # 测试集配置
    print(f"\n🔍 测试集配置:")
    if 'test' in CONFIG['dataset_splits'] and CONFIG['dataset_splits']['test']['images']:
        test_img_dir = CONFIG['dataset_splits']['test']['images']
        print(f" 🖼️ 图像目录: {test_img_dir}")
    if 'test' in CONFIG['dataset_splits'] and CONFIG['dataset_splits']['test'].get('labels'):
        test_label_dir = CONFIG['dataset_splits']['test']['labels']
        print(f" 🏷️ 标签目录: {test_label_dir}")
    if not ('test' in CONFIG['dataset_splits'] and (
            CONFIG['dataset_splits']['test']['images'] or CONFIG['dataset_splits']['test'].get('labels'))):
        print(" ⚠️ 未配置测试集")

    # 训练参数
    print(f"\n⚙️ 训练参数:")
    print(f" 🔁 训练轮数: {CONFIG['train_params']['epochs']}")
    print(f" 📦 批次大小: {CONFIG['train_params']['batch_size']}")
    print(f" 🖼️ 图像尺寸: {CONFIG['train_params']['imgsz']}")
    print(f" 💾 保存目录: {CONFIG['train_params']['save_dir']}/{CONFIG['train_params']['experiment_name']}")

    # 预测参数
    print(f"\n🔮 预测参数:")
    print(f" 📂 预测源: {CONFIG['predict_params']['source']}")
    print(f" 💾 保存目录: {CONFIG['predict_params']['save_dir']}/{CONFIG['predict_params']['experiment_name']}")
    print(f" ✅ 置信度阈值: {CONFIG['predict_params']['conf_threshold']}")
    print(f" 🎯 IOU阈值: {CONFIG['predict_params']['iou_threshold']}")

    # ========================
    # 2. 验证和创建YAML配置
    # ========================
    print("\n" + "=" * 50)
    print("⚙️ 验证数据集和创建YAML配置")
    print("=" * 50)

    try:
        # 创建配置器
        configurator = DatasetConfigurator(CONFIG)

        # 创建YAML配置
        yaml_path = configurator.create_yaml_config(override=True)

        # 验证文件匹配性
        configurator.verify_file_matching()
    except Exception as e:
        print(f"❌ 配置创建失败: {str(e)}")
        print("💡 请检查以下内容:")
        print(f" - 数据集根目录是否存在: {CONFIG['dataset_root']}")
        print(f" - 训练集图像目录是否存在: {train_img_dir}")
        print(f" - 训练集标签目录是否存在: {train_label_dir}")
        print(f" - 验证集图像目录是否存在: {val_img_dir}")
        print(f" - 验证集标签目录是否存在: {val_label_dir}")
        exit(1)

    # ========================
    # 3. 训练模型
    # ========================
    print("\n" + "=" * 50)
    print("🏋️ 模型训练")
    print("=" * 50)

    try:
        # 初始化检测器
        detector = YOLOv10Detector(
            model_size='s',  # 使用小型模型
            checkpoint=None,
            data_yaml=yaml_path
        )

        # 开始训练
        results = detector.train(
            epochs=CONFIG['train_params']['epochs'],
            batch_size=CONFIG['train_params']['batch_size'],
            imgsz=CONFIG['train_params']['imgsz'],
            patience=15,
            save_dir=CONFIG['train_params']['save_dir'],
            name=CONFIG['train_params']['experiment_name'],
            exist_ok=True,
            resume=False
        )
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        print("💡 请检查以下内容:")
        print(f" - YAML配置文件是否正确: {yaml_path}")
        print(f" - 训练集/验证集路径是否在YAML中正确配置")
        print(f" - 图像和标签文件是否匹配")
        print(f" - GPU内存是否足够 (尝试减小batch_size)")
        # 不退出，继续尝试预测

    # ========================
    # 4. 预测/检测
    # ========================
    print("\n" + "=" * 50)
    print("🔍 模型预测/检测")
    print("=" * 50)

    try:
        # 确保模型已加载
        if 'detector' not in locals():
            detector = YOLOv10Detector(
                model_size='s',
                checkpoint=None,
                data_yaml=yaml_path
            )

        # 检查是否有训练好的权重
        best_weight_path = Path(CONFIG['train_params']['save_dir']) / CONFIG['train_params'][
            'experiment_name'] / 'weights' / 'best.pt'
        if best_weight_path.exists():
            print(f"✅ 找到训练好的最佳权重: {best_weight_path}")
            detector = YOLOv10Detector(
                model_size='s',
                checkpoint=str(best_weight_path),  # 使用训练好的权重
                data_yaml=yaml_path
            )
        else:
            print("⚠️ 未找到训练好的权重，使用预训练模型进行预测")

        # 创建预测保存目录
        predict_save_dir = Path(CONFIG['predict_params']['save_dir']) / CONFIG['predict_params']['experiment_name']
        predict_save_dir.mkdir(parents=True, exist_ok=True)

        # 为每个测试图像创建单独的保存路径
        predict_source = CONFIG['predict_params']['source']

        # 确定预测源是目录还是文件
        if os.path.isdir(predict_source):
            # 获取所有测试图像
            test_images = get_image_files(predict_source, include_roboflow=True)
            print(f"🖼️ 找到 {len(test_images)} 个测试图像")

            # 逐一预测
            for i, img_path in enumerate(test_images[:5]):  # 只预测前5个作为示例
                img_name = os.path.basename(img_path)
                save_path = predict_save_dir / f"result_{i + 1}_{img_name}"
                print(f"\n🔍 预测图像 {i + 1}/{len(test_images[:5])}: {img_name}")
                print(f"💾 保存结果到: {save_path}")

                detector.detect(
                    source=img_path,
                    conf_threshold=CONFIG['predict_params']['conf_threshold'],
                    iou_threshold=CONFIG['predict_params']['iou_threshold'],
                    save=True,
                    save_path=str(save_path),
                    visualize=False
                )

            if len(test_images) > 5:
                print(f"\n💡 仅预测了前5个图像作为示例，总共 {len(test_images)} 个测试图像")
        else:
            # 单个文件预测
            save_path = predict_save_dir / f"result_{os.path.basename(predict_source)}"
            print(f"🔍 预测单个文件: {predict_source}")
            print(f"💾 保存结果到: {save_path}")

            detector.detect(
                source=predict_source,
                conf_threshold=CONFIG['predict_params']['conf_threshold'],
                iou_threshold=CONFIG['predict_params']['iou_threshold'],
                save=True,
                save_path=str(save_path),
                visualize=True
            )

        print("\n✅ 预测完成!")
        print(f"📁 所有预测结果保存在: {predict_save_dir}")
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        print("💡 请检查以下内容:")
        print(f" - 预测源路径是否存在: {predict_source}")
        print(f" - 模型权重是否正确加载")
        print(f" - 预测保存目录是否有写入权限: {CONFIG['predict_params']['save_dir']}")

    print("\n" + "=" * 60)
    print("🎉 程序执行完成!")
    print("=" * 60)