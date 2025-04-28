# 文本分类模型使用说明

## 1. 模型类型

本系统支持两种文本分类模型：
1. 单标签分类模型（`single_label.py`）：每个文本只能属于一个类别
2. 多标签分类模型（`multi_label.py`）：每个文本可以属于多个类别

## 2. 模型训练

### 2.1 单标签分类模型训练
```bash
# 开始新训练
python single_label.py

# 继续训练已有模型
python single_label.py --load saved_models/best_model_epoch10_acc0.85_20240101_120000
```

### 2.2 多标签分类模型训练
```bash
# 开始新训练
python multi_label.py

# 继续训练已有模型
python multi_label.py --load saved_models/best_model_epoch10_acc0.85_20240101_120000
```

## 3. 模型管理

### 3.1 查看已保存的模型
```python
from model_manager import ModelManager
manager = ModelManager()
models = manager.list_saved_models()  # 会显示模型类型（单标签/多标签）
```

### 3.2 删除模型
```python
from model_manager import ModelManager
manager = ModelManager()
manager.delete_model('saved_models/model_epoch10_acc0.85_20240101_120000')
```

## 4. 模型预测

### 4.1 单标签预测
```bash
# 单个文本预测
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --text "这辆车动力强劲，外观时尚，但油耗有点高"

# 批量预测
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --file test.txt
```

### 4.2 多标签预测
```bash
# 单个文本预测
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --text "这辆车动力强劲，外观时尚，但油耗有点高"

# 批量预测
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --file test.txt

# 使用自定义阈值预测（仅多标签模型支持）
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --text "这辆车动力强劲，外观时尚，但油耗有点高" --threshold 0.6

# 交互式预测（支持实时调整阈值）
python interactive_predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000
```

### 4.3 交互式预测功能
交互式预测模式支持以下命令：
- `quit` 或 `exit`：退出程序
- `threshold`：查看当前预测阈值
- `set_threshold 0.5`：设置新的预测阈值

交互式预测示例：
```
正在加载模型...
模型加载完成！
输入 'quit' 或 'exit' 退出程序
输入 'threshold' 查看当前阈值
输入 'set_threshold 0.5' 设置新的阈值
--------------------------------------------------

请输入要预测的文本: 这辆车动力强劲，外观时尚，但油耗有点高

预测结果:
文本: 这辆车动力强劲，外观时尚，但油耗有点高
标签: 动力, 概率: 0.9234
标签: 外观, 概率: 0.8765
标签: 油耗, 概率: 0.8123

请输入要预测的文本: threshold
当前阈值: 0.5

请输入要预测的文本: set_threshold 0.6
阈值已更新为: 0.6
```

## 5. 参数说明

### 5.1 训练参数
- `--load`: 加载已有模型继续训练（可选）
- 其他训练参数在各自的配置类中设置

### 5.2 预测参数
- `--model`: 要使用的模型路径（必需）
- `--text`: 要预测的文本（与`--file`二选一）
- `--file`: 包含多行文本的文件路径（与`--text`二选一）
- `--threshold`: 预测阈值（仅多标签模型支持）

## 6. 输出说明

### 6.1 训练输出
- 每个epoch的训练损失和准确率
- 测试集评估结果
- 模型保存信息

### 6.2 预测输出
- 单标签预测结果示例：
```
预测结果:
文本: 这辆车动力强劲，外观时尚，但油耗有点高
标签: 动力, 概率: 0.9234
```

- 多标签预测结果示例：
```
预测结果:
文本: 这辆车动力强劲，外观时尚，但油耗有点高
标签: 动力, 概率: 0.9234
标签: 外观, 概率: 0.8567
标签: 油耗, 概率: 0.7123
```

## 7. 文件说明

- `single_label.py`: 单标签分类训练脚本
- `multi_label.py`: 多标签分类训练脚本
- `model_manager.py`: 模型管理模块
- `predict.py`: 预测脚本
- `interactive_predict.py`: 交互式预测脚本
- `saved_models/`: 模型保存目录

## 8. 注意事项

1. 训练前确保数据文件路径正确
2. 预测时选择性能最好的模型
3. 批量预测时文本文件每行一个文本
4. 多标签模型可以根据需要调整预测阈值
5. 模型会自动保存到`saved_models`目录
6. 保存的模型包含完整的训练信息和配置
7. 预测时会自动识别模型类型（单标签/多标签）

## 9. 示例

### 9.1 训练示例
```bash
# 单标签模型训练
python single_label.py
python single_label.py --load saved_models/best_model_epoch10_acc0.85_20240101_120000

# 多标签模型训练
python multi_label.py
python multi_label.py --load saved_models/best_model_epoch10_acc0.85_20240101_120000
```

### 9.2 预测示例
```bash
# 单标签预测
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --text "这辆车动力强劲，外观时尚，但油耗有点高"
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --file test.txt

# 多标签预测
python predict.py --model saved_models/best_model_epoch5_acc0.9189_20250428_153158 --text "这辆车动力强劲，外观时尚，但油耗有点高"
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --file test.txt
python predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000 --text "这辆车动力强劲，外观时尚，但油耗有点高" --threshold 0.6

# 交互式预测
python interactive_predict.py --model saved_models/best_model_epoch10_acc0.85_20240101_120000
``` 