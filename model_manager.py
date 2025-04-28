import os
import torch
import json
from datetime import datetime
import shutil
from transformers import BertTokenizer

#加载已有模型继续训练
#python multi_label.py --load saved_models/best_model_epoch10_acc0.85_20240101_120000

class ModelManager:
    def __init__(self, save_dir='saved_models'):
        """
        初始化模型管理器
        :param save_dir: 模型保存的目录
        """
        self.save_dir = save_dir
        self.best_accuracy = 0
        self.best_model_path = None
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def save_model(self, model, optimizer, config, epoch, accuracy, label_accuracy, is_best=False):
        """
        保存模型状态
        :param model: 模型实例
        :param optimizer: 优化器实例
        :param config: 配置对象
        :param epoch: 当前训练轮数
        :param accuracy: 当前准确率
        :param label_accuracy: 各标签的准确率
        :param is_best: 是否为最佳模型
        """
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建保存信息
        # 只保存可序列化的config参数
        config_dict = {}
        for k, v in config.__dict__.items():
            try:
                json.dumps(v)
                config_dict[k] = v
            except TypeError:
                pass  # 跳过不可序列化的（如device等）

        # 确定模型类型
        model_type = 'multi' if hasattr(config, 'threshold') else 'single'
        
        save_info = {
            'epoch': epoch,
            'accuracy': accuracy,
            'label_accuracy': label_accuracy.tolist() if isinstance(label_accuracy, torch.Tensor) else label_accuracy,
            'timestamp': timestamp,
            'config': config_dict,
            'model_type': model_type
        }
        
        # 生成文件名，添加模型类型标识
        filename = f'{model_type}_label_model_epoch{epoch}_acc{accuracy:.4f}_{timestamp}'
        if is_best:
            filename = f'best_{filename}'
        
        # 创建保存路径
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'save_info': save_info
        }, os.path.join(save_path, 'model.pth'))
        
        # 保存配置信息
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(save_info, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {save_path}")
        
        # 如果是当前最佳模型，更新最佳模型路径
        if is_best:
            self.best_accuracy = accuracy
            self.best_model_path = save_path
            print(f"新的最佳模型已保存，准确率: {accuracy:.4f}")
    
    def load_model(self, model, optimizer, load_path):
        """
        加载模型状态
        :param model: 模型实例
        :param optimizer: 优化器实例
        :param load_path: 模型保存路径
        :return: 加载的模型信息
        """
        checkpoint = torch.load(os.path.join(load_path, 'model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        save_info = checkpoint['save_info']
        
        print(f"模型已从 {load_path} 加载")
        print(f"加载的模型信息:")
        print(f"训练轮数: {save_info['epoch']}")
        print(f"准确率: {save_info['accuracy']:.4f}")
        print(f"训练时间: {save_info['timestamp']}")
        
        return save_info
    
    def list_saved_models(self):
        """列出所有保存的模型"""
        if not os.path.exists(self.save_dir):
            print("没有找到保存的模型")
            return []
        
        models = []
        for item in os.listdir(self.save_dir):
            if os.path.isdir(os.path.join(self.save_dir, item)):
                config_path = os.path.join(self.save_dir, item, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        models.append({
                            'path': os.path.join(self.save_dir, item),
                            'epoch': config['epoch'],
                            'accuracy': config['accuracy'],
                            'timestamp': config['timestamp'],
                            'model_type': config.get('model_type', 'unknown')
                        })
        
        # 按时间戳排序
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        
        print("\n已保存的模型列表:")
        for i, model in enumerate(models):
            print(f"{i+1}. 路径: {model['path']}")
            print(f"   轮数: {model['epoch']}, 准确率: {model['accuracy']:.4f}")
            print(f"   模型类型: {model['model_type']}")
            print(f"   保存时间: {model['timestamp']}")
            print()
        
        return models
    
    def delete_model(self, model_path):
        """删除指定的模型"""
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            print(f"已删除模型: {model_path}")
        else:
            print(f"未找到模型: {model_path}")
    
    def should_save_model(self, accuracy, threshold=0.0):
        """
        根据准确率判断是否应该保存模型
        :param accuracy: 当前准确率
        :param threshold: 保存阈值，默认保存所有模型
        :return: 是否应该保存
        """
        return accuracy > threshold

class Predictor:
    def __init__(self, model_path):
        """
        初始化预测器
        :param model_path: 模型保存路径
        """
        # 加载配置
        with open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 动态导入所需的模块
        if self.config['model_type'] == 'multi_label':
            from multi_label import BertMultiLabelClassifier, MultiLabelConfig
            self.model_class = BertMultiLabelClassifier
            self.config_class = MultiLabelConfig
        else:
            from single_label import BertClassifier, Config
            self.model_class = BertClassifier
            self.config_class = Config
        
        # 创建配置对象
        self.model_config = self.config_class()
        
        # 设置配置参数
        self.model_config.model_name = self.config['config']['model_name']
        self.model_config.batch_size = self.config['config']['batch_size']
        self.model_config.learning_rate = self.config['config']['learning_rate']
        self.model_config.max_epochs = self.config['config']['max_epochs']
        # 严格使用保存时的max_seq_len
        if 'max_seq_len' in self.config['config']:
            self.model_config.max_seq_len = self.config['config']['max_seq_len']
        
        # 根据模型类型设置特定参数
        if self.config['model_type'] == 'multi_label':
            self.model_config.threshold = self.config['config']['threshold']
        else:
            self.model_config.num_classes = self.config['config']['num_classes']
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config.model_path)
        
        # 初始化模型
        self.model = self.model_class(self.model_config).to(self.model_config.device)
        
        # 加载模型权重
        checkpoint = torch.load(os.path.join(model_path, 'model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"预测器初始化完成，使用模型: {model_path}")
        print(f"模型类型: {self.config['model_type']}")
    
    def predict(self, text, threshold=None):
        """
        对单个文本进行预测
        :param text: 输入文本
        :param threshold: 预测阈值，如果为None则使用模型配置中的阈值
        :return: 预测结果，包含标签和概率
        """
        # 文本预处理
        token = self.tokenizer.tokenize(text)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)
        
        # 填充到最大长度
        if len(token_ids) < self.model_config.max_seq_len:
            padding_length = self.model_config.max_seq_len - len(token_ids)
            mask += [0] * padding_length
            token_ids += [0] * padding_length
            token_type_ids += [0] * padding_length
        else:
            token_ids = token_ids[:self.model_config.max_seq_len]
            mask = mask[:self.model_config.max_seq_len]
            token_type_ids = token_type_ids[:self.model_config.max_seq_len]
        
        # 转换为张量并确保维度一致
        token_ids = torch.LongTensor([token_ids]).to(self.model_config.device)
        mask = torch.LongTensor([mask]).to(self.model_config.device)
        token_type_ids = torch.LongTensor([token_type_ids]).to(self.model_config.device)
        
        # 打印调试信息
        print(f"token_ids shape: {token_ids.shape}")
        print(f"mask shape: {mask.shape}")
        print(f"token_type_ids shape: {token_type_ids.shape}")
        
        # 预测
        with torch.no_grad():
            if self.config['model_type'] == 'multi_label':
                probs = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                if threshold is None:
                    threshold = self.model_config.threshold
                predictions = (probs > threshold).float()
                
                # 整理多标签结果
                results = []
                for i, label in enumerate(self.model_config.label_list):
                    if predictions[0, i].item() == 1:
                        results.append({
                            'label': label,
                            'probability': probs[0, i].item()
                        })
            else:
                logits = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                
                # 整理单标签结果
                results = [{
                    'label': self.model_config.label_list[pred],
                    'probability': probs[0, pred].item()
                }]
        
        return results
    
    def predict_batch(self, texts, threshold=None):
        """
        对多个文本进行批量预测
        :param texts: 输入文本列表
        :param threshold: 预测阈值，如果为None则使用模型配置中的阈值
        :return: 预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict(text, threshold)
            results.append({
                'text': text,
                'predictions': result
            })
        return results 