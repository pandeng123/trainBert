import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.optim as optim  # 使用PyTorch的优化器
import os
import matplotlib.pyplot as plt
import random
import single_label
from single_label import Config, DataProcessor
import argparse

# 配置类需要修改标签数量
class MultiLabelConfig(Config):
    def __init__(self):
        super().__init__()
        self.label_list = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']
        
        # 修改为多标签分类
        self.multi_label = True
        # 阈值 - 概率超过这个值被视为正类
        self.threshold = 0.5
        # 评估参数
        self.do_eval = True  # 是否开启测试集评估

        self.early_stop = 300

        self.num_classes = 3

        self.hidden_size = 768
        self.mid_size = 512  # BERT输出层与分类层之间的隐藏层大小
        self.dropout_rate = 0.3  # Dropout比率
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.max_epochs = 60

# 多标签分类模型类
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 从本地加载模型
        if config.use_local_model and os.path.exists(config.model_path):
            print(f"从本地加载BERT模型: {config.model_path}")
            self.bert = BertModel.from_pretrained(config.model_path)
        else:
            # 从在线加载
            print(f"从在线加载BERT模型: {config.model_name}")
            self.bert = BertModel.from_pretrained(config.model_name)
            
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad_(False)
            
        self.hidden = nn.Sequential(
            nn.Linear(config.hidden_size, config.mid_size),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate)
        )
        # 多标签分类：输出层神经元数量等于标签数量
        self.fc = nn.Linear(config.mid_size, len(config.label_list))
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        # 获取[CLS]的表示
        cls_output = self.hidden(out.last_hidden_state[:, 0])
        # 输出每个标签的概率 (使用sigmoid而不是softmax)
        logits = self.fc(cls_output)
        probs = torch.sigmoid(logits)
        
        return probs

# 计算二元交叉熵损失的函数（逐元素）
def compute_element_wise_bce(probs, targets):
    # 对每个预测值计算损失: -[y*log(p) + (1-y)*log(1-p)]
    losses = -targets * torch.log(probs + 1e-10) - (1 - targets) * torch.log(1 - probs + 1e-10)
    return losses

def draw(iterations, loss_values, accuracy_values):
    # 绘制损失和准确率图形
    plt.figure(figsize=(12, 5))
    
    if len(loss_values) != 0:
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(iterations, loss_values, 'r-', label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracy_values, 'b-', label='Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('multilabel_training_metrics.png')
    plt.show()

# 多标签训练器类
class MultiLabelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 使用PyTorch的优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        # 使用BCEWithLogitsLoss或自己实现的BCE
        self.criterion = nn.BCELoss()
        # 初始化模型管理器
        from model_manager import ModelManager
        self.model_manager = ModelManager()
        
    def train(self, train_iter, epochs=1):
        """
        训练模型
        train_iter: 训练数据迭代器
        epochs: 训练轮数，默认为1
        """
        self.model.train()  # 开启训练模式
        
        # 创建列表存储训练指标
        iterations = []
        loss_values = []
        accuracy_values = []
        
        # 记录实际处理的样本数
        total_samples_processed = 0
        total_loss = 0
        total_batches = 0
        global_step = 0
        
        # 多轮训练
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")
            
            epoch_loss = 0
            epoch_samples = 0
            epoch_correct_preds = 0
            
            for i, (token_ids, label_1, label_2, seq_len, mask, token_type_ids) in enumerate(train_iter):
                # 确保所有张量在正确的设备上
                token_ids = token_ids.to(self.config.device)
                label_1 = label_1.to(self.config.device)
                mask = mask.to(self.config.device)
                token_type_ids = token_type_ids.to(self.config.device)
                
                # 累加当前批次的样本数
                batch_size = token_ids.size(0)
                total_samples_processed += batch_size
                epoch_samples += batch_size
                
                # 直接使用label_1作为多标签目标
                # label_1已经是one-hot格式: [batch_size, num_labels]
                multi_label_targets = label_1
                
                # 前向传播
                probs = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                
                # 计算损失
                loss = self.criterion(probs, multi_label_targets)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                epoch_loss += loss.item() * batch_size
                total_batches += 1
                
                # 计算准确率
                predictions = (probs > self.config.threshold).float()
                correct = (predictions == multi_label_targets).float().mean(dim=1).sum().item()
                epoch_correct_preds += correct
                
                # 打印训练进度
                if global_step % self.config.print_step == 0:
                    # 计算每个标签的准确率
                    accuracy = (predictions == multi_label_targets).float().mean().item()
                    
                    # 记录训练指标
                    iterations.append(global_step)
                    loss_values.append(loss.item())
                    accuracy_values.append(accuracy)
                    
                    # 使用compute_element_wise_bce计算每个标签的损失
                    element_losses = compute_element_wise_bce(probs, multi_label_targets)
                    avg_element_loss = element_losses.mean().item()
                    
                    print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, Avg Element Loss: {avg_element_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                global_step += 1
            
            # 计算并显示当前epoch的统计信息
            epoch_accuracy = epoch_correct_preds / epoch_samples if epoch_samples > 0 else 0
            epoch_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            print(f"Epoch {epoch+1} 完成: Loss={epoch_avg_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
            
        # 保存当前模型
        if self.model_manager.should_save_model(epoch_accuracy):
            self.model_manager.save_model(
                self.model,
                self.optimizer,
                self.config,
                epoch + 1,
                epoch_accuracy,
                torch.zeros(len(self.config.label_list)),  # 这里使用零向量，因为训练时没有标签级别的准确率
                is_best=(epoch_accuracy > self.model_manager.best_accuracy)
            )
        
        # 绘制训练指标图形
        draw(iterations, loss_values, accuracy_values)
        
        print(f"训练完成! 共训练 {epochs} 个epoch")
        
        # 返回平均损失和实际处理的样本数
        return total_loss / total_batches if total_batches > 0 else 0, total_samples_processed

    def evaluate(self, data_iter):
        """评估模型在测试集上的性能"""
        print("开始评估模型...")
        self.model.eval()
        
        total_loss = 0
        label_correct = torch.zeros(len(self.config.label_list)).to(self.config.device)
        label_total = torch.zeros(len(self.config.label_list)).to(self.config.device)
        
        with torch.no_grad():
            for token_ids, label_1, label_2, seq_len, mask, token_type_ids in tqdm(data_iter):
                # 确保所有张量在正确的设备上
                token_ids = token_ids.to(self.config.device)
                label_1 = label_1.to(self.config.device)
                mask = mask.to(self.config.device)
                token_type_ids = token_type_ids.to(self.config.device)
                
                # 直接使用label_1作为多标签目标
                multi_label_targets = label_1
                
                # 前向传播
                probs = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                
                # 计算损失
                loss = self.criterion(probs, multi_label_targets)
                total_loss += loss.item()
                
                # 计算每个标签的准确率
                predictions = (probs > self.config.threshold).float()
                
                # 累计每个标签的正确预测和总数
                for j in range(len(self.config.label_list)):
                    mask_j = multi_label_targets[:, j] > 0  # 找出真实标签为1的样本
                    if mask_j.sum() > 0:
                        label_correct[j] += (predictions[:, j][mask_j] == multi_label_targets[:, j][mask_j]).sum().item()
                        label_total[j] += mask_j.sum().item()
        
        # 计算每个标签的准确率
        label_accuracy = torch.zeros_like(label_correct)
        for j in range(len(self.config.label_list)):
            if label_total[j] > 0:
                label_accuracy[j] = label_correct[j] / label_total[j]
            
        mean_accuracy = label_accuracy.mean().item()
        
        # 输出评估结果
        print(f"测试集评估结果:")
        print(f"平均准确率: {mean_accuracy:.4f}")
        
        return total_loss / len(data_iter), mean_accuracy, label_accuracy


# 主函数
def main_multi_label(load_model_path=None):
    # 1. 初始化配置
    config = MultiLabelConfig()
    
    # 2. 数据处理
    processor = DataProcessor(config)
    train_data = processor.load_dataset(config.train_path, config.max_seq_len)
    
    # 创建训练数据迭代器，启用随机打乱
    train_iter = processor.get_dataloader(train_data, config.batch_size, config.device, shuffle=True)
    
    # 加载测试数据(如果需要评估)
    test_data = None
    test_iter = None
    if config.do_eval:
        test_data = processor.load_dataset(config.test_path, config.max_seq_len)
        test_iter = processor.get_dataloader(test_data, config.batch_size, config.device, shuffle=False)
        print(f"训练集样本数: {len(train_data)}")
        print(f"测试集样本数: {len(test_data)}")
    else:
        print(f"训练集样本数: {len(train_data)}")
        print("测试集评估已关闭")
    
    # 展示一个批次的数据示例
    for i, (token_ids, label_1, label_2, seq_len, mask, token_type_ids) in enumerate(train_iter):
        print("数据示例:")
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Label_1 shape: {label_1.shape}")
        print(f"Label_2 shape: {label_2.shape}")
        break
    
    # 3. 模型初始化 - 使用多标签分类模型
    model = BertMultiLabelClassifier(config).to(config.device)
    
    # 4. 训练模型 - 使用多标签训练器
    trainer = MultiLabelTrainer(model, config)
    
    # 如果提供了模型路径，则加载模型
    if load_model_path:
        print(f"\n正在加载模型: {load_model_path}")
        trainer.model_manager.load_model(model, trainer.optimizer, load_model_path)
    
    # 5. 训练模型
    avg_loss, actual_samples_processed = trainer.train(
        train_iter=train_iter, 
        epochs=config.max_epochs
    )
    
    # 6. 评估模型(如果启用)
    if config.do_eval and test_iter is not None:
        print("\n======== 开始测试集评估 ========")
        loss, accuracy, label_accuracy = trainer.evaluate(test_iter)
        print(f"训练完成，平均损失: {avg_loss:.4f}，共处理了{actual_samples_processed}个样本（数据集总大小：{len(train_data)}）")
        print(f"测试集评估 - 损失: {loss:.4f}, 平均准确率: {accuracy:.4f}")
        print("各标签准确率:")
        for i, label in enumerate(config.label_list):
            print(f"{label}: {label_accuracy[i].item():.4f}")
    else:
        print(f"训练完成，平均损失: {avg_loss:.4f}，共处理了{actual_samples_processed}个样本（数据集总大小：{len(train_data)}）")
        print("未进行测试集评估")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多标签分类模型训练和评估')
    parser.add_argument('--load', type=str, help='要加载的模型路径')
    args = parser.parse_args()
    
    main_multi_label(load_model_path=args.load)