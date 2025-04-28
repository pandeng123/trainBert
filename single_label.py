import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import random
import hashlib
from torch.utils.data import Dataset, DataLoader

# 配置类：存储所有配置参数
class Config:
    def __init__(self):
        # 模型参数
        self.model_name = 'bert-base-chinese'
        self.model_path = './bert-base-chinese'  # 本地模型路径
        self.use_local_model = True  # 使用本地模型
        self.hidden_size = 768
        self.mid_size = 512  # BERT输出层与分类层之间的隐藏层大小
        self.dropout_rate = 0.5  # Dropout比率
        self.num_classes = 3
        
        # 训练参数
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.max_epochs = 1
        
        self.max_seq_len = 64
        self.print_step = 5
        self.early_stop = 60
        
        # 数据加载参数
        self.num_workers = 2  # 数据加载的工作进程数
        
        # 数据参数
        self.train_path = "./data/train.txt"
        self.test_path = "./data/test.txt"  # 添加测试数据路径
        
        # 评估参数
        self.do_eval = False  # 是否开启测试集评估
        
        # 特殊标记
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        
        # 标签列表
        self.label_list = ['动力', '价格', '内饰', '配置', '安全性', '外观', '操控', '油耗', '空间', '舒适性']
        self.label2idx = {label: idx for idx, label in enumerate(self.label_list)}
        self.idx2label = {idx: label for idx, label in enumerate(self.label_list)}
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类，符合PyTorch Dataset接口
class BertDataset(Dataset):
    def __init__(self, contents, device):
        self.contents = contents
        self.device = device
        
    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, idx):
        item = self.contents[idx]
        
        # 转换数据为张量
        token_ids = torch.LongTensor(item[0])
        multi_labels = torch.FloatTensor(item[1])
        
        # 标签映射：-1 -> 2, 0 -> 0, 1 -> 1
        label_map = {-1: 2, 0: 0, 1: 1}
        label_2 = torch.LongTensor([label_map[item[2]]])
        
        seq_len = torch.LongTensor([item[3]])
        mask = torch.LongTensor(item[4])
        token_type_ids = torch.LongTensor(item[5])
        
        return token_ids, multi_labels, label_2.squeeze(), seq_len, mask, token_type_ids


# 数据处理类：负责数据加载、处理和迭代
class DataProcessor:
    def __init__(self, config):
        self.config = config
        
        # 从本地加载模型
        if config.use_local_model and os.path.exists(config.model_path):
            print(f"从本地加载模型: {config.model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(config.model_path)
        else:
            # 从在线加载
            print(f"从在线加载模型: {config.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    def calculate_dataset_hash(self, contents):
        """计算数据集的哈希值"""
        # 将数据集内容转换为字符串
        content_str = str(contents)
        # 计算SHA-256哈希值
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def load_dataset(self, path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                if not line:
                    continue
                parts = line.strip().split('\t')
                content = parts[0]
                
                # 获取情感标签 (label_2)
                _, label_2 = parts[1].split('#')
                
                # 创建多标签向量，初始全为0
                multi_labels = [0] * len(self.config.label_list)
                
                # 遍历每个标签部分
                for label_part in parts[1:]:
                    label, _ = label_part.split('#')
                    # 获取标签索引并置为1（表示存在该标签）
                    label_idx = self.config.label2idx[label]
                    multi_labels[label_idx] = 1
                
                # 其他处理保持不变
                token = self.tokenizer.tokenize(content)
                token = [self.config.cls_token] + token
                seq_len = len(token)
                mask = []
                token_ids = self.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                token_type_ids = [0] * len(token_ids)
                
                # 保留原有的label_2，同时加入多标签向量
                contents.append((token_ids, multi_labels, int(label_2), seq_len, mask, token_type_ids))
        
        # 计算并打印数据集哈希值
        dataset_hash = self.calculate_dataset_hash(contents)
        print(f"数据集哈希值: {dataset_hash}")
        
        # 保存哈希值到文件
        hash_file = path + '.hash'
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                saved_hash = f.read().strip()
            if saved_hash != dataset_hash:
                print("警告：数据集哈希值不匹配！")
                print(f"保存的哈希值: {saved_hash}")
                print(f"当前哈希值: {dataset_hash}")
            else:
                print("数据集验证通过：哈希值匹配")
        else:
            with open(hash_file, 'w') as f:
                f.write(dataset_hash)
            print("已保存新的数据集哈希值")
        
        return contents
    
    def get_dataloader(self, data, batch_size, device, shuffle=True):
        """创建PyTorch DataLoader"""
        dataset = BertDataset(data, device)
        
        # 设置是否启用pin_memory（当使用GPU时建议启用）
        pin_memory = device.type == 'cuda'
        
        # 创建DataLoader
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            # 注意: PyTorch的DataLoader已经处理batch内的张量转移，
            # 所以我们不需要在__getitem__中手动将数据移至device
        )


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
    plt.savefig('training_metrics.png')
    plt.show()

# 模型类：定义网络结构
class BertClassifier(nn.Module):
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
        self.fc = nn.Linear(config.mid_size, config.num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        out = self.fc(self.hidden(out.last_hidden_state[:, 0]))
        return out

# 训练器类：负责模型训练和评估
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
    def train(self, train_loader, epochs=1):
        """
        训练模型
        train_loader: 训练数据加载器(PyTorch DataLoader)
        epochs: 训练轮数，默认为1
        """
        self.model.train()  # 开启训练模式

        # 创建列表存储训练指标
        iterations = []
        loss_values = []
        accuracy_values = []
        
        # 记录实际处理的样本数
        total_samples_processed = 0
        best_accuracy = 0

        # 多轮训练
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")
            
            # 设置新的随机种子，使每个epoch的洗牌不同
            # (DataLoader在每个epoch开始时不会自动重新洗牌)
            if hasattr(train_loader.sampler, 'set_epoch'):  
                # 分布式训练时使用
                train_loader.sampler.set_epoch(epoch)
                
            epoch_loss = 0
            epoch_samples = 0
            epoch_correct = 0
            
            for i, (token_ids, label_1, label_2, seq_len, mask, token_type_ids) in enumerate(train_loader):
                # 将数据移至设备（GPU/CPU）
                token_ids = token_ids.to(self.config.device)
                label_2 = label_2.to(self.config.device)
                mask = mask.to(self.config.device)
                token_type_ids = token_type_ids.to(self.config.device)
                
                # 累加当前批次的样本数
                batch_size = token_ids.size(0)
                total_samples_processed += batch_size
                epoch_samples += batch_size
                
                # 先清零梯度
                self.optimizer.zero_grad()
                
                out = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                loss = self.criterion(out, label_2)
                loss.backward()
                self.optimizer.step()
                
                # 计算准确率
                pred = out.argmax(dim=1)
                correct = (pred == label_2).sum().item()
                epoch_correct += correct
                epoch_loss += loss.item() * batch_size
                
                if i % self.config.print_step == 0:
                    accuracy = (pred == label_2).sum().item() / len(label_2)
                    print(f"Batch {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

                    # 记录训练指标
                    iterations.append(i + epoch * len(train_loader))
                    loss_values.append(loss.item())
                    accuracy_values.append(accuracy)
                
                # 如果到达early_stop，则提前结束当前epoch
                # if i >= self.config.early_stop:
                #     print(f"到达early_stop({self.config.early_stop})，提前结束当前epoch")
                #     break
            
            # 计算并显示当前epoch的统计信息
            epoch_accuracy = epoch_correct / epoch_samples
            epoch_avg_loss = epoch_loss / epoch_samples
            print(f"Epoch {epoch+1} 完成: Loss={epoch_avg_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
            
            # 记录最佳准确率
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                print(f"*** 新的最佳准确率: {best_accuracy:.4f} ***")
                
            # 更新学习率
            # self.scheduler.step(epoch_accuracy)
        
        # 绘制训练指标图形
        draw(iterations, loss_values, accuracy_values)
        
        print(f"训练完成! 共训练 {epochs} 个epoch, 最佳准确率: {best_accuracy:.4f}")
        
        # 返回实际处理的样本数和最佳准确率
        return total_samples_processed, best_accuracy
    
    def evaluate(self, test_loader):
        """评估模型在测试集上的性能"""
        print("开始评估模型...")
        self.model.eval()  # 切换到评估模式
        
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for token_ids, label_1, label_2, seq_len, mask, token_type_ids in tqdm(test_loader):
                # 将数据移至设备（GPU/CPU）
                token_ids = token_ids.to(self.config.device)
                label_2 = label_2.to(self.config.device)
                mask = mask.to(self.config.device)
                token_type_ids = token_type_ids.to(self.config.device)
                
                out = self.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                
                # 获取预测结果
                preds = out.argmax(dim=1)
                
                # 统计正确预测数
                correct = (preds == label_2).sum().item()
                total_correct += correct
                total_samples += len(label_2)
                
                # 收集所有预测和标签，用于计算其他指标
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_2.cpu().numpy())
        
        # 计算总体准确率
        accuracy = total_correct / total_samples
        
        # 输出评估结果
        print(f"测试集评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"样本数: {total_samples}")
        
        return accuracy, all_preds, all_labels

# 主函数：组织整个流程
def main():
    # 1. 初始化配置
    config = Config()
    
    # 显示当前使用的设备信息
    print(f"当前使用设备: {config.device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备索引: {torch.cuda.current_device()}")
    
    # 2. 数据处理
    processor = DataProcessor(config)
    train_data = processor.load_dataset(config.train_path, config.max_seq_len)
    train_loader = processor.get_dataloader(train_data, config.batch_size, config.device, shuffle=True)
    
    # 加载测试数据(如果需要评估)
    test_data = None
    test_loader = None
    if config.do_eval:
        test_data = processor.load_dataset(config.test_path, config.max_seq_len)
        test_loader = processor.get_dataloader(test_data, config.batch_size, config.device, shuffle=False)
        print(f"训练集样本数: {len(train_data)}")
        print(f"测试集样本数: {len(test_data)}")
    else:
        print(f"训练集样本数: {len(train_data)}")
        print("测试集评估已关闭")
    
    # 展示一个批次的数据
    for i, (token_ids, label_1, label_2, seq_len, mask, token_type_ids) in enumerate(train_loader):
        print("数据样例:")
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Token Type IDs shape: {token_type_ids.shape}")
        print(f"Label_1 shape: {label_1.shape}")
        print(f"Label_2 shape: {label_2.shape}")
        print(f"Seq_len shape: {seq_len.shape}")
        break
    
    # 3. 模型初始化
    model = BertClassifier(config).to(config.device)
    
    # 4. 训练模型 - 使用config.max_epochs参数控制训练多少轮
    trainer = Trainer(model, config)
    actual_samples_processed, best_train_accuracy = trainer.train(train_loader, epochs=config.max_epochs)
    
    # 5. 评估模型(如果启用)
    if config.do_eval and test_loader is not None:
        print("\n======== 开始测试集评估 ========")
        accuracy, _, _ = trainer.evaluate(test_loader)
        print(f"训练完成，共处理了{actual_samples_processed}个样本（数据集总大小：{len(train_data)}）")
        print(f"最佳训练准确率: {best_train_accuracy:.4f}")
        print(f"测试集准确率: {accuracy:.4f}")
    else:
        print(f"训练完成，共处理了{actual_samples_processed}个样本（数据集总大小：{len(train_data)}）")
        print(f"最佳训练准确率: {best_train_accuracy:.4f}")
        print("未进行测试集评估")


if __name__ == "__main__":
    main()

