import torch
from tqdm import tqdm
import hashlib
import json
import sys
import io
from single_label import Config, DataProcessor

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_batch_hash(batch):
    """计算一个批次数据的哈希值"""
    # 将批次数据转换为字符串
    batch_str = str([item.tolist() for item in batch])
    return hashlib.sha256(batch_str.encode()).hexdigest()

def verify_training_data():
    # 初始化配置
    config = Config()
    processor = DataProcessor(config)
    
    # 加载训练数据
    print("正在加载训练数据...")
    train_data = processor.load_dataset(config.train_path, config.max_seq_len)
    train_loader = processor.get_dataloader(train_data, config.batch_size, config.device, shuffle=True)
    
    # 记录每个epoch的批次哈希值
    epochs = 3  # 验证3个epoch
    batch_hashes = {}
    
    print(f"\n开始验证{epochs}个epoch的训练数据随机性...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}:")
        epoch_hashes = []
        
        # 遍历每个批次
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # 计算批次哈希值
            batch_hash = calculate_batch_hash(batch)
            epoch_hashes.append(batch_hash)
            
            # 打印前几个批次的哈希值
            if batch_idx < 3:
                print(f"批次 {batch_idx + 1} 哈希值: {batch_hash}")
        
        batch_hashes[f"epoch_{epoch + 1}"] = epoch_hashes
    
    # 比较不同epoch的批次顺序
    print("\n比较不同epoch的批次顺序...")
    for i in range(epochs - 1):
        epoch1 = f"epoch_{i + 1}"
        epoch2 = f"epoch_{i + 2}"
        
        # 比较两个epoch的批次哈希值
        matches = sum(1 for h1, h2 in zip(batch_hashes[epoch1], batch_hashes[epoch2]) if h1 == h2)
        total_batches = len(batch_hashes[epoch1])
        
        print(f"\n比较 {epoch1} 和 {epoch2}:")
        print(f"总批次数量: {total_batches}")
        print(f"相同批次数量: {matches}")
        print(f"批次重复率: {matches/total_batches:.2%}")
        
        if matches > 0:
            print("警告：发现重复的批次！")
        else:
            print("验证通过：没有发现重复的批次")
    
    # 保存哈希值到文件
    with open('training_data_hashes.json', 'w', encoding='utf-8') as f:
        json.dump(batch_hashes, f, ensure_ascii=False, indent=2)
    print("\n批次哈希值已保存到 training_data_hashes.json")

if __name__ == "__main__":
    verify_training_data() 