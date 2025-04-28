from model_manager import ModelManager, Predictor
import argparse

def main():
    parser = argparse.ArgumentParser(description='多标签分类模型预测')
    parser.add_argument('--model', type=str, required=True, help='要使用的模型路径')
    parser.add_argument('--text', type=str, help='要预测的文本')
    parser.add_argument('--file', type=str, help='包含多行文本的文件路径')
    parser.add_argument('--threshold', type=float, help='预测阈值')
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = Predictor(args.model)
    
    # 单个文本预测
    if args.text:
        results = predictor.predict(args.text, args.threshold)
        print("\n预测结果:")
        print(f"文本: {args.text}")
        for result in results:
            print(f"标签: {result['label']}, 概率: {result['probability']:.4f}")
    
    # 批量预测
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts, args.threshold)
        print("\n批量预测结果:")
        for result in results:
            print(f"\n文本: {result['text']}")
            for pred in result['predictions']:
                print(f"标签: {pred['label']}, 概率: {pred['probability']:.4f}")

if __name__ == "__main__":
    main() 