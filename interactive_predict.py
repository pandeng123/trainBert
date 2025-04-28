from model_manager import Predictor
import argparse

def main():
    parser = argparse.ArgumentParser(description='多标签分类模型交互式预测')
    parser.add_argument('--model', type=str, required=True, help='要使用的模型路径')
    parser.add_argument('--threshold', type=float, help='预测阈值')
    args = parser.parse_args()
    
    # 初始化预测器
    print("正在加载模型...")
    predictor = Predictor(args.model)
    print("模型加载完成！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'threshold' 查看当前阈值")
    print("输入 'set_threshold 0.5' 设置新的阈值")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n请输入要预测的文本: ").strip()
            
            if text.lower() in ['quit', 'exit']:
                print("再见！")
                break
                
            elif text.lower() == 'threshold':
                print(f"当前阈值: {predictor.model_config.threshold}")
                continue
                
            elif text.startswith('set_threshold'):
                try:
                    new_threshold = float(text.split()[1])
                    if 0 <= new_threshold <= 1:
                        predictor.model_config.threshold = new_threshold
                        print(f"阈值已更新为: {new_threshold}")
                    else:
                        print("阈值必须在0到1之间")
                except:
                    print("设置阈值失败，请使用格式: set_threshold 0.5")
                continue
                
            if not text:
                print("输入不能为空！")
                continue
                
            # 进行预测
            results = predictor.predict(text, args.threshold)
            
            # 打印结果
            print("\n预测结果:")
            print(f"文本: {text}")
            for result in results:
                print(f"标签: {result['label']}, 概率: {result['probability']:.4f}")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    main() 