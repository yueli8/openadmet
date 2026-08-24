"""
02_test.py - 测试ADMET-AI是否安装成功
运行：python 02_test.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用CPU

import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)

from admet_ai import ADMETModel

def main():
    print("=" * 50)
    print("ADMET-AI 安装测试")
    print("=" * 50)
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {torch.__version__}")
    
    # 加载模型
    print("\n正在加载模型...")
    model = ADMETModel()
    print("✓ 模型加载成功！")
    
    # 测试预测
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    print(f"\n测试化合物: {smiles}")
    results = model.predict(smiles)
    
    print("\n预测结果（部分）:")
    print(f"  分子量: {results['molecular_weight']:.2f}")
    print(f"  logP: {results['logP']:.2f}")
    print(f"  氢键供体: {results['hydrogen_bond_donors']:.0f}")
    print(f"  氢键受体: {results['hydrogen_bond_acceptors']:.0f}")
    print(f"  拓扑极性表面积: {results['tpsa']:.2f}")
    
    print("\n✓ 测试成功！ADMET-AI可以正常使用。")

if __name__ == "__main__":
    main()
