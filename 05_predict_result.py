#!/usr/bin/env python3
"""
ADMET 预测工具 - 综合评估与排序版本（评估详情分列）
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ========== 关键修复：必须在导入 ADMETModel 之前 ==========
import torch
import functools
import argparse

# 修复 PyTorch 2.6 的 weights_only 问题
torch.load = functools.partial(torch.load, weights_only=False)

# 添加 argparse.Namespace 到安全全局变量
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")

# ========== 现在才导入 ADMETModel ==========
from admet_ai import ADMETModel
import pandas as pd
import numpy as np
import re

# 定义关键属性及其中文注释
KEY_PROPERTIES = {
    'molecular_weight': '分子量',
    'logP': '脂水分配系数',
    'hydrogen_bond_acceptors': '氢键受体数',
    'hydrogen_bond_donors': '氢键供体数',
    'Lipinski': 'Lipinski五规则',
    'QED': '药物相似性',
    'tpsa': '拓扑极性表面积',
    'AMES': '致突变性',
    'BBB_Martins': '血脑屏障穿透性',
    'Bioavailability_Ma': '生物利用度',
    'HIA_Hou': '人体肠道吸收',
    'Caco2_Wang': 'Caco-2细胞渗透性',
    'hERG': '心脏毒性风险',
    'CYP3A4_Veith': 'CYP3A4酶抑制',
    'CYP2D6_Veith': 'CYP2D6酶抑制',
    'CYP2C9_Veith': 'CYP2C9酶抑制',
    'CYP1A2_Veith': 'CYP1A2酶抑制',
    'Clearance_Hepatocyte_AZ': '肝细胞清除率',
    'Half_Life_Obach': '半衰期',
    'Solubility_AqSolDB': '水溶性'
}

# 初始化模型
print("正在加载ADMET-AI模型...")
model = ADMETModel()
print("✓ 模型加载成功")

def read_input_file(input_file='input.txt'):
    """读取输入文件，解析SMILES和名称"""
    compounds = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过第一行（表头）
    for i, line in enumerate(lines):
        if i == 0 and ('SMILES' in line or 'smiles' in line):
            continue
        
        if line.strip():
            # 使用Tab或多个空格分割
            parts = re.split(r'\t+|\s{2,}', line.strip())
            if len(parts) >= 2:
                smiles = parts[0].strip()
                name = ' '.join(parts[1:]).strip()
                compounds.append((name, smiles))
    
    print(f"✓ 从 {input_file} 读取了 {len(compounds)} 个化合物")
    return compounds

def calculate_comprehensive_score(df):
    """计算综合评估分数和详情（分列）"""
    scores = []
    
    for idx, row in df.iterrows():
        score = 0
        evaluation = {}
        
        # 1. 分子量评分 (理想: 150-500)
        mw = row.get('molecular_weight', 0)
        if 150 <= mw <= 500:
            score += 10
            evaluation['分子量评估'] = f"✓ 适宜({mw:.1f})"
        elif 100 <= mw <= 600:
            score += 5
            evaluation['分子量评估'] = f"△ 可接受({mw:.1f})"
        else:
            evaluation['分子量评估'] = f"✗ 不佳({mw:.1f})"
        
        # 2. logP评分 (理想: 0-3)
        logp = row.get('logP', 0)
        if 0 <= logp <= 3:
            score += 10
            evaluation['logP评估'] = f"✓ 适宜({logp:.2f})"
        elif -2 <= logp <= 5:
            score += 5
            evaluation['logP评估'] = f"△ 可接受({logp:.2f})"
        else:
            evaluation['logP评估'] = f"✗ 不佳({logp:.2f})"
        
        # 3. Lipinski规则评分
        lipinski = row.get('Lipinski', 0)
        if lipinski >= 4:
            score += 15
            evaluation['Lipinski评估'] = f"✓ 符合({lipinski:.0f}/5)"
        elif lipinski >= 3:
            score += 10
            evaluation['Lipinski评估'] = f"△ 基本符合({lipinski:.0f}/5)"
        elif lipinski >= 2:
            score += 5
            evaluation['Lipinski评估'] = f"△ 部分符合({lipinski:.0f}/5)"
        else:
            evaluation['Lipinski评估'] = f"✗ 较差({lipinski:.0f}/5)"
        
        # 4. QED药物相似性评分
        qed = row.get('QED', 0)
        if qed >= 0.7:
            score += 15
            evaluation['QED评估'] = f"✓ 高({qed:.3f})"
        elif qed >= 0.5:
            score += 10
            evaluation['QED评估'] = f"△ 中等({qed:.3f})"
        elif qed >= 0.3:
            score += 5
            evaluation['QED评估'] = f"△ 一般({qed:.3f})"
        else:
            evaluation['QED评估'] = f"✗ 低({qed:.3f})"
        
        # 5. 致突变性评分 (AMES: 越低越好)
        ames = row.get('AMES', 0.5)
        if ames < 0.3:
            score += 10
            evaluation['致突变性评估'] = f"✓ 低({ames:.3f})"
        elif ames < 0.5:
            score += 5
            evaluation['致突变性评估'] = f"△ 中等({ames:.3f})"
        else:
            evaluation['致突变性评估'] = f"✗ 高({ames:.3f})"
        
        # 6. 生物利用度评分
        bioavail = row.get('Bioavailability_Ma', 0)
        if bioavail >= 0.7:
            score += 10
            evaluation['生物利用度评估'] = f"✓ 高({bioavail:.3f})"
        elif bioavail >= 0.5:
            score += 5
            evaluation['生物利用度评估'] = f"△ 中等({bioavail:.3f})"
        else:
            evaluation['生物利用度评估'] = f"✗ 低({bioavail:.3f})"
        
        # 7. 人体肠道吸收评分
        hia = row.get('HIA_Hou', 0)
        if hia >= 0.7:
            score += 10
            evaluation['肠道吸收评估'] = f"✓ 好({hia:.3f})"
        elif hia >= 0.5:
            score += 5
            evaluation['肠道吸收评估'] = f"△ 中等({hia:.3f})"
        else:
            evaluation['肠道吸收评估'] = f"✗ 差({hia:.3f})"
        
        # 8. hERG心脏毒性评分 (越低越好)
        herg = row.get('hERG', 0.5)
        if herg < 0.3:
            score += 10
            evaluation['心脏毒性评估'] = f"✓ 低({herg:.3f})"
        elif herg < 0.5:
            score += 5
            evaluation['心脏毒性评估'] = f"△ 中等({herg:.3f})"
        else:
            evaluation['心脏毒性评估'] = f"✗ 高({herg:.3f})"
        
        # 9. 水溶性评分
        solubility = row.get('Solubility_AqSolDB', 0)
        if solubility > -2:
            score += 10
            evaluation['水溶性评估'] = f"✓ 好({solubility:.2f})"
        elif solubility > -4:
            score += 5
            evaluation['水溶性评估'] = f"△ 中等({solubility:.2f})"
        else:
            evaluation['水溶性评估'] = f"✗ 差({solubility:.2f})"
        
        # 10. CYP酶抑制评分 (越低越好)
        cyp_props = ['CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith']
        cyp_inhibited = []
        cyp_score = 0
        for prop in cyp_props:
            val = row.get(prop, 0.5)
            if val < 0.5:
                cyp_score += 2.5
                cyp_inhibited.append(prop.replace('_Veith', ''))
        
        if cyp_inhibited:
            evaluation['CYP酶抑制评估'] = f"△ 抑制: {', '.join(cyp_inhibited)}"
        else:
            evaluation['CYP酶抑制评估'] = f"✓ 无明显抑制"
        score += cyp_score
        
        # 计算总分
        total_score = min(score, 100)  # 最高100分
        
        # 评级
        if total_score >= 80:
            grade = "优秀"
        elif total_score >= 65:
            grade = "良好"
        elif total_score >= 50:
            grade = "中等"
        elif total_score >= 35:
            grade = "较差"
        else:
            grade = "差"
        
        # 构建结果字典
        result = {
            'Name': row['Name'],
            'SMILES': row['SMILES'],
            '综合评分': total_score,
            '评级': grade
        }
        # 添加评估详情
        result.update(evaluation)
        scores.append(result)
    
    return pd.DataFrame(scores)

# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("ADMET-AI 预测工具 - 综合评估与排序版")
    print("=" * 60)
    
    # 读取化合物
    compounds = read_input_file('input.txt')
    
    if not compounds:
        print("✗ 文件中没有找到有效的化合物数据")
        exit(1)
    
    # 分离名称和SMILES
    names = [c[0] for c in compounds]
    smiles_list = [c[1] for c in compounds]
    
    # 批量预测
    print(f"\n批量预测 {len(smiles_list)} 个化合物...")
    results = model.predict(smiles_list)
    
    # 创建完整结果表
    full_df = pd.DataFrame(results)
    full_df.insert(0, 'Name', names)
    full_df.insert(1, 'SMILES', smiles_list)
    
    # 保存完整结果
    full_df.to_csv('admet_results_full.csv', index=False)
    print(f"✓ 完整结果已保存到: admet_results_full.csv")
    
    # 创建关键属性结果表
    key_columns = ['Name', 'SMILES']
    available_properties = []
    
    for prop, chinese_name in KEY_PROPERTIES.items():
        if prop in full_df.columns:
            key_columns.append(prop)
            available_properties.append((prop, chinese_name))
    
    key_df = full_df[key_columns]
    
    # 计算综合评分
    print("\n计算综合评估分数...")
    score_df = calculate_comprehensive_score(key_df)
    
    # 合并关键属性和评分
    final_df = key_df.merge(score_df, on=['Name', 'SMILES'], how='left')
    
    # 按综合评分排序
    final_df = final_df.sort_values('综合评分', ascending=False)
    final_df = final_df.reset_index(drop=True)
    final_df.index = final_df.index + 1  # 排名从1开始
    final_df.index.name = '排名'
    
    # 保存完整排序结果
    final_df.to_csv('admet_results_ranked.csv')
    print(f"✓ 排序结果已保存到: admet_results_ranked.csv")
    
    # 创建评估详情文件 (admet_result01.csv)
    eval_columns = ['Name', 'SMILES', '综合评分', '评级',
                    '分子量评估', 'logP评估', 'Lipinski评估', 'QED评估',
                    '致突变性评估', '生物利用度评估', '肠道吸收评估',
                    '心脏毒性评估', '水溶性评估', 'CYP酶抑制评估']
    
    detail_df = final_df[eval_columns].copy()
    detail_df = detail_df.reset_index()
    detail_df = detail_df.rename(columns={
        'Name': '化合物名称',
        'SMILES': 'SMILES结构',
        '综合评分': '综合评分',
        '评级': '评级'
    })
    detail_df.to_csv('admet_result01.csv', index=False)
    print(f"✓ 评估详情已保存到: admet_result01.csv")
    
    # 创建带中文注释的关键属性结果
    chinese_df = final_df.drop(['分子量评估', 'logP评估', 'Lipinski评估', 'QED评估',
                                '致突变性评估', '生物利用度评估', '肠道吸收评估',
                                '心脏毒性评估', '水溶性评估', 'CYP酶抑制评估'], axis=1).copy()
    rename_dict = {'Name': '化合物名称', 'SMILES': 'SMILES结构', '综合评分': '综合评分', '评级': '评级'}
    for prop, chinese_name in available_properties:
        if prop in chinese_df.columns:
            rename_dict[prop] = f"{chinese_name}({prop})"
    
    chinese_df = chinese_df.rename(columns=rename_dict)
    chinese_df.to_csv('admet_results_chinese.csv')
    print(f"✓ 中文注释结果已保存到: admet_results_chinese.csv")
    
    # 显示排序结果
    print("\n" + "=" * 80)
    print("ADMET 综合评估排序结果")
    print("=" * 80)
    
    # 创建简洁的汇总表
    summary_df = final_df[['Name', '综合评分', '评级']].copy()
    summary_df['综合评分'] = summary_df['综合评分'].round(1)
    
    print("\n排名汇总：")
    print(summary_df.to_string())
    
    # 显示每个化合物的评估详情
    print("\n" + "=" * 80)
    print("化合物评估详情")
    print("=" * 80)
    
    for idx, row in final_df.iterrows():
        print(f"\n第{idx}名 - {row['Name']}")
        print(f"  综合评分: {row['综合评分']:.1f}/100")
        print(f"  评级: {row['评级']}")
        print(f"  分子量: {row['分子量评估']}")
        print(f"  logP: {row['logP评估']}")
        print(f"  Lipinski: {row['Lipinski评估']}")
        print(f"  QED: {row['QED评估']}")
        print(f"  致突变性: {row['致突变性评估']}")
        print(f"  生物利用度: {row['生物利用度评估']}")
        print(f"  肠道吸收: {row['肠道吸收评估']}")
        print(f"  心脏毒性: {row['心脏毒性评估']}")
        print(f"  水溶性: {row['水溶性评估']}")
        print(f"  CYP酶抑制: {row['CYP酶抑制评估']}")
    
    print("\n✓ 预测和评估完成！")
