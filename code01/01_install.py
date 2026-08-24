"""
01_install.py - 安装ADMET-AI所需的依赖
运行一次即可：python 01_install.py
"""
import subprocess
import sys

def install_package(package):
    """安装Python包"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("开始安装依赖...")
    
    # 基础依赖
    packages = [
        "pandas",
        "numpy",
        "rdkit",
        "admet-ai",
    ]
    
    for pkg in packages:
        print(f"安装 {pkg}...")
        install_package(pkg)
    
    print("\n✓ 安装完成！")

if __name__ == "__main__":
    main()
