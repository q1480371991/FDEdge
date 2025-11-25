import matplotlib.pyplot as plt
import os
import numpy as np


def draw():
    # 创建保存目录
    log_dir = 'compare_plot_log'
    os.makedirs(log_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ======== 1. 对比的实验（修改为读取 CSV 文件） ========
    # 格式：{方法名称: CSV文件路径}
    # 注意：如果 CSV 文件和脚本在同一目录，直接写文件名；否则写绝对路径
    runs = {
        "FDEdge": "../results/AveDelay_FDEdge_BS10_tasks100_f50_steps5_episode100.csv",
        "DQN": "../results/AveDelay_dqn_BS10_tasks100_f50_episode100.csv",
        "OPT": "../results/AveDelay_Opt_BS10_tasks100_f50_episode100.csv",
        "SAC": "../results/AveDelay_sac_BS10_tasks100_f50_episode100.csv",
    }

    # ======== 2. 指标设置（只保留时延指标） ========
    metric_keys = {
        "平均服务时延 (Average Service Delay)": "average_delay"
    }

    # ======== 3. 读取所有 CSV 文件（适配旧版本 NumPy） ========
    results = {}
    min_length = float('inf')

    # 第一遍：读取数据并找到最小长度
    for name, path in runs.items():
        # 转换为绝对路径，避免相对路径问题
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"[警告] {name} 的文件不存在：{abs_path}，跳过")
            continue

        try:
            # 移除 fmt 参数，适配旧版本 NumPy
            # 先读取为字符串，再转换为浮点数（兼容不同格式）
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 过滤空行，转换为浮点数
                avg_delays = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            avg_delays.append(float(line))
                        except:
                            pass  # 跳过无法转换的行

            if not avg_delays:
                print(f"[警告] {name} 的文件 {abs_path} 中没有有效数据，跳过")
                continue

            data_len = len(avg_delays)
            min_length = min(min_length, data_len)

            results[name] = {
                "average_delay": avg_delays
            }
            print(f"成功读取 {name}：{data_len} 个 episode 的平均时延数据（文件：{abs_path}）")
        except Exception as e:
            print(f"[错误] 读取 {name} 的文件 {abs_path} 失败：{str(e)}，跳过")
            continue

    if not results:
        print("[错误] 没有成功读取到任何数据，程序退出")
        return

    print(f"\n统一数据长度为: {min_length} episodes")

    # 第二遍：统一截取数据
    for name, data in results.items():
        unified_avg_delays = data["average_delay"][:min_length]
        results[name]["average_delay"] = unified_avg_delays
        print(f"{name} 截取后数据长度：{len(unified_avg_delays)}")

    # ======== 4. 绘图并保存 ========
    for zh_name, key in metric_keys.items():
        plt.figure(figsize=(10, 6))

        for name, data in results.items():
            y = data[key]
            x = range(1, len(y) + 1)  # x轴从1开始
            plt.plot(x, y, linewidth=2.5, label=name, alpha=0.8, markersize=4)

        # 图表美化
        plt.xlabel("训练轮次 / Episode", fontsize=12)
        plt.ylabel(zh_name, fontsize=12)
        plt.title(f"不同卸载调度算法的 {zh_name} 对比", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()

        # 保存图片
        safe_name = zh_name.replace(" ", "_").replace("(", "").replace(")", "").replace("（", "").replace("）", "")
        save_path = os.path.join(log_dir, f"{safe_name}_comparison.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n已保存图像: {os.path.abspath(save_path)}")

        # 显示图片
        plt.show()
        plt.close()


if __name__ == "__main__":
    draw()