import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

lr_fixed, lr_tuned = 46, 18
bs_fixed, bs_tuned = 62, 2
rank_fixed, rank_tuned = 24, 40
all_fixed, all_tuned = 63, 1

categories = ["Learning Rate", "Batch Size", "Rank", "All Three"]
fixed_data = [lr_fixed, bs_fixed, rank_fixed, all_fixed]
tuned_data = [lr_tuned, bs_tuned, rank_tuned, all_tuned]

OUTPUT_FILE = "lora-paper-counts.png"

# ==========================================
# 2. 配色與樣式
# ==========================================
# 顏色保留，但加上 hatch 提升色盲友善性
color_fixed = '#C0C0C0'   # 中性銀灰
color_tuned = '#4C72B0'   # 深藍

# 不同 pattern
hatch_fixed = '//'
hatch_tuned = '..'

# ==========================================
# 3. 繪圖
# ==========================================
def plot_fixed_vs_tuned():
    plt.style.use("default")

    plt.rcParams.update({
        "font.size": 20,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "hatch.linewidth": 1.2,   # 讓 hatch 更清楚
    })

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    x = np.arange(len(categories))
    width = 0.35

    # 繪製柱狀圖：加入 hatch
    rects_fixed = ax.bar(
        x - width/2, fixed_data, width,
        label="Fixed",
        color=color_fixed,
        edgecolor="black",
        linewidth=1.5,
        hatch=hatch_fixed,
        zorder=3, 
        alpha=0.6, 
    )

    rects_tuned = ax.bar(
        x + width/2, tuned_data, width,
        label="Tuned",
        color=color_tuned,
        edgecolor="black",
        linewidth=1.5,
        hatch=hatch_tuned,
        zorder=3
    )

    # Y 軸標題
    ax.set_ylabel("Number of Papers", fontsize=22, labelpad=10)

    # X 軸設定
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=22, rotation=0, ha='center')

    ax.tick_params(axis="x", labelsize=18.5, pad=8)
    ax.tick_params(axis="y", labelsize=20)

    # Y 軸刻度間距與範圍
    max_height = max(max(fixed_data), max(tuned_data))
    ax.set_ylim(0, 
                # max_height * 1.30)
                max_height * 1.25)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    # 細節美化
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 數字標註
    def autolabel(rects):
        for r in rects:
            h = r.get_height()
            ax.annotate(
                f"{int(h)}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=14,
                fontweight='bold',
                color="black"
            )

    autolabel(rects_fixed)
    autolabel(rects_tuned)

    # Legend
    ax.legend(
        fontsize=15.5,
        frameon=True,
        framealpha=0.9,
        loc="best",
        edgecolor='black',
        ncol=2
    )

    plt.tight_layout()

    # 儲存 PNG 與 PDF
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor='white')
    pdf_output = OUTPUT_FILE.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches="tight", facecolor='white')

    plt.close()
    print(f"Plot saved to {OUTPUT_FILE} \nand PDF: {pdf_output}")

if __name__ == "__main__":
    plot_fixed_vs_tuned()