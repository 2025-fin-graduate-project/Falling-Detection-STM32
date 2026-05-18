# Pure Python - No dependencies
internal_data = {
    "ISR Vector": 844,
    "Text (Code)": 169276,
    "RO Data": 353088,
    "Data (Init)": 12796,
    "BSS (Global)": 24800,
    "Heap/Stack": 16896,
}

total_internal = sum(internal_data.values())
psram_ai_buffer = 2304000
pose_pipeline_size = 14252

# Markdown Table
table = "| Region | Component | Size (Bytes) | Size (KB) | % of Region |\n"
table += "| :--- | :--- | ---: | ---: | ---: |\n"

for k, v in internal_data.items():
    kb = v / 1024
    perc = (v / total_internal) * 100
    table += "| AXISRAM | {} | {:,} | {:.2f} | {:.1f}% |\n".format(k, v, kb, perc)

table += "| **Total** | **Internal SRAM** | **{:,}** | **{:.2f}** | **100%** |\n".format(total_internal, total_internal/1024)
table += "| PSRAM | AI Buffers | {:,} | {:.2f} | 100% |\n".format(psram_ai_buffer, psram_ai_buffer/1024)

# Visualization in Text (Bar Chart)
def draw_bar(val, total, width=20):
    filled = int((val/total) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

bar_chart = "\n### Memory Usage Visualization (Internal SRAM)\n\n"
for k, v in internal_data.items():
    bar_chart += "{:<15} {:<25} {:.1f}%\n".format(k, draw_bar(v, total_internal), (v/total_internal)*100)

summary = "\n### 핵심 요약\n"
summary += "- **AXISRAM 점유율**: {:.1f} KB / 1,023 KB ({:.1f}%)\n".format(total_internal/1024, (total_internal/1024/10.23))
summary += "- **Pipeline D (전처리)**: {:,} Bytes ({:.1f}% of Internal SRAM)\n".format(pose_pipeline_size, (pose_pipeline_size/total_internal)*100)
summary += "- **PSRAM (AI 모델 전용)**: {:,} KB (Activations & Buffers)\n".format(int(psram_ai_buffer/1024))

with open('Doc/MEMORY_REPORT.md', 'w') as f:
    f.write("# STM32N6570-DK Fall Detection Memory Analysis\n\n")
    f.write(table)
    f.write(bar_chart)
    f.write(summary)

print("Memory report generated: Doc/MEMORY_REPORT.md")
