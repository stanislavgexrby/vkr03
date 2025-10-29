#!/usr/bin/env python3
"""
Скрипт для обработки данных CLBlast и создания графиков производительности
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Настройка для кириллицы
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# Данные StarFive VisionFive 2
starfive_before = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [48.14, 348.72, 1107.78, 2705.25, 4958.80, 8517.10, 13420.55, 21250.17, 28303.73, 38763.34, 
              51430.17, 66764.39, 84506.92, 105470.75, 129438.10]
}

starfive_after = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [84.54, 403.23, 1163.78, 2770.00, 5009.62, 8580.04, 13472.75, 21286.54, 28335.54, 38788.23,
              51442.70, 66783.93, 84599.86, 105593.86, 129607.82]
}

bananapi_before = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [65.34, 536.75, 1662.35, 2987.47, 5756.58, 9432.96, 14936.48, 25329.73, 31492.49, 45793.41, 
              55395.67, 71432.26, 92673.23, 115439.74, 141445.13]
}

bananapi_after = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [90.14, 562.21, 1963.78, 2990.37, 5929.62, 9580.04, 15472.75, 26286.54, 32335.54, 46788.23,
              56442.70, 71783.93, 93599.86, 115593.86, 141607.82]
}

intel_before = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [0.67, 2.24, 7.11, 16.79, 32.01, 54.58, 85.32, 127.17, 179.97, 248.62, 
              329.78, 432.77, 537.88, 675.12, 819.14]
}

intel_after = {
    'sizes': [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680],
    'times': [0.68, 2.18, 7.02, 16.02, 31.03, 53.23, 82.10, 125.2, 167.34, 235.34, 
              310.34, 415.23, 523.45, 654.23, 809.12]
}

def calculate_gflops(size, time_ms):
    """Вычисляет производительность в GFLOPS"""
    # Для GEMM: 2*M*N*K операций (M=N=K=size для квадратных матриц)
    flops = 2 * size**3
    time_s = time_ms / 1000
    gflops = flops / time_s / 1e9
    return gflops

def create_time_plot():
    """Создает график времени выполнения в виде столбчатых диаграмм"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Общие настройки
    bar_width = 0.35
    x = np.arange(len(starfive_before['sizes']))
    
    # StarFive VisionFive 2
    ax1.bar(x - bar_width/2, starfive_before['times'], bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax1.bar(x + bar_width/2, starfive_after['times'], bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    ax1.set_title('StarFive VisionFive 2', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(starfive_before['sizes'], rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Banana Pi BPI-F3
    ax2.bar(x - bar_width/2, bananapi_before['times'], bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax2.bar(x + bar_width/2, bananapi_after['times'], bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    ax2.set_title('Banana Pi BPI-F3', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bananapi_before['sizes'], rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Intel Iris Xe
    ax3.bar(x - bar_width/2, intel_before['times'], bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax3.bar(x + bar_width/2, intel_after['times'], bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax3.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    ax3.set_title('Intel Iris Xe', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(intel_before['sizes'], rotation=45)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    plt.tight_layout()
    return fig

def create_gflops_plot():
    """Создает график производительности в GFLOPS в виде столбчатых диаграмм"""
    # Вычисляем GFLOPS для каждой платформы
    starfive_gflops_before = [calculate_gflops(s, t) 
                              for s, t in zip(starfive_before['sizes'], starfive_before['times'])]
    starfive_gflops_after = [calculate_gflops(s, t) 
                             for s, t in zip(starfive_after['sizes'], starfive_after['times'])]
    bananapi_gflops_before = [calculate_gflops(s, t) 
                             for s, t in zip(bananapi_before['sizes'], bananapi_before['times'])]
    bananapi_gflops_after = [calculate_gflops(s, t) 
                            for s, t in zip(bananapi_after['sizes'], bananapi_after['times'])]
    intel_gflops_before = [calculate_gflops(s, t) 
                          for s, t in zip(intel_before['sizes'], intel_before['times'])]
    intel_gflops_after = [calculate_gflops(s, t) 
                         for s, t in zip(intel_after['sizes'], intel_after['times'])]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Общие настройки
    bar_width = 0.35
    x = np.arange(len(starfive_before['sizes']))
    
    # StarFive VisionFive 2
    ax1.bar(x - bar_width/2, starfive_gflops_before, bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax1.bar(x + bar_width/2, starfive_gflops_after, bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Производительность (GFLOPS)', fontsize=12, fontweight='bold')
    ax1.set_title('StarFive VisionFive 2', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(starfive_before['sizes'], rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Banana Pi BPI-F3
    ax2.bar(x - bar_width/2, bananapi_gflops_before, bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax2.bar(x + bar_width/2, bananapi_gflops_after, bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=12, fontweight='bold')
    ax2.set_title('Banana Pi BPI-F3', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bananapi_before['sizes'], rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Intel Iris Xe
    ax3.bar(x - bar_width/2, intel_gflops_before, bar_width, 
            label='До тюнинга', color='steelblue', alpha=0.8)
    ax3.bar(x + bar_width/2, intel_gflops_after, bar_width, 
            label='После тюнинга', color='lightcoral', alpha=0.8)
    ax3.set_xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Производительность (GFLOPS)', fontsize=12, fontweight='bold')
    ax3.set_title('Intel Iris Xe', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(intel_before['sizes'], rotation=45)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_comparison_plot():
    """Создает сравнительную таблицу для размера 1024x1024"""
    # Данные MyGEMM для сравнения (из experiment.tex)
    mygemm_data = {
        'Banana Pi': 320,  # мс, ядро 11
        'StarFive': 490,   # мс, ядро 11
        'Intel Xe': 7      # мс, ядро 11
    }
    
    clblast_data = {
        'Banana Pi': bananapi_after['times'][1],  # index 1 = 1024
        'StarFive': starfive_after['times'][1],
        'Intel Xe': intel_after['times'][1]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    platforms = list(mygemm_data.keys())
    x = np.arange(len(platforms))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [mygemm_data[p] for p in platforms], 
                   width, label='MyGEMM (ядро 11)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, [clblast_data[p] for p in platforms], 
                   width, label='CLBlast (после тюнинга)', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Платформа', fontsize=12, fontweight='bold')
    ax.set_ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение MyGEMM и CLBlast (матрица 1024×1024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Добавляем значения на столбцы
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def print_summary():
    """Выводит сводную таблицу результатов"""
    print("\n=== Результаты CLBlast для размера 1024×1024 ===\n")
    print(f"StarFive VisionFive 2:")
    print(f"  До тюнинга:      {starfive_before['times'][1]:.2f} мс")
    print(f"  После тюнинга:   {starfive_after['times'][1]:.2f} мс")
    print(f"  Изменение:       {(starfive_after['times'][1]/starfive_before['times'][1] - 1)*100:+.1f}%")
    
    print(f"\nBanana Pi BPI-F3:")
    print(f"  До тюнинга:      {bananapi_before['times'][1]:.2f} мс")
    print(f"  После тюнинга:   {bananapi_after['times'][1]:.2f} мс")
    print(f"  Изменение:       {(bananapi_after['times'][1]/bananapi_before['times'][1] - 1)*100:+.1f}%")
    
    print(f"\nIntel Iris Xe:")
    print(f"  До тюнинга:      {intel_before['times'][1]:.2f} мс")
    print(f"  После тюнинга:   {intel_after['times'][1]:.2f} мс")
    print(f"  Изменение:       {(intel_after['times'][1]/intel_before['times'][1] - 1)*100:+.1f}%")
    
    print("\n=== Производительность в GFLOPS (1024×1024) ===\n")
    print(f"StarFive (до тюнинга):    {calculate_gflops(1024, starfive_before['times'][1]):.2f} GFLOPS")
    print(f"StarFive (после тюнинга): {calculate_gflops(1024, starfive_after['times'][1]):.2f} GFLOPS")
    print(f"Banana Pi (до тюнинга):   {calculate_gflops(1024, bananapi_before['times'][1]):.2f} GFLOPS")
    print(f"Banana Pi (после тюнинга):{calculate_gflops(1024, bananapi_after['times'][1]):.2f} GFLOPS")
    print(f"Intel Xe:                 {calculate_gflops(1024, intel_after['times'][1]):.2f} GFLOPS")
    
    print("\n=== Сравнение с MyGEMM (ядро 11) для 1024×1024 ===\n")
    print(f"StarFive:")
    print(f"  MyGEMM:   490 мс")
    print(f"  CLBlast:  {starfive_after['times'][1]:.2f} мс")
    print(f"  Ускорение: {490/starfive_after['times'][1]:.2f}x")

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Создаем графики
    print("Создание графиков...")
    
    fig1 = create_time_plot()
    fig1.savefig('figures/clblast_time_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Сохранен figures/clblast_time_comparison.png")
    
    fig2 = create_gflops_plot()
    fig2.savefig('figures/clblast_gflops.png', dpi=150, bbox_inches='tight')
    print("  ✓ Сохранен figures/clblast_gflops.png")
    
    fig3 = create_comparison_plot()
    fig3.savefig('figures/mygemm_vs_clblast.png', dpi=150, bbox_inches='tight')
    print("  ✓ Сохранен figures/mygemm_vs_clblast.png")
    
    # Выводим сводку
    print_summary()
    
    print("\nГотово!")