import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import logging

def generate_umap_plot(
    reference_fps: np.ndarray,
    query_fps: np.ndarray,
    output_path: str,
    umap_plot_params: dict = None
):
    """
    Создает 2D UMAP-визуализацию, на которой показаны все атомарные окружения из
    референсной базы и выделены окружения из query-структуры.

    Args:
        reference_fps (np.ndarray): Массив фингерпринтов из базы [N_ref, F_len + 1].
        query_fps (np.ndarray): Массив фингерпринтов из SMILES [N_query, F_len].
        output_path (str): Путь для сохранения итогового изображения.
        umap_plot_params (dict): Параметры для UMAP (n_neighbors, min_dist).
    """
    if umap_plot_params is None:
        umap_plot_params = {'n_neighbors': 15, 'min_dist': 0.1}

    logging.info("Создание 2D UMAP-визуализации...")
    
    # 1. Подготовка данных
    ref_fp_only = reference_fps
    #ref_config_ids = reference_fps[:, -1] # для раскраски

    # Объединяем все фингерпринты в один массив для UMAP
    all_fps = ref_fp_only
    
    # 2. Нормализация
    logging.info("Нормализация всех фингерпринтов для UMAP...")
    scaler = StandardScaler()
    all_fps_scaled = scaler.fit_transform(all_fps)

    # 3. UMAP
    logging.info("Запуск UMAP для понижения размерности до 2D...")
    reducer = umap.UMAP(n_components=2, **umap_plot_params)
    embedding = reducer.fit_transform(all_fps_scaled)

    # Разделяем обратно на референсные и query-точки
    n_ref_atoms = ref_fp_only.shape[0]
    ref_embedding = embedding
    query_embedding = reducer.transform(query_fps)
    
    # 4. Отрисовка
    logging.info("Отрисовка графика...")
    plt.figure(figsize=(12, 12))
    
    # Рисуем референсные точки
    # Используем `c=ref_config_ids` для раскраски точек в соответствии с их исходной конфигурацией
    plt.scatter(
        ref_embedding[:, 0],
        ref_embedding[:, 1],
        c='grey',
        s=5,           # Маленький размер для фона
        alpha=0.3,     # Прозрачность
        cmap='viridis',
        label='Атомы из базы данных'
    )
    
    # Рисуем query-точки (оранжевые, как в примере)
    plt.scatter(
        query_embedding[:, 0],
        query_embedding[:, 1],
        c='orange',
        s=100,         # Большой размер для выделения
        edgecolors='black',
        linewidths=1.5,
        alpha=0.5,
        label=f'Атомы из SMILES'
    )
    
    plt.title('2D UMAP-визуализация атомарных окружений')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 5. Сохранение
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"UMAP-визуализация сохранена в: {output_path}")
