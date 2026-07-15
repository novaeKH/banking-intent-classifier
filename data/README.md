# Banking77 data

Источник: [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets).
Набор содержит банковские обращения на английском языке и 77 категорий
интентов.

Для запуска первого notebook подготовьте два CSV-файла:

```text
data/raw/
├── train.csv
└── test.csv
```

Каждый файл должен содержать столбцы:

| Столбец | Описание |
|---|---|
| `text` | текст обращения клиента |
| `category` | название банковского интента |

`notebooks/01_data_analysis.ipynb` проверяет данные, удаляет пересечения train с
официальным test и создаёт фиксированные файлы:

```text
data/processed/
├── train.csv
├── validation.csv
└── test.csv
```

Папки `data/raw/` и `data/processed/` исключены из Git. Это предотвращает
случайную публикацию данных и гарантирует, что подготовка выполняется явно.
