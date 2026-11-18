# Banking77 data

Источник: [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets).
Набор содержит банковские обращения на английском языке и 77 категорий
интентов.

Для обучения нужны два исходных CSV-файла:

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

Папки `data/raw/` и `data/processed/` исключены из Git.
Новый эксперимент `banking_intent.experiment` читает исходные CSV и отдельно
выделяет tune и policy из validation; его размеры разбиений сохранены
в `reports/evaluation/split_summary.csv`.
