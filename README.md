# Banking Intent Classifier

Мультиклассовая классификация банковских обращений по 77 интентам. В проекте
сравниваются простой промышленный baseline, Transformer Encoder без предобучения
и fine-tuning DistilBERT.

## Результат

| Модель | Validation accuracy | Validation macro F1 | Время обучения* |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.865 | 0.861 | **0.6 c** |
| Transformer с нуля | 0.824 | 0.824 | 19.5 c |
| DistilBERT fine-tuning | **0.892** | **0.882** | 100.6 c |

На официальном test выбранный DistilBERT получил `accuracy = 0.889` и
`macro F1 = 0.886`.

\* Время относится к использованному локальному окружению и приведено только для
сравнения экспериментов внутри проекта.

## Основной вывод

- **DistilBERT** — лучшая модель по качеству;
- **TF-IDF** — лучший вариант по соотношению качества, скорости и сложности;
- **Transformer с нуля** — учебный эксперимент: небольшой размеченной выборки
  недостаточно, чтобы самостоятельно выучить качественные языковые
  представления.

Улучшение Transformer-подхода связано прежде всего с предобучением, а не только
с архитектурой. DistilBERT сократил количество ошибок на validation с 270 до
217, но потребовал значительно больше вычислений.

## Данные и защита от утечки

Использован официальный
[Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets):
10 003 обучающих и 3 080 тестовых обращения, размеченных по 77 интентам.

Подготовка данных выполняется один раз в `01_data_analysis.ipynb`:

1. проверяются пропуски, дубликаты и набор классов;
2. нормализация текста используется только для поиска совпадений;
3. обращения, совпадающие с официальным test, удаляются только из train;
4. из очищенного train создаётся фиксированная стратифицированная validation;
5. официальный test не участвует в выборе модели.

Все модели используют одни и те же сохранённые разбиения, поэтому их метрики
сопоставимы.

## Эксперименты

1. `TF-IDF + Logistic Regression` — быстрый интерпретируемый baseline.
2. `Transformer Encoder from scratch` — собственная токенизация, embeddings,
   positional encoding, multi-head attention и classifier head.
3. `DistilBERT` — fine-tuning предобученной модели
   `distilbert/distilbert-base-uncased`.
4. Финальный notebook загружает сохранённые predictions, сравнивает модели и
   один раз оценивает выбранную модель на test.

## Структура проекта

```text
.
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_transformer_from_scratch.ipynb
│   ├── 04_distilbert_finetuning.ipynb
│   └── 05_model_comparison.ipynb
├── reports/
│   ├── predictions/
│   ├── final_metrics.json
│   └── *_summary.json
├── README.md
└── requirements.txt
```

Данные и веса моделей не хранятся в Git. В репозитории сохранены выполненные
notebook’и, история обучения, метрики и predictions, необходимые для проверки
выводов.

## Запуск

```bash
git clone https://github.com/novaeKH/banking-intent-classifier.git
cd banking-intent-classifier
python -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
jupyter lab
```

Сначала разместите исходные `train.csv` и `test.csv` в `data/raw/`, затем
выполняйте notebook’и по номеру. Подробности находятся в `data/README.md`.

## Ограничения

- данные короткие и англоязычные, поэтому качество не переносится автоматически
  на длинные диалоги или русский язык;
- DistilBERT требует больше памяти и времени, чем TF-IDF baseline;
- проект сравнивает offline-качество и не включает production API, latency-тест
  на сервере и мониторинг drift.

