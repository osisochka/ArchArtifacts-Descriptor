# SIMILIS — Классификация археологических артефактов

Классификация изображений артефактов ИИМК РАН: одновременно предсказывается тип (`norm_name`, 7 классов) и материал (`norm_material`, 5 классов).

## Метод

**Модель** — EfficientNet-B0 с двумя независимыми головками, обучаемыми совместно через взвешенную кросс-энтропию.

**Предобработка** — масштабирование по длинной стороне до 224 px + белый padding (сохраняет все ракурсы многопроекционных карточек).

**Разбивка** 
- 50% train · 15% val · 15% test · 20% pool (для active learning)

**Нормализация меток** — 21 вариант материала → 5 классов; устранены синонимы (`Красноглиняная керамика` → `Керамика`).

**Аугментация описаний** правило-базовая замена синонимов, ~1120 → ~5600 пар. (интнгрирование с декодером пока не реализовано)

**Active learning** (budget = 50 групп)

 Стратегия | Отбор

Random | случайная выборка из пула 

Uncertainty | максимальная энтропия на группу 

k-Center Coreset | максимальное cosine-расстояние от labeled set 

**Метрика** — macro-F1 по `norm_name` на `val_gold` / `test_gold`.


## Запуск

Ноутбук рассчитан на **Google Colab + Google Drive**.

1. Загрузить `dataset` и `selected_by_name_iimk_subset_public.csv` в `MyDrive/CU_4/SIMILIS/`
2. Открыть `similis_notebook_gc.ipynb` в Colab
3. Выполнить ячейки последовательно (Runtime → Run all)

Чекпоинты сохраняются в Drive и подгружаются при повторном запуске (resume).

## Зависимости

```
torch · torchvision · numpy · pandas · matplotlib · seaborn
scikit-learn · Pillow · tqdm
```

Устанавливаются автоматически в Colab; локально: `pip install torch torchvision scikit-learn pandas matplotlib seaborn pillow tqdm`.


## Требования перед запуском

В папке artifacts/checkpoints/ должен лежать файл baseline_best.pt, а в data/processed/ — name_le.pkl и mat_le.pkl. Они генерируются при прогоне ноутбука.

## Установка зависимостей

pip install -r requirements.txt
## Запуск
```
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
### После этого:

Фронтенд (UI) — http://localhost:8000/
API docs (Swagger) — http://localhost:8000/docs
Health check — http://localhost:8000/health
Использование

Загрузить изображение артефакта через форму на главной странице — в ответ придут предсказанные тип (pred_name), материал (pred_material), уверенность и топ-3 варианта по каждому полю.

