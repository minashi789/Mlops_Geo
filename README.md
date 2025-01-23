# MLOps проект: Сопоставление геолокаций Foursquare

## 1. Краткое описание ML проекта

Цель данного проекта — освоение подходов MLOps на основе практической задачи машинного обучения. Основной фокус сделан на построении MLOps-обертки, обеспечивающей эффективное проведение исследований и итеративное улучшение сервиса, а не на достижении максимальной производительности ML-модели. ML-часть представляет собой простой, но рабочий бейзлайн.

В качестве ML-задачи выбрана задача матчинга геолокаций из соревнования Kaggle Foursquare - Location Matching ([https://www.kaggle.com/competitions/foursquare-location-matching](https://www.kaggle.com/competitions/foursquare-location-matching)). На вход подается датасет точек интереса (POI) с информацией о них. Данные, полученные из различных источников, могут содержать несоответствия, избыточность, конфликты, двусмысленность и неточности. Задача состоит в определении дубликатов POI.

**Постановка задачи (легенда):**

Разработчики являются сотрудниками Foursquare и создают внутренний сервис сопоставления POI, доступный через API. Проект должен обеспечивать возможность итеративного улучшения сервиса и проведения исследований без влияния на работу основной версии.

**Рамки проекта:**

*   Сервис реализован в виде API (пользовательский интерфейс не требуется).
*   Входные данные: .csv файл заданного формата (см. ниже).
*   Выходные данные: .csv файл с данными о найденных дубликатах POI (формат sample_submission.csv).
*   Сервис работает внутри организации и доверенной инфраструктуры (меры безопасности для публикации в Интернет не требуются).
*   Проект должен обеспечивать возможность проведения экспериментов и изменений ML-пайплайна без прерывания работы текущей версии сервиса.

**Описание исходных данных:**

Набор данных содержит более полутора миллионов записей о POI.

*   `train.csv`: обучающий набор с одиннадцатью атрибутами POI:
    *   `id`: уникальный идентификатор записи.
    *   `point_of_interest`: уникальный идентификатор POI (целевая переменная для обучения).
    *   `latitude`, `longitude`: географические координаты.
    *   `name`, `address`, `city`, `state`, `zip`, `country`, `url`, `phone`, `categories`: информация о POI.
*   `sample_submission.csv`: пример выходного формата:
    *   `id`: исходные id.
    *   `matches`: список id найденных дубликатов (включая исходный id, если дубликатов нет).

**Результаты анализа и обработки данных:**

EDA показал наличие пропущенных значений (особенно в `url`, `phone` и `zip`), категориальный характер большинства признаков (кроме координат) и возможные ошибки в данных.

**Описание бейзлайна:**

Задача сведена к бинарной классификации путем отбора кандидатов для сравнения. Отбор кандидатов производится по округленным координатам `latitude` и `longitude`. Для каждой пары формируются признаки:

*   Географическое расстояние между точками.
*   Расстояние Левенштейна между названиями.

Целевая переменная формируется сравнением `point_of_interest`. Обучается `XGBClassifier`. Выходной файл соответствует формату `sample_submission.csv`.

**Метрики качества:**

Используется Jaccard score (среднее значение Jaccard index). Jaccard score рассчитывается на этапе формирования пар кандидатов (максимальный достижимый) и после предсказаний модели (итоговый).

## 2. Описание MLOps подходов

**Система контроля версий:** Git (хранилище: GitLab).

**Инструменты контроля codestyle:**

*   Форматер: Black (максимальная длина строки 100 символов).
*   Линтеры: pylint, flake8, mypy, bandit.

Проверки выполняются в IDE (VSCode) и в GitLab CI пайплайне (задача `test_lint`). Pre-commit hooks не используются.

**Шаблон проекта:** Cookiecutter Data Science ([https://github.com/drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)).

**Менеджер зависимостей:** poetry.

**Структура проекта:** Исходный код в `src`, данные в `data`.

**Workflow менеджер и версионирование данных:** DVC + локально развернутый s3 сервис (minio + nginx). Бакет DVC: `dvc`. Учетные данные хранятся в `.dvc/config.local` (исключен из git). В s3 версионируются исходные данные, train/test, метрики и модели. Для остальных файлов `cache: false` в `dvc.yaml`. Пайплайн реализован через `dvc.yaml` и `params.yaml`. Код разбит на CLI модули (с использованием click). Исполнение: `dvc repro`.

**Трекинг экспериментов:** MLflow (PostgreSQL, s3 minio бакет `mlflow`).

**Методы и инструменты тестирования:**

*   pytest (`test_evaluate.py`, `test_predict_model.py` с использованием great_expectations).
*   dvc repro.

Тесты выполняются в задаче `pytest` стейджа `tests` пайплайна `gitlab-ci.yml`.

**Описание CI пайплайна:**

*   `tests`: линтинг (`test_lint`), юнит-тесты (`pytest`). Запускается при коммите/пуше в любую ветку (кроме main).
*   `build`: сборка Docker-образа `dev_ml_service` и доставка в GitLab registry. Запускается после успешного merge request в main.

## 3. Описание сервиса/продукта

Сервис реализован в виде Docker-контейнера `dev_ml_service` с API: `http://<ip or hostname>:8004/invocations`. Принимает POST-запрос с .csv файлом (параметр `file`) и возвращает файл в формате `sample_submission.csv`.

**Описание CD пайплайна:**

*   `deploy`: загрузка Docker-образа из GitLab registry и деплой (запуск контейнера). Запускается после успешного merge request в main и успешного завершения CI стейджа `build`. Предыдущие версии контейнеров с префиксом `_dev` останавливаются и удаляются.

**Итоговый технологический стек:**

*   Python 3.9
*   poetry
*   cookiecutter data science
*   git/gitlab
*   dvc + minio
*   mlflow
*   click
*   XGBClassifier
*   pytest, great expectations
*   FastAPI+Uvicorn
*   docker

**Запуск** 
*   git clone https://gitlab.com/mlops-23reg-team/mlops-23reg-project.git

*   все нижеуказанные команды выполняем находясь в корневой директории проекта
создаем в корне проекта файл .env следующего содержания:
```
MINIO_ROOT_USER = admin
MINIO_ROOT_PASSWORD = miniotestpass
AWS_ACCESS_KEY_ID = L6Vt9Pw72XDw26Mt
AWS_SECRET_ACCESS_KEY = IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
AWS_S3_MLFLOW_BUCKET = mlflow
AWS_S3_DVC_BUCKET = dvc
MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
MLFLOW_TRACKING_URI = "http://localhost:5000"
POSTGRES_USER = dbuser
POSTGRES_PASSWORD = dbtestpass
POSTGRES_DB = mlflow
PGADMIN_DEFAULT_EMAIL = "admin@admin.com"
PGADMIN_DEFAULT_PASSWORD = pgtestpass
```


*   создаем файл .dvc/config.local следующего содержания
```
['remote "s3minio"']
    access_key_id = L6Vt9Pw72XDw26Mt
    secret_access_key = IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
```

```
docker-compose up -d --build (дожидаемся старта контейнеров)
```
*   производим предварительную настройку minio s3

*   Заходим через браузер http://127.0.0.1:9001 и авторизуемся admin/miniotestpass
Через веб-интерфейс создаем два бакета:

*   Buckets -> Create bucket/
mlflow/
dvc

*   Через веб-интерфейс создаем сервисный аккаунт (по сути API Key):

*   Identity -> Service Accounts -> Create service account,
Name: L6Vt9Pw72XDw26Mt,
Pass: IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G

*   Добавляем в проект исходные данные (датасеты)
*   Заходим в соревнование https://www.kaggle.com/competitions/foursquare-location-matching/data
*   копируем из этого соревнования файлы train.csv, test.csv, sample_submission.csv, pairs.csv в директорию проекта ./data/raw/

*   Создаем виртуальную среду
```
poetry install
poetry shell
```
*   Запускаем эксперимент
```
dvc repro
```

Заходим в mlflow через http://127.0.0.1:5000

*   проверяем что трекинг эксперимента успешно добавлен,
заходим в модели и выставляем для текущей модели stage = staging


*   Все последующие шаги возможны, только после привязки локального репозитория к удаленному в гитлабе, в который есть полный доступ (перепривязываем в какой-либо свой)
Для запуска целевого сервиса, на текущей машине необходимо поднять гитлаб раннер (подробная инструкция есть в подразделе "Настраиваем локальный CI Runner" дневника курса)
*   В удаленном репозитории gitlab в Settings -> CI/CD -> Variables создаем следующие переменные:
```
AWS_ACCESS_KEY_ID (Protected: False, Masked: True): L6Vt9Pw72XDw26Mt
AWS_SECRET_ACCESS_KEY (Protected: False, Masked: True): IEnnVlaJqfKgNWKTFu1RE9Uk8jSfd52G
MLFLOW_S3_ENDPOINT_URL (Protected: False, Masked: False):  http://<ip_docker_server_host>:9000 (<ip_docker_server_host> узнаем через PowerShell выполнив wsl hostname -I (должна быть именно заглавная английская "и")
MLFLOW_TRACKING_URI (Protected: False, Masked: False):  http://<ip_docker_server_host>:5000
APP_PROJECT_VERSION(Protected: False, Masked: False): 0.1.0
```
```
git tag -af v0.1.0 -m "Project version 0.1.0"
git push v0.1.0 (или git push --tags, в этом случае запушатся все локальные теги)
git push origin main
```
*   По итогу отработки всех gitlab пайплайнов должны получить работающий контейнер dev_ml_service и API по адресу http://127.0.0.1:8004/invocations:
  Проверить работу API можно через postman
```
POST
http://127.0.0.1:8004/invocations
BODY -> from-data
```
В первой строке, в поле key вводим "file" (это имя параметра функции async def create_upload_file(file: UploadFile = File(...)), в этой же ячейке справа в выпадающем списке выбираем File
указываем предварительно подготовленный файл .csv с несколькими строками, исходного формата датафрейма.
