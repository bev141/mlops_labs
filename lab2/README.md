MLOps. Практическое задание №2.

Учебный автоматический конвейер проекта машинного обучения для Jenkins.

Шаги для выполния задания.
1. Установлен Jenkins.
2. Установленны приложения: `apt install python3 python-is-python3 python3-pip python3-venv`
3. Создан проект в Jenkins, где указан данный репозитарий. Чтобы запустить pipeline.sh из репозитария в Jenkins введена команда:
```
cd lab2
sh pipeline.sh
```

После этого при сборке pipeline.sh автоматически выполняет все шаги: 
1. Создает и активирует виртуальное окружение.
2. Устанавливает зависимости.
3. Создает и предобрабатывает датасет.
4. Обучает модель.
5. Проверяет модель на тестовых данных.