
# PyPokerEngine Diploma Edition 🎓♠️

[English version below ⬇](#english-version)



## 📌 Описание (RU)

**PyPokerEngine Diploma Edition** — это улучшенная версия популярного движка для покера **PyPokerEngine**, адаптированная для учебных и исследовательских целей.  
Движок позволяет симулировать игры в **No-Limit Texas Hold'em**, подключать различных агентов (ботов) и проводить эксперименты с их стратегиями.

🔹 Подходит для:
- написания дипломных и научных работ 🧑‍🎓  
- экспериментов в области **Reinforcement Learning (RL)** и машинного обучения 🤖  
- тестирования и сравнения стратегий агентов ♣️  

---

## 🚀 Возможности
- 🎲 Поддержка **No-Limit Texas Hold'em**
- 🧩 Простое подключение пользовательских агентов
- 📊 Массовая симуляция партий и сбор статистики (winrate, EV и др.)
- 🐳 Поддержка Docker для быстрого запуска
- 📦 Установка как отдельного пакета (pip install)

---

## ⚙️ Установка

### Вариант 1. Через git
```bash
git clone https://github.com/romanhse/PyPokerEngine_diploma.git
cd PyPokerEngine_diploma
pip install -r requirements.txt
```

### Вариант 2. Через Docker
```bash
docker-compose run --rm pypokerengine bash
```
Можно запустить команду
```bash
python3 example_game.py
```
чтобы запустить по 10 игр модели и Честного, Случайного и Fish - игрока

---

## ▶️ Пример использования

```python
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random

# Простейший агент: выбирает случайное действие
class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return random.choice(valid_actions)['action'], 0

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

config = setup_config(max_round=10, initial_stack=1000, small_blind=10)
config.register_player(name="p1", algorithm=RandomPlayer())
config.register_player(name="p2", algorithm=RandomPlayer())

game_result = start_poker(config, verbose=1)
print(game_result)
```

---

## 🗂️ Структура проекта

```
PyPokerEngine_diploma/
├─ pypokerengine/      # исходники движка
├─ examples/           # примеры использования
├─ tests/              # тесты
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 🧠 Механики и API

- **setup_config(max_round, initial_stack, small_blind)** — настройка турнира  
- **register_player(name, algorithm)** — регистрация агента  
- **start_poker(config)** — запуск симуляции  

**Агент (бот)** должен наследоваться от `BasePokerPlayer` и реализовывать:
- `declare_action(...)` — выбор действия  
- `receive_*` методы — получение событий о ходе игры  

---

## 👨‍💻 Автор
Проект создан в рамках диплома.  
Автор: [Roman HSE](https://github.com/romanhse)  

---

# English version

## 📌 Description (EN)

**PyPokerEngine Diploma Edition** is an enhanced version of the popular **PyPokerEngine** library, adapted for educational and research purposes.  
It allows to simulate **No-Limit Texas Hold'em** games, plug various agents (bots), and experiment with their strategies.

🔹 Use cases:
- 🧑‍🎓 diploma or academic projects  
- 🤖 reinforcement learning and AI research  
- ♣️ poker strategy testing  

---

## 🚀 Features
- 🎲 Supports **No-Limit Texas Hold'em**
- 🧩 Easy agent integration
- 📊 Mass simulations and statistics collection (winrate, EV, etc.)
- 🐳 Docker support for quick start
- 📦 Can be installed as a package (pip install)

---

## ⚙️ Installation

### Option 1. From git
```bash
git clone https://github.com/romanhse/PyPokerEngine_diploma.git
cd PyPokerEngine_diploma
pip install -r requirements.txt
```

### Option 2. With Docker
```bash
docker-compose run --rm pypokerengine bash
```
You can run the command
```bash
python3 example_game.py
```
to run 10 games with Honest, Random and Fish player with the model

---

## ▶️ Example usage

```python
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return random.choice(valid_actions)['action'], 0

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

config = setup_config(max_round=10, initial_stack=1000, small_blind=10)
config.register_player(name="p1", algorithm=RandomPlayer())
config.register_player(name="p2", algorithm=RandomPlayer())

game_result = start_poker(config, verbose=1)
print(game_result)
```

---

## 🗂️ Project structure

```
PyPokerEngine_diploma/
├─ pypokerengine/      # core engine
├─ examples/           # usage examples
├─ tests/              # test suite
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 🧠 API Mechanics
- **setup_config(max_round, initial_stack, small_blind)** — configure game  
- **register_player(name, algorithm)** — register player agent  
- **start_poker(config)** — run simulation  

**Agent (bot)** must inherit from `BasePokerPlayer` and implement:
- `declare_action(...)` — decision logic  
- `receive_*` methods — receive game events  

---

## 👨‍💻 Author
Developed as part of a diploma project.  
Author: [Roman HSE](https://github.com/romanhse)  

---
```

---

👉 Хочешь, я дальше подготовлю `requirements.txt` с актуальными зависимостями, или сначала внесём `README.md` и посмотрим как он смотрится в репозитории?