# Обзор литературы и инструментов

Дата среза: **15 июля 2026 года**.

## 1. Как читать этот обзор

Покер — игра с неполной информацией, случайностью и стратегической реакцией соперника.
Поэтому высокий средний выигрыш против одного слабого бота ещё не означает общую силу:
стратегия может быть легко эксплуатируема другим соперником. Для магистерской работы нужно
развести три задачи:

1. **приближение к равновесию** — CFR-family, search и best-response оценки;
2. **эксплуатация ограниченно рационального соперника** — opponent modelling и adaptation;
3. **достоверное измерение** — duplicate poker, variance reduction и независимые
   статистические единицы.

Ниже peer-reviewed работы и широко воспроизводимые методы отделены от preprint/frontier
2026 года. Последние полезны для постановки новизны, но не должны быть единственным
фундаментом диссертации до независимой проверки.

## 2. Установленный фундамент

### 2.1. Regret minimization

**Counterfactual Regret Minimization (CFR).** Zinkevich et al. показали, что минимизация
counterfactual regret в extensive-form games приводит average strategy к равновесию Нэша в
двухсторонней zero-sum постановке. Это базовая точка отсчёта для Kuhn/Leduc и главный
методологический контраст обычному DQN. Первичный источник: [Regret Minimization in Games
with Incomplete Information, NeurIPS 2007](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).

**Discounted CFR (DCFR).** Brown и Sandholm предложили по-разному дисконтировать старые
положительные/отрицательные regrets и contribution стратегий, ускоряя практическую
сходимость. Для проекта это более сильный tabular reference, чем vanilla CFR.
[Solving Imperfect-Information Games via Discounted Regret Minimization, AAAI
2019](https://ojs.aaai.org/index.php/AAAI/article/view/4007).

**Deep CFR.** Brown et al. заменили табличные regrets нейросетевой аппроксимацией и
reservoir sampling, что позволяет работать с большими information-state spaces.
[Deep Counterfactual Regret Minimization, ICML
2019](https://proceedings.mlr.press/v97/brown19b.html). Официальная воспроизводимая
реализация автора SD-CFR/Deep-CFR доступна в
[EricSteinberger/Deep-CFR](https://github.com/EricSteinberger/Deep-CFR).

**Single Deep CFR (SD-CFR).** Steinberger предложил хранить набор advantage networks по
итерациям вместо отдельной approximation average-strategy network. Для магистерского
blueprint это привлекательная базовая архитектура: она проще для аудита и устраняет один
источник approximation error. [Single Deep Counterfactual Regret Minimization,
2019](https://arxiv.org/abs/1901.07621).

### 2.2. Search и крупномасштабный покер

**DeepStack.** Continual re-solving, depth-limited search и learned counterfactual values
показали, как избегать полного построения дерева heads-up no-limit hold'em.
[DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker,
2017](https://arxiv.org/abs/1701.01724). Для проекта важен не масштаб системы сам по себе, а
сочетание blueprint/value network с локальным re-solving.

**Libratus.** Система Brown и Sandholm продемонстрировала superhuman performance в
heads-up no-limit Texas hold'em, комбинируя blueprint, nested subgame solving и
self-improvement. [Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top
Professionals, Science 2018](https://doi.org/10.1126/science.aao1733). Это ориентир уровня
задачи, а не реалистичный baseline для одной магистерской работы.

**Pluribus.** Brown и Sandholm распространили search-based подход на six-player no-limit
Texas hold'em. [Superhuman AI for Multiplayer Poker, Science
2019](https://doi.org/10.1126/science.aay2400). В multiplayer zero-sum свойства и смысл
exploitability сложнее, поэтому Pluribus нельзя использовать как доказательство того, что
любой multiplayer агент близок к Nash equilibrium.

**ReBeL.** Brown et al. объединили reinforcement learning и search через public belief
states и self-play. [Combining Deep Reinforcement Learning and Search for Imperfect-Information
Games, 2020](https://arxiv.org/abs/2007.13544), официальный код:
[facebookresearch/rebel](https://github.com/facebookresearch/rebel). Для проекта ReBeL —
следующий уровень после проверенного SD-CFR blueprint, а не первая реализация.

**Student of Games.** Работа развивает универсальное сочетание learning и search для игр с
полной и неполной информацией. [Student of Games: A Unified Learning Algorithm for Both
Perfect and Imperfect Information Games, 2021](https://arxiv.org/abs/2112.03178). Она важна
как аргумент в пользу общей архитектуры, но её computational budget не является честной
точкой сравнения для локального дипломного проекта.

**AlphaHoldem.** End-to-end self-play система для heads-up no-limit hold'em полезна как
сильный empirical baseline и источник открытых данных.
[AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via
End-to-End Reinforcement Learning, AAAI
2022](https://ojs.aaai.org/index.php/AAAI/article/view/20394),
[AlphaHoldem_Data](https://github.com/ZhaoEnMin/AlphaHoldem_Data/).

### 2.3. Action abstraction и современный CFR

No-limit poker имеет большое и контекстное пространство bet sizes. Фиксированная
абстракция удобна для первой версии, но может систематически пропускать важные действия.
**RL-CFR** рассматривает выбор action abstraction как обучаемую задачу.
[RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with
Reinforcement Learning, ICML 2024](https://arxiv.org/abs/2403.04344). В этом проекте
динамическая абстракция должна быть отдельной ablation после стабильного fixed-action
blueprint; иначе невозможно понять источник улучшения.

### 2.4. Эксплуатация и безопасная адаптация

Равновесная стратегия защищает от худшего случая, но оставляет деньги на столе против
систематически слабого соперника. Научно интересный вопрос — как эксплуатировать слабость,
не становясь безгранично уязвимым.

**OX-Search** изучает online adaptation/exploitation поверх search в
imperfect-information games. [Opponent Exploitation in No-Limit Texas Hold'em via OX-Search,
ICML 2024](https://proceedings.mlr.press/v235/ge24b.html). Для проекта это ближайший
peer-reviewed ориентир confidence-aware exploitation.

**EVPA** формулирует value of information/opponent adaptation в крупных
imperfect-information games. [Efficient Value-Based Policy Adaptation in
Imperfect-Information Games, ICLR
2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c1b5863a6b0f925617b917bb2f55be0-Abstract-Conference.html).
Работа полезна для выбора критерия, когда наблюдений уже достаточно для отхода от
blueprint.

Практическая архитектура, следующая из этой линии работ: сильный замороженный blueprint,
модель соперника с uncertainty, exploitative correction и явный safety gate. Без
последнего улучшение против calling station может быть просто обменом общей защищённости на
узкую специализацию.

### 2.5. Оценка стратегии и variance reduction

**Local Best Response (LBR).** Когда точный best response в большой игре недоступен, LBR
пытается найти локальные эксплойты и даёт практический нижний ориентир уязвимости стратегии.
[Local Best Response Techniques for Evaluating Imperfect-Information Game
Strategies, 2016](https://arxiv.org/abs/1612.07547). LBR не является точной exploitability и
должен так и называться в выводах.

**AIVAT.** Burch et al. предложили unbiased variance reduction для оценки poker agents,
используя известные стратегии и контрольные значения. [AIVAT: A New Variance Reduction
Technique for Agent Evaluation in Imperfect Information Games, AAAI
2018](https://ojs.aaai.org/index.php/AAAI/article/view/11481). Duplicate design проще и уже
реализован; AIVAT имеет смысл добавлять после value-function validation, а не имитировать
контрольными величинами сомнительного качества.

Итого для диссертации:

- exact best response/exploitability — в Kuhn/Leduc;
- duplicate `bb/100` — первичная HUNL метрика;
- LBR/RL-BR — HUNL safety proxy;
- AIVAT — последующее снижение дисперсии, если его assumptions действительно выполнены.

## 3. LLM и покер

**PokerBench.** Набор задач измеряет покерные знания/reasoning LLM, а не долгосрочную
game-theoretic силу в закрытом цикле. [PokerBench: Training Large Language Models to become
Professional Poker Players, 2025](https://arxiv.org/abs/2501.08328),
[код](https://github.com/pokerllm/pokerbench),
[dataset](https://huggingface.co/datasets/RZ412/PokerBench). Его можно использовать для
offline sanity check, но высокий accuracy нельзя преобразовывать в `bb/100`.

**ToolPoker / How Far Are LLMs from Professional Poker Players?** Работа показывает важное
для дизайна сравнение: vanilla LLM, LLM со структурированным контекстом и LLM с внешними
solver-like tools. [How Far Are LLMs from Professional Poker Players? An Investigation into
LLM Capabilities and Tool-Assisted Enhancement, ICLR
2026](https://arxiv.org/abs/2602.00528). Следующий LLM-эксперимент проекта должен повторять
именно такую декомпозицию, а не сравнивать один произвольный prompt с обученным агентом.

Из установленной литературы не следует, что свободный chain-of-thought сам по себе заменяет
CFR. Обоснованный гибрид выглядит так:

- движок формирует строгий детерминированный state;
- tools считают equity, ranges или blueprint advice;
- LLM выбирает только из legal actions по JSON Schema;
- полный prompt/response/fallback trace сохраняется;
- сила измеряется в той же duplicate arena, что и у остальных политик.

## 4. Frontier 2026: кандидаты на новизну, не готовые аксиомы

Следующие источники были доступны на дату среза, но часть из них — свежие preprints или
OpenReview submissions. Их идеи нужно перепроверять на малых играх и собственных ablations.

| Работа | Заявленное направление | Как использовать осторожно |
|---|---|---|
| [WEVA, 2026](https://arxiv.org/abs/2605.10900), [код](https://github.com/lbn187/WEVA) | opponent adaptation / exploitation | воспроизвести сначала в малой игре; сравнить с fixed mixture |
| [Parallel CFR, 2026](https://arxiv.org/abs/2605.14277) | ускорение CFR | измерять wall-clock, samples и exploitability, не только iterations |
| [Real-Time Parallel CFR, 2026](https://arxiv.org/abs/2605.19928) | параллельный real-time solving | рассматривать после профилирования последовательного reference |
| [TurboReBeL, 2026](https://openreview.net/forum?id=yMo7Z670f6) | более эффективный ReBeL-style learning/search | не начинать с него до воспроизводимого CFR/SD-CFR |
| [PokerSkill, 2026](https://arxiv.org/abs/2605.30094), [код](https://github.com/lbn187/PokerSkill) | structured deterministic context и skill library для LLM | заимствовать интерфейс/ablation, отдельно считать стоимость tools |
| [GTO Wizard AI Benchmark, 2026](https://arxiv.org/abs/2603.23660) | сравнение AI на solver-backed poker spots | использовать как внешний offline benchmark, не замену online match |

Отдельно [SpinGPT, 2025](https://arxiv.org/abs/2509.22387) относится к тому же быстро
меняющемуся LLM-poker frontier. До peer review и независимой репликации его следует считать
источником гипотез, а не evidence превосходства LLM над game-theoretic агентами.

### Возможная исследовательская ниша

Реалистичная новизна проекта находится на пересечении established и frontier:

> **SD-CFR blueprint + uncertainty-aware opponent model + confidence-gated exploitation,
> проверенные exact exploitability в малых играх и duplicate/LBR protocol в HUNL.**

LLM можно добавить как один из типов соперника и как интерпретируемый policy/tool layer.
Если LLM-компонент не улучшает primary metric после учёта tools и latency, это тоже полезный
результат: он отделяет языковое объяснение от стратегической силы.

## 5. Выбор инженерного стека

### 5.1. Основные зависимости

| Компонент | Назначение в проекте | Почему выбран |
|---|---|---|
| [PokerKit](https://github.com/uoftcprg/pokerkit), [статья](https://arxiv.org/abs/2308.07327) | HUNL state machine, payoffs, PHH | современная типизированная poker library; поддерживает воспроизводимые hand histories |
| [Poker Hand History](https://github.com/uoftcprg/phh-std) | формат обмена раздачами | машинно читаемый raw artifact, независимый от таблиц анализа |
| [PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) | быстрый showdown rank/equity sampling | отделяет hand evaluation от игровой логики; удобен для differential tests |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel), [статья](https://arxiv.org/abs/1908.09453) | CFR, best response, exploitability, Kuhn/Leduc | reference implementation и oracle для малых игр |
| SciPy/NumPy | pair-level tests, CI и power | зрелые численные примитивы; формулы остаются видимыми в коде |

Зависимости зафиксированы lockfile; обновление версии является изменением experiment
environment и требует нового manifest/validation.

### 5.2. Полезные, но не основные инструменты

- [LiteEFG](https://github.com/liumy2010/LiteEFG) — кандидат для быстрых экспериментов с
  extensive-form algorithms после OpenSpiel reference;
- [PokerRL](https://github.com/EricSteinberger/PokerRL) и официальный
  [Deep-CFR](https://github.com/EricSteinberger/Deep-CFR) — источники сверки алгоритма, но их
  старый dependency stack лучше не встраивать напрямую в основной runtime;
- [RLCard](https://github.com/datamllab/rlcard) — быстрые RL sanity checks, но не
  authoritative HUNL evaluator;
- [GTO Wizard Research API](https://github.com/gtowizard-ai/researcher-api-client) — внешний
  solver-backed benchmark при наличии доступа и соблюдении условий использования;
- [TexasSolver](https://github.com/bupticybee/TexasSolver) — внешний solver для spot checks;
  AGPL-лицензия требует отдельного решения и поэтому он не включён как core dependency.

### 5.3. Почему не продолжать только с DDQN

DDQN оптимизирует payoff против распределения опыта, но не минимизирует exploitability и
сталкивается с нестационарностью self-play. В старой реализации к этому добавляются
ошибки trajectory collection и evaluation contamination. DDQN допустим как negative/control
baseline после исправления, но CFR-family лучше соответствует математической структуре
двухсторонней zero-sum игры с неполной информацией.

Это не означает, что «RL не работает в покере»: ReBeL, AlphaHoldem и RL-CFR как раз
объединяют learning с game structure. Вывод уже: простой DQN над тремя действиями без
корректного information state и game-theoretic оценки — слишком слабый фундамент для
магистерской заявки.

## 6. Рекомендуемый порядок воспроизведения

1. OpenSpiel CFR/DCFR в Kuhn: проверить payoff и exploitability.
2. Собственная tabular реализация против OpenSpiel oracle.
3. Leduc Deep CFR и SD-CFR, минимум 5 training seeds.
4. PokerKit transparent baseline против scripted opponent suite.
5. HUNL SD-CFR blueprint с фиксированной action abstraction.
6. LBR/RL-BR и robust pool для оценки уязвимости.
7. Confidence-gated exploit module и ablations.
8. Strict-JSON LLM policy: vanilla → structured → tools → blueprint-assisted.

На каждом шаге сначала фиксируется metric/protocol, затем запускается модель. Такой порядок
делает отрицательный результат научно пригодным и не позволяет незаметно подогнать тест под
красивый график.

## 7. Матрица claims и необходимого evidence

| Claim | Минимально необходимое evidence |
|---|---|
| «Обыгрывает calling station» | preregistered duplicate run, one-sided lower CI `> 0` |
| «Практически сильное преимущество» | lower CI выше ненулевого preregistered margin |
| «Близка к равновесию в Kuhn/Leduc» | exploitability/NashConv против OpenSpiel oracle |
| «Безопасно адаптируется» | target gain + bounded degradation против robust pool/LBR |
| «LLM добавляет силу» | ablation с теми же tools, seeds, latency budget и legal-action contract |
| «Сильнее baseline» | одинаковые game rules, action space, fixed horizon и correction for multiplicity |

Эта матрица должна сопровождать все таблицы результатов в тексте диссертации.
