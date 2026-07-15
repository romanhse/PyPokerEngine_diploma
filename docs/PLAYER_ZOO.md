# Зоопарк игроков: что уже реализовано и как это исследовать

Дата среза: **15 июля 2026 года**.

Этот документ — карта всех реализаций игроков в каноническом слое `poker_research/`.
Здесь слово «игрок» означает объект, который видит только публичный `Observation` и
возвращает проверяемый `Decision`. Наличие игрока в каталоге означает, что реализация готова
к запуску и полезна как контроль, соперник или исследовательская гипотеза. Оно **не означает**,
что стратегия сильная, равновесная или уже прошла confirmatory-проверку.

На текущем срезе есть:

- 28 версионированных scripted policies в CLI-каталоге;
- общие конструкторы собственных sizing-, equity-, chart- и mixture-стратегий;
- два явно stateful подхода: rule-based adaptation и UCB meta-policy;
- безопасный NumPy MLP для inference из `.npz` checkpoint;
- behavior cloning trainer;
- исправленный Double DQN trainer;
- provider-independent strict-JSON адаптер для LLM.

Из них пока только `equity_value_v1` имеет отдельный замороженный confirmatory-результат —
против конкретного `calling_station_v1`. Для остальных реализация и тесты являются
инфраструктурой, а будущие результаты должны появляться только после pilot, freeze и
отдельного протокола.

## 1. Общий контракт

Все игроки входят в одну и ту же цепочку:

```text
PokerKit -> Observation -> Policy.decide() -> Decision -> legal-action validator -> PokerKit
```

Политика получает собственные карты, публичный board, позицию, стеки, ставки, pot, размер
call, SPR, публичную историю и уже построенный список legal actions. Она не получает raw
PokerKit state, порядок колоды, карты соперника или будущий board.

Для stochastic policy в `Observation` также есть `rng_key`: это отдельный, не связанный с
колодой идентификатор расписания случайности. Arena хранит `hand_id` для аудита, но runner
строит `rng_key` из заранее фиксированного `policy_schedule_seed`, номера пары/руки и leg.
Так изменение config hash или имени output directory не должно само по себе менять policy
draws. При сравнении stochastic policies замораживаются оба seed: deal `master_seed` и
`policy_schedule_seed`.

Каждый `Decision` обязан содержать:

1. одно выбранное legal action;
2. вероятность каждого и только каждого legal action;
3. конечные неотрицательные вероятности с суммой `1`;
4. только компактную audit metadata.

Общее action space фиксировано версией абстракции:

```text
fold, check_call, raise_min, raise_half_pot, raise_pot, raise_2pot, raise_all_in
```

Конкретное действие присутствует только когда PokerKit считает его легальным. Поэтому даже
патологический `jammer_v1` не может протолкнуть illegal all-in.

## 2. Двадцать восемь готовых scripted policies

Эти имена доступны через `poker-benchmark catalog` и через фабрики
`poker_research.catalog.policy_factories`. Все фабрики используют policy-local seed и
возвращают свежий объект.

### 2.1. Отрицательные контроли

| Policy | Что делает | Зачем нужна | Главная граница |
|---|---|---|---|
| `check_fold_v1` | Чекает бесплатно, иначе фолдит | Нижняя граница силы и smoke-test выигрыша | Намеренно ужасная стратегия |
| `random_valid_v1` | Равномерно выбирает из legal abstract actions | Проверка масок, stochastic path и воспроизводимости seed | Несколько raise sizes искусственно повышают aggression |

Отдельный API-примитив `CallCheckPolicy` (`call_check_v1`) также всегда чекает/коллирует.
Он нужен как простой переименовываемый компонент в programmatic training; каноническое имя
того же архетипа в CLI — `calling_station_v1`.

### 2.2. Базовые проверки чувствительности к размеру ставки

Эти игроки игнорируют карты и почти всегда применяют один профиль давления. Они изолируют
эффект sizing и помогают находить стратегии, которые корректно отвечают только на один тип
ставки.

| Policy | Приоритетный размер | Интерпретация |
|---|---:|---|
| `min_raiser_v1` | minimum legal raise | Много дешёвого давления |
| `half_pot_bettor_v1` | 0.5 pot | Малые value/bluff-like ставки без range logic |
| `pot_bettor_v1` | 1 pot | Средний card-blind pressure probe |
| `overbettor_v1` | 2 pot | Высокодисперсионный overbet probe |
| `jammer_v1` | all-in | Патологический тест call/fold и risk thresholds |

Если приоритетный размер недоступен, `BetSizingProfile` выбирает следующий legal raise; если
raise отсутствует, политика чекает/коллирует либо использует legal fallback. Общий
`BetSizingPolicy` позволяет задать свой упорядоченный профиль и вероятность raise, но новое
сочетание размеров требует нового versioned policy name.

### 2.3. Диагностические probes позиции, улицы, SPR и sizing

Эти восемь deterministic policies тоже полностью игнорируют силу закрытых карт. В отличие
от пяти fixed-sizing probes выше они меняют давление по одному публичному признаку. Это
контролируемые intervention tests: они показывают, реагирует ли проверяемый агент на позицию,
улицу, SPR или последовательность размеров. Высокий win rate такого probe не делает его
сильной стратегией — у него нет value/bluff ranges и его легко эксплуатировать после
идентификации правила.

| Policy | Триггер и действие | Что изолирует | Почему это не сильная стратегия |
|---|---|---|---|
| `button_bully_v1` | На button предпочитает 1-pot raise, вне позиции check/call | Чувствительность к позиции | Ставит с любыми картами и не строит positional ranges |
| `preflop_raiser_v1` | Только preflop minimum-raise, дальше check/call | Чистое preflop pressure | Нет hand selection и postflop плана |
| `flop_bettor_v1` | Только flop 0.5-pot raise | Реакцию на flop continuation pressure | Ставит весь range на одной улице |
| `turn_bettor_v1` | Только turn 1-pot raise | Реакцию на delayed turn pressure | Игнорирует board, equity и предыдущую линию |
| `river_overbettor_v1` | До river check/call, затем предпочитает 2-pot raise | Устойчивость к river overbet | Нет разделения value и bluffs |
| `postflop_pressure_v1` | Preflop check/call, на всех postflop streets 1-pot raise | Совокупное postflop pressure | Намеренно чрезмерная card-blind aggression |
| `spr_jammer_v1` | При публичном `SPR <= 2` предпочитает all-in | Поведение при малом SPR | Jam не зависит от private-card strength |
| `geometric_sizer_v1` | Min-raise preflop, 0.5 pot flop, 1 pot turn, 2 pot river | Реакцию на street-escalating sizing | Это фиксированная лестница, не оптимизированное geometric betting solution |

Когда целевой raise недоступен, probe выбирает legal check/call fallback. Поэтому эти
политики полезны ещё и как тест min/max sizing translation, но не добавляют evidence о Nash
или человеческом уровне игры.

### 2.4. Архетипы слабых и регулярных игроков

| Policy | Стиль | Реализация | Что она моделирует |
|---|---|---|---|
| `calling_station_v1` | loose-passive extreme | Всегда check/call | Игрока, который никогда не выбрасывает и не ставит |
| `maniac_v1` | hyper-aggressive | Около 82% raise, около 2% fold | Чрезмерную агрессию и крупные sizing |
| `loose_passive_v1` | loose-passive | Широкий equity-call, редкий малый value raise | Более мягкую версию calling station |
| `tight_passive_v1` | nit | Высокие thresholds, мало давления | Overfolder/нит-архетип |
| `tag_v1` | tight-aggressive | Position-aware equity thresholds, value bets, редкие bluffs | Объяснимый крепкий heuristic baseline |
| `lag_v1` | loose-aggressive | Более широкий continue, overbets, частые bluffs | Агрессивного и уязвимого к контрэксплуатации соперника |

Это разные операциональные типы, а не психологические диагнозы. Один calling station не
представляет всех «фишей», а один TAG не представляет всех сильных игроков.

### 2.5. Equity и value heuristics

| Policy | Основная логика | Назначение | Ограничение |
|---|---|---|---|
| `pot_odds_equity_v1` | Сравнивает equity против uniform legal range с pot odds и street thresholds | Прозрачный математический baseline | Модель range соперника почти всегда неточна |
| `equity_value_v1` | Не блефует, фолдит ниже pot-odds gate, ставит сильное equity на value | Специализированный exploit calling station | Не является robust или GTO стратегией |

Общий `EquityThresholdPolicy` позволяет зафиксировать отдельные call/raise thresholds по
улицам, bluff/value frequencies, button discount, jam gate и sizing profile. Любая новая
конфигурация — новая версия политики и новый pilot, а не незаметный tuning существующего
имени.

### 2.6. Композиции

| Policy | Состав | Для чего |
|---|---|---|
| `noisy_tag_v1` | `tag_v1` + 8% uniform exploration | Проверка чувствительности к action noise и stochastic evaluation |
| `balanced_mix_v1` | 60% TAG + 20% LAG + 20% nit на уровне distributions | Разнообразный frozen ensemble без обучения |

`MixedPolicy` усредняет полные distributions дочерних политик, а
`EpsilonExplorationPolicy` смешивает distribution ребёнка с uniform legal distribution.
Оба валидируют дочерний `Decision` до композиции. `balanced_mix_v1` медленнее одиночной
equity policy, потому что на каждом решении запускает три дочерних вычисления equity.

### 2.7. Preflop charts

| Policy | Preflop | Postflop | Роль |
|---|---|---|---|
| `chart_tag_v1` | Chen-style TAG thresholds | `pot_odds_equity_v1` | Объяснимая умеренно узкая range policy |
| `chart_lag_v1` | Более широкие open/defend thresholds | `lag_v1` | Широкий и агрессивный chart archetype |
| `chart_nit_v1` | Узкие continue/reraise thresholds | `tight_passive_v1` | Явный overfolder |

`PreflopChartPolicy` приводит две карты к одному из 169 starting-hand classes, считает
объяснимый Chen-style score, учитывает позицию и число raises, а postflop делегирует другой
policy. Это эвристическая таблица, не solver chart и не оценка точного preflop equity.

## 3. Каталог и готовые наборы

Полный JSON-каталог содержит family, style, stochastic/adaptive/trainable flags, оценку
скорости, краткое описание и явное ограничение каждого игрока:

```bash
uv run poker-benchmark catalog
uv run poker-benchmark catalog --suite quick
uv run poker-benchmark catalog --suite standard
uv run poker-benchmark catalog --suite extended
```

Наборы намеренно вложены друг в друга:

| Suite | Policies | Unordered matchups | Роль |
|---|---:|---:|---|
| `quick` | 6 | 15 | Быстрый smoke/pilot: controls, fish, maniac, TAG, exploit |
| `standard` | 18 | 153 | Основная scripted population, включая position/postflop/SPR diagnostics |
| `extended` | 28 | 378 | Все scripted политики, включая sizing, street, position, SPR и chart probes |

Статический JSON-каталог описывает только эти 28 scripted policies. Безопасный neural `.npz`
можно динамически зарегистрировать для league через `--checkpoint` и использовать как hero
или opponent в TOML benchmark. Imitation и DQN trainers, adaptive policies и LLM backend не
становятся скрытым состоянием каталога: обучение, checkpoint, backend и match-long state
передаются явно и получают отдельный provenance.

## 4. Stateless exploratory league

Минимальный прогон с resource-conscious defaults:

```bash
uv run poker-benchmark league \
  --suite quick \
  --pairs 10 \
  --output artifacts/exploratory-quick-p10
```

Выбор конкретных игроков:

```bash
uv run poker-benchmark league \
  --policies calling_station_v1 maniac_v1 tag_v1 equity_value_v1 \
  --pairs 25 \
  --master-seed 20260715 \
  --bootstrap-resamples 2000 \
  --output artifacts/exploratory-four-policy-p25
```

Safe neural checkpoint можно добавить к выбранному suite. Встроенное в `.npz` policy name
должно быть уникальным относительно scripted catalog:

```bash
uv run poker-benchmark league \
  --suite quick \
  --checkpoint artifacts/checkpoints/bc_tag_v1.npz \
  --pairs 10 \
  --output artifacts/exploratory-quick-plus-bc
```

Если одновременно указан `--policies`, имя neural policy нужно включить и в этот явный
список. League загружает `.npz` с `allow_pickle=False`; SHA-256 checkpoint попадает в
metadata каждого neural decision. Сам checkpoint exploratory league не копирует в output и
не включает отдельной записью в `manifest.json`, поэтому исходный `.npz` нужно сохранять
рядом с результатом. Полный hash/copy provenance для confirmatory run обеспечивает отдельный
TOML runner из раздела 6.1.

Для каждой неупорядоченной пары игроков лига создаёт `N` duplicate-пар. Внутри пары
используется одинаковая колода и swap seats. Каждая policy factory вызывается заново для
каждого leg: обучение и накопленная история между руками невозможны.

Лига пишет:

- `pairs.csv` — независимые duplicate-pair payoffs;
- `matchups.csv` — mean, standard deviation, standard error, two-sided 95% CI и status
  каждой пары стратегий;
- `leaderboard.csv` — pair-weighted агрегат только по полностью завершённым matchups;
- `hands.csv` — обе руки каждой успешной пары и сохранившиеся legs ошибочных пар с
  `pair_status`;
- `decisions.jsonl` — legal actions, requested/executed decision, probabilities, latency и
  audit metadata каждого решения;
- `hands.phhs` — replayable Poker Hand History для всех завершённых рук;
- `errors.jsonl` — все failed pairs, в том числе пустой файл при успехе;
- `summary.json` и `manifest.json` с Git/source/runtime/dependency provenance и SHA-256
  каждого перечисленного артефакта.

Лига **exploratory по определению**. Leaderboard удобен для поиска слабостей и выбора
следующего эксперимента, но не является множеством confirmatory claims: нет preregistration,
multiplicity correction и заранее замороженного набора сравнений. Если хотя бы одна
duplicate-пара завершилась ошибкой, соответствующий matchup получает `summary = None` и
целиком исключается из leaderboard statistics — intervals по surviving pairs этого matchup
не строятся. Runner продолжает остальные matchups и сохраняет run со статусом
`complete_with_errors`, но весь такой run не используется для inference/ranking: его
partial artifacts предназначены только для диагностики. Если leg A успел завершиться, а
leg B упал, первая рука сохраняется с `pair_status = "invalid_pair"`, но в `pairs.csv` и
статистику не входит. CLI при наличии ошибок по умолчанию завершает процесс с кодом `2` уже
после записи артефактов. Флаг `--allow-partial-debug` разрешает нулевой exit code только для
отладки и не делает partial run статистически валидным.

Serialization намеренно остаётся descriptive: CSV, `summary.json` и CLI показывают
mean/std/SE, two-sided t-CI и bootstrap CI, но не публикуют one-sided p-value или
`significant_win`. В `summary.json` это зафиксировано как
`inferential_statistics = "not_reported_exploratory_round_robin"`. Наличие CI не превращает
round-robin в confirmatory test; формальный one-sided criterion применяется только отдельным
preregistered runner с фиксированным horizon и test rule.

Stochastic draws теперь отделены от `hand_id`: league строит `policy_rng_key` из отдельного
`policy_schedule_seed`, canonical policy names, pair id и leg. При неизменных policy seeds и
schedule seed порядок входного mapping и config-derived run id не меняют расписание draws.
Сам `policy_schedule_seed` является частью протокола: его смена создаёт новый exploratory
run, а для planned common-random-number comparison он должен совпадать.

## 5. Почему adaptive match — отдельный режим

| Свойство | `run_league` | `run_adaptive_match` |
|---|---|---|
| Жизненный цикл policy | Новый объект на каждый leg | Тот же объект весь матч |
| Состояние между руками | Не сохраняется | Сохраняется явно |
| Terminal history/reward hooks | Не вызываются | Вызываются после обоих legs пары |
| Основная цель | Сравнение frozen/stateless policies | Online adaptation и meta-learning |
| Статистическая трактовка | Duplicate-пары можно суммировать как frozen observations | Последовательные пары зависимы и не i.i.d. |
| Артефакты | CSV/JSON manifest | Пока in-memory result |
| Допустимый статус сейчас | Exploratory; затем отдельный frozen confirmatory | Только exploratory до специального sequential protocol |

`run_adaptive_match` задерживает terminal callbacks до завершения двух legs одной колоды,
чтобы payoff первого физически сыгранного leg не изменил выбор во втором. Для компенсации
order effect физический порядок чередуется по `pair_id`: `("a", "b")`, затем `("b", "a")`;
он явно записывается в `AdaptivePairResult.play_order`. Однако сама policy может обновлять
документированное online-state во время `decide`, а между соседними парами состояние
намеренно сохраняется. Значит, decision-time история первого физически сыгранного leg
способна повлиять на второй ещё до terminal callbacks; чередование только балансирует этот
эффект, но не превращает policy в frozen duplicate. Обычный pair-level t-test для frozen
policy нельзя механически переносить на такую последовательность.
`AdaptiveMatchConfig.policy_schedule_seed` отдельно фиксирует stochastic draws, но не
устраняет эту последовательную зависимость state.

### 5.1. Rule-based opponent adaptation

`OpponentStatsTracker` строит только из публичной истории:

- VPIP и PFR;
- postflop aggression factor/frequency;
- fold-to-bet rate и число возможностей для оценки.

`AdaptiveExploitPolicy` до `minimum_hands=20` играет blueprint. После заранее заданного
минимума данных она может классифицировать соперника как loose-passive, overfolder или
aggressive и выбрать соответствующую дочернюю policy. Это rule-based confidence gate, а не
доказанная safe-exploitation гарантия.

Пример exploratory-матча:

```python
from poker_research.adaptive import AdaptiveExploitPolicy
from poker_research.adaptive_match import AdaptiveMatchConfig, run_adaptive_match
from poker_research.baselines import (
    LooseAggressivePolicy,
    TightAggressivePolicy,
    TightPassivePolicy,
)
from poker_research.policies import CallingStationPolicy, EquityValuePolicy

adaptive = AdaptiveExploitPolicy(
    TightAggressivePolicy(seed=11),
    loose_passive_policy=EquityValuePolicy(seed=12),
    overfolder_policy=LooseAggressivePolicy(seed=13),
    aggressive_policy=TightPassivePolicy(seed=14),
    name="adaptive_exploit_v1",
)
opponent = CallingStationPolicy()
result = run_adaptive_match(
    adaptive,
    opponent,
    config=AdaptiveMatchConfig(
        pair_count=50,
        master_seed=20260715,
        policy_schedule_seed=20260716,
    ),
)
```

Числа из `result.pairs` нельзя выдавать за confirmatory evidence без отдельного плана для
sequential/adaptive data и durable artifact writer.

### 5.2. Reward-based meta-policy

`UCBMetaPolicy` выбирает одну дочернюю policy на всю руку. Непроверенные arms пробуются
первыми, затем применяется UCB1 по terminal chip reward. `update_reward` вызывается внешним
adaptive runner ровно один раз для руки, в которой meta-policy принимала решение.

Это исследовательский baseline для online policy selection, не poker solver. Он зависит от
масштаба reward, порядка соперников и stationarity assumptions. Его нужно сравнивать с
фиксированной смесью тех же children, а не только с худшим arm.

## 6. Neural inference: безопасная общая граница

`ObservationEncoder` (`public_hunl_v1`) превращает только разрешённый `Observation` в
fixed-width vector. В него входят:

- семь видимых card slots: две hole cards и до пяти board cards;
- street и position;
- normalized pot, call, effective stack, own/opponent stacks и street bets;
- clipped SPR и длина истории;
- counts/last public action;
- legal-action mask;
- optional equity против uniform range.

`NumpyMLPPolicy` (`numpy_mlp_v1`) выполняет две ReLU hidden layers и action head, маскирует
illegal logits и работает greedy при `temperature=0`. При положительной temperature sampling
детерминированно привязан к policy seed и публичной идентичности решения.

Inference checkpoint — сжатый `.npz` без pickle. В нём лежат версия schema, encoder config,
feature names, policy name и float32 weights. Загрузка использует `allow_pickle=False` и
может потребовать заранее известный SHA-256:

```python
from pathlib import Path
from poker_research.neural import NumpyMLPPolicy

policy = NumpyMLPPolicy.from_checkpoint(
    Path("artifacts/checkpoints/candidate.npz"),
    expected_sha256="<frozen sha256>",
    temperature=0.0,
)
```

### 6.1. Neural checkpoint в league и confirmatory benchmark

Для confirmatory runner checkpoint указывается в `[hero]` или `[opponent]`. Относительный
путь разрешается относительно TOML-файла, а значение `policy` обязано совпасть с именем,
записанным внутри `.npz`:

```toml
[experiment]
name = "confirmatory-bc-tag-v1"
mode = "confirmatory"
master_seed = 2026071501
policy_schedule_seed = 2026071502
pair_count = 500
practical_margin_bb100 = 5.0
bootstrap_resamples = 50000
bootstrap_seed = 2026071503

[game]
starting_stack = 200
small_blind = 1
big_blind = 2
ante = 0

[hero]
policy = "bc_tag_v1"
checkpoint = "../artifacts/checkpoints/bc_tag_v1.npz"
seed = 101

[opponent]
policy = "calling_station_v1"
seed = 202
```

До первой руки runner считает SHA-256, включает его в config/run hash, загружает checkpoint
с hash guard и проверяет policy name. Затем он копирует байты в
`hero_checkpoint.npz`/`opponent_checkpoint.npz`, повторно проверяет hash и добавляет и
provenance, и frozen copy в `manifest.json`. Обычные PHH, hands, decisions, pairs и
`validate-run` остаются теми же, что для scripted policy. Это готовая инфраструктура
confirmatory evaluation, но не подтверждение силы конкретного BC/DQN checkpoint: такой run
ещё нужно preregister и выполнить.

`MLPWeights.random(...)` нужен для contract tests и инициализации trainer. Случайная сеть
не является осмысленным покерным игроком.

## 7. Imitation / behavior cloning

Behavior cloning даёт быстрый путь от прозрачного teacher к neural policy:

1. `RecordingTeacherPolicy` записывает только те же публичные observations, которые видел
   teacher;
2. `collect_demonstrations` играет teacher на обоих seats против выбранной population;
3. `split_demonstrations` делит **целые deal seeds**, не отдельные решения;
4. `train_behavior_cloning` обучает masked cross-entropy MLP в PyTorch;
5. итог экспортируется в безопасный NumPy `.npz` и запускается через `NumpyMLPPolicy`.

Готовая CLI-команда собирает данные против нескольких catalog opponents, обучает сеть и
пишет рядом JSON-отчёт с split, learning curve и SHA-256 checkpoint:

```bash
uv run poker-benchmark imitate \
  --teacher tag_v1 \
  --opponents calling_station_v1 maniac_v1 tight_passive_v1 \
  --deals-per-opponent 100 \
  --epochs 12 \
  --device auto \
  --policy-name bc_tag_v1 \
  --checkpoint artifacts/checkpoints/bc_tag_v1.npz \
  --report artifacts/checkpoints/bc_tag_v1.training.json
```

Минимальный exploratory pipeline:

```python
from pathlib import Path
from poker_research.baselines import LoosePassivePolicy, TightAggressivePolicy
from poker_research.imitation import (
    BehaviorCloningConfig,
    collect_demonstrations,
    train_behavior_cloning,
)
from poker_research.neural import ObservationEncoder

encoder = ObservationEncoder()
samples = collect_demonstrations(
    lambda policy_seed, seat: TightAggressivePolicy(seed=policy_seed),
    lambda policy_seed, seat: LoosePassivePolicy(seed=policy_seed),
    tuple(range(10_000, 11_000)),
    encoder=encoder,
    policy_seed=20260715,
)
result = train_behavior_cloning(
    samples,
    encoder,
    Path("artifacts/checkpoints/bc_tag_v1.npz"),
    config=BehaviorCloningConfig(device="auto", policy_name="bc_tag_v1"),
)
```

Validation accuracy измеряет сходство с teacher на held-out deals, но не poker strength.
Behavior cloning ограничено качеством и покрытием teacher dataset; после обучения checkpoint
нужно отдельно оценивать в duplicate league против held-out opponent population.

## 8. Исправленный Double DQN

`DoubleDQNTrainer` — новый research baseline, а не продолжение сломанного legacy DDQN. В
новой реализации:

- в replay попадает переход для **каждого** решения hero;
- non-terminal transition ведёт к следующему публичному решению hero, а terminal transition
  получает итоговый reward в BB;
- online network выбирает следующий legal action, target network оценивает его;
- legal masks применяются и при action selection, и в Bellman target;
- target network синхронизируется по фиксированному interval;
- action, replay, train-deal и validation-deal RNG streams разделены;
- stochastic policy schedule использует отдельный `policy_schedule_seed`, а не deal seed или
  audit `hand_id`;
- train/validation decks не пересекаются: namespaces различаются младшим битом seed;
- validation работает greedy, не добавляет replay и не делает optimizer steps;
- полный resumable trainer checkpoint хранит model/target/optimizer/replay/RNG/counters;
- frozen evaluation использует безопасный NumPy `.npz`.

Resource-conscious CLI-старт использует по кругу пять разных opponent archetypes, 5 000
training hands, 500 held-out-deck validation hands, replay 100 000 и MPS при его доступности:

```bash
uv run poker-benchmark train-dqn \
  --output artifacts/training/double-dqn-v1 \
  --device auto \
  --training-hands 5000 \
  --validation-hands 500
```

Вместо random Xavier initialization DQN можно начать с совместимого behavior-cloning или
другого NumPy MLP checkpoint:

```bash
uv run poker-benchmark train-dqn \
  --initial-checkpoint artifacts/checkpoints/bc_tag_v1.npz \
  --policy-name bc_tag_then_dqn_v1 \
  --output artifacts/training/bc-tag-then-dqn-v1 \
  --device auto
```

При warm-start trainer наследует encoder и weights из `.npz`; `hidden-size` должен совпадать
с checkpoint, а `starting-stack` и `big-blind` — с encoder config. Начальный path и SHA-256
попадают в `training.json`. Warm-start является отдельным training condition: сравнение с
random initialization требует одинакового hand/update budget и нескольких seeds, а само
наличие BC weights не доказывает улучшение DQN.

Для чистой random-init control ветки с **тем же самым encoder**, но без BC weights, используй
`--encoder-checkpoint` вместо `--initial-checkpoint`. Эти флаги взаимоисключающие:

```bash
uv run poker-benchmark train-dqn \
  --encoder-checkpoint artifacts/checkpoints/bc_tag_v1.npz \
  --policy-name dqn_random_same_encoder_v1 \
  --output artifacts/training/dqn-random-same-encoder-v1 \
  --device auto
```

Так action features, equity sampling seed и normalization совпадают между ablation-ветками;
различается только инициализация weights.

Output directory содержит `trainer.pt`, `policy.npz` и `training.json` с config, device,
training/validation summaries, per-opponent validation и обоими SHA-256. Команда не
перезаписывает непустой каталог. Validation использует отдельные decks и greedy policy, но
тех же opponent types; это alternating-seat diagnostic, не duplicate evaluation и не
held-out-opponent robustness test.

Тот же pipeline доступен программно:

```python
from pathlib import Path
from poker_research.dqn import DQNConfig, DoubleDQNTrainer
from poker_research.policies import CallingStationPolicy

def opponent_factory(policy_seed: int, seat: int) -> CallingStationPolicy:
    return CallingStationPolicy()

trainer = DoubleDQNTrainer(
    config=DQNConfig(
        device="auto",
        master_seed=20260715,
        replay_capacity=100_000,
        batch_size=128,
        min_replay_size=1_000,
    )
)
trainer.train_for(5_000, opponent_factory)
trainer.validate_for(500, opponent_factory)
trainer.save_training_checkpoint(Path("artifacts/checkpoints/dqn_trainer_v1.pt"))
sha256 = trainer.export_inference_checkpoint(
    Path("artifacts/checkpoints/double_dqn_v1.npz")
)
```

Файл `.pt` основан на PyTorch pickle и загружается только с явным `trusted=True` для
доверенного локального checkpoint. Для league/confirmatory evaluation используется `.npz`,
а не `.pt`.

Показанные `5_000/500` рук — smoke/pilot budget, а не обещание качества. Programmatic пример
выше использует одного соперника, а CLI — фиксированный круговой scripted pool. Текущий
trainer использует uniform replay; он ещё не реализует prioritized replay, population
self-play или game-theoretic safety. DQN оптимизирует payoff в своём training distribution и
сам по себе ничего не говорит об exploitability.

## 9. Strict-JSON LLM policy

`StrictJSONLLMPolicy` отделяет покерную логику от провайдера. Встроенный слой:

- сериализует только разрешённый `Observation` в canonical prompt;
- передаёт backend специализированную JSON Schema со списком текущих legal actions;
- требует конкретные `provider` и `model`, а от каждого успешного ответа — непустой
  `model_revision`; optional `expected_model_revision` превращает несовпадение в fallback;
- требует выбранное action и полный probability vector;
- отклоняет unknown fields, illegal action, пропущенные/лишние вероятности, NaN и сумму не
  равную `1`;
- ограничивает число попыток (`1` по умолчанию, максимум `3`) без рекурсивных retry;
- ограничивает общий match-long бюджет через `max_total_calls`, включая неуспешные попытки;
- после исчерпания attempts при transport/parse/revision/deadline/budget failure
  детерминированно выбирает check/call, затем fold, затем первое legal action;
- пишет отдельный audit каждого attempt и агрегаты: prompt/response hashes,
  provider/model/revision, latency, token usage, deadline/parse status и fallback reason;
  произвольный текст исключения не логируется, остаются только санитизированные type/code.

В репозитории пока **нет** provider-specific transport, API credentials или подтверждённого
онлайн LLM-прогона. Пользователь должен реализовать маленький `CompletionBackend.complete`.
Именно backend обязан оборвать transport по `timeout_seconds`: adapter может отметить поздний
синхронный ответ, но не способен безопасно прервать произвольный blocking Python call.
Backend не должен логировать ключи, а ответу не следует возвращать chain-of-thought — только
короткую причину и distribution.

LLM нельзя считать сопоставленной с локальными agents, пока не выполнена одна duplicate
ablation-сетка с одинаковыми rules/seeds:

```text
vanilla -> structured observation -> tools -> tools + frozen blueprint
```

В итог входят не только `bb/100`, но и cost, prompt/completion tokens, p50/p95 latency,
timeout/parse/fallback rate. Offline PokerBench accuracy не заменяет этот online protocol.

## 10. Настройки для MacBook M3 Pro, 36 GB

Проект не требует занимать всю доступную память. Разумная последовательность:

1. Установить training extra только для BC/DQN:

   ```bash
   uv sync --locked --extra dev --extra analysis --extra training
   ```

2. Для scripted league начать с `quick --pairs 10`. Это 15 matchups, 150 duplicate-пар и
   300 рук. `standard --pairs 10` — 153 matchups, 1 530 пар и 3 060 рук;
   `extended --pairs 10` — уже 378 matchups, 3 780 пар и 7 560 рук.
3. Equity policies в основном CPU-bound: MPS их автоматически не ускорит. Не запускать
   несколько extended leagues одновременно до профилирования.
4. Для BC/DQN оставить `device="auto"`: код выберет `mps`, если PyTorch его видит, иначе
   `cpu`. На маленьких batches CPU иногда конкурентен из-за overhead, поэтому сравнить
   wall-clock на коротком pilot полезнее, чем заранее завышать batch.
5. Базовые CLI DQN defaults — hidden size 128, batch 128, replay 100 000 transitions, 32
   equity samples и один update на transition. Это исходная точка, не приглашение сразу
   увеличивать сеть/replay.
6. После каждого training seed экспортировать `.npz`, освобождать trainer перед длинной
   evaluation и оценивать frozen checkpoint без MPS/PyTorch mutation.
7. MPS training не следует считать бит-в-бит переносимым между версиями PyTorch/macOS.
   Записывать runtime/device и использовать минимум несколько независимых training seeds.

## 11. Что exploratory, а что confirmatory

| Действие | Статус по умолчанию | Почему |
|---|---|---|
| `catalog` и одиночные hands | Debug | Проверяют contract, а не силу |
| Quick/standard/extended league | Exploratory | Много сравнений и выбор кандидатов после просмотра результатов |
| Tuning thresholds, charts, mixtures | Pilot/exploratory | Параметры меняются по данным |
| BC/DQN train и internal validation | Training/pilot | Выбирается checkpoint; руки не являются финальной оценкой |
| Adaptive match | Exploratory sequential | Policy меняется, пары зависимы |
| LLM prompt/backend shakeout | Debug/pilot | Возможны parse, latency и provider changes |
| Новый fixed-horizon TOML после freeze | Confirmatory | Safe `.npz` поддерживается для hero/opponent; policy/checkpoint hash, deal/schedule seeds, rules и test фиксируются заранее |

Переход к confirmatory требует нового output directory, clean tests/validator, отдельной
оценки sample size по pilot seeds, hash config/checkpoint и запрета на optional stopping.
Если после league изменился threshold, sizing, encoder, checkpoint или prompt, следующий
run снова exploratory.

## 12. Известные ограничения и следующий научный шаг

- Scripted zoo специально разнообразен, но восемь position/street/SPR/sizing probes
  card-blind и являются только диагностикой. Остальные сильнее выглядящие игроки остаются
  heuristics, часто с uniform-range equity. Zoo создаёт opponent population, а не заменяет
  solver.
- Общая fixed action abstraction пропускает многие реальные bet sizes; её изменение создаёт
  новую игру и требует отдельной ablation.
- League leaderboard pair-weighted и описательный. Он не измеряет worst-case robustness и
  не исправляет автоматически множественные сравнения.
- При любой league pair error весь run невалиден для inference/ranking. Диагностические
  error/artifact rows сохраняются, но не становятся partial evidence.
- Stochastic policy schedule отделён от deal и audit ids, но `policy_schedule_seed` всё равно
  нужно замораживать; смена seed является сменой протокола.
- Adaptive runner пока не пишет durable artifacts и не реализует sequential uncertainty.
- Decision-time update в adaptive policy может переносить информацию из первого duplicate
  leg во второй; одного delayed reward callback недостаточно для симметрии frozen duplicate.
- Behavior cloning может только приблизить teacher в покрытых information states.
- Double DQN не даёт Nash/exploitability guarantee и пока не имеет опубликованного
  обученного checkpoint или frozen evaluation result.
- Training CLI сохраняют config, metrics и hashes, но пока не формируют полный
  confirmatory manifest с Git SHA, PHH и decision-level trace. Их outputs являются
  exploratory training artifacts.
- NumPy MLP — безопасный inference format; случайная инициализация не является baseline
  силы.
- Safe NumPy checkpoint уже подключён к TOML benchmark с hash/copy provenance. Внешний LLM
  backend пока не имеет такой confirmatory integration.
- Strict-JSON LLM adapter не делает внешний provider детерминированным и пока не имеет
  costed online benchmark.
- Ни одна из новых реализаций не называется GTO, professional или superhuman без
  соответствующего evidence.

Правильное использование zoo — построить held-out opponent grid, выбрать несколько
разнообразных training seeds/checkpoints, а затем сравнить будущий SD-CFR blueprint,
DDQN/BC controls и adaptive correction на одинаковых duplicate decks. Для HUNL safety к
этому всё равно нужен LBR/RL-best-response proxy; победа в лиге scripted opponents его не
заменяет.

## 13. Как добавить нового игрока

1. Реализовать `Policy` и дать стабильное имя вида `name_v1`.
2. Использовать только `Observation`; случайность привязать к явному policy seed и
   `observation.rng_key`, не к deal seed или config-derived `hand_id`.
3. Всегда возвращать полный legal probability vector.
4. Зафиксировать action abstraction, thresholds/checkpoint/prompt version.
5. Добавить unit tests на hidden-information isolation, illegal actions, reproducibility и
   отсутствие mutation в evaluation.
6. Для stateless scripted policy добавить `PolicyDescriptor`, factory и suite membership в
   `catalog.py`.
7. Для safe NumPy checkpoint использовать существующие `checkpoints=`/`--checkpoint` и TOML
   `checkpoint` integrations; для нового backend/stateful state создать явную
   factory/config/runner integration и audit provenance.
8. Сначала запустить debug и pilot league, затем зафиксировать отдельный confirmatory
   protocol. Не переносить красивый exploratory leaderboard в таблицу основных claims.
