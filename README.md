# Poker Research Lab

Воспроизводимый стенд для магистерского исследования покерных агентов: корректный
heads-up no-limit hold'em, duplicate evaluation, полные audit-логи, статистические гейты и
reference-контур CFR/exploitability.

Исходный `PyPokerEngine` и бакалаврские DDQN/LLM-эксперименты сохранены как legacy-слой.
Новые научные выводы строятся только через пакет `poker_research/`, потому что аудит нашёл в
старом HU-движке неправильные blinds/button и порядок postflop, а в DDQN — смешение train и
evaluation и отсутствие полноценных многошаговых переходов.

## Первый подтверждённый результат

Замороженный прозрачный агент `equity_value_v1` сравнивает equity с pot odds и делает
фиксированные value bets. На заранее отделённых от pilot данных против конкретного
`calling_station_v1` он получил:

| Метрика | Результат |
|---|---:|
| Duplicate-пары / руки | 500 / 1000 |
| Средний выигрыш | **+873.5 bb/100** |
| Односторонняя нижняя 95% граница | **+777.5 bb/100** |
| Preregistered practical margin | +5 bb/100 |
| Bootstrap 95% CI | [761.1; 990.0] bb/100 |
| Ошибки / исключённые строки | 0 / 0 |

Критерий `lower confidence bound > +5 bb/100` пройден. Это доказывает исправность первого
baseline и его преимущество над одним calling-station-архетипом. Это **не** доказательство
GTO-, professional- или superhuman-силы; следующий обязательный этап — популяция слабых
соперников и LBR/RL-best-response.

Полный self-contained отчёт: [reports/foundation_report/report.html](reports/foundation_report/report.html).
Точный исторический TOML сохранён с исходным SHA-256 `fe13325f...` в
[`configs/archive/confirmatory_equity_vs_calling_station_v1.toml`](configs/archive/confirmatory_equity_vs_calling_station_v1.toml).
Он предшествует strict v0.3 schema и не является шаблоном для повторного запуска.

## Архитектура

```text
PokerKit state machine
        ↓ только разрешённая игроку информация
typed Observation → frozen Policy → Decision + full probability vector
        ↓ legal-action validation
PokerKit transition → PHH / CSV / JSONL / immutable manifest
        ↓
duplicate-pair statistics (bb/100, t-CI, bootstrap, power)

OpenSpiel Kuhn/Leduc → CFR / exact best response / exploitability oracle
PHEvaluator         → fast deterministic equity estimates
```

- [PokerKit](https://github.com/uoftcprg/pokerkit) — authoritative HUNL rules, payouts и
  Poker Hand History.
- [PHEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) — быстрый hand evaluator.
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) — CFR и exact exploitability в
  малых играх.
- NumPy/SciPy — pair-level inference; Python 3.12 + `uv.lock` — воспроизводимая среда.

Action abstraction: `fold`, `check/call`, `min-raise`, `0.5 pot`, `1 pot`, `2 pot`,
`all-in` с PokerKit min/max validation. Политика никогда не получает raw engine state,
колоду или карты соперника.

## Зоопарк игроков и exploratory league

В новом контуре есть 28 готовых scripted policies: отрицательные контроли, пять базовых
fixed-sizing probes, восемь card-blind диагностик позиции/улицы/SPR/sizing, несколько
fish/regular archetypes, equity/value heuristics, mixtures и preflop charts. Новые probes —
stress tests, а не сильные стратегии. Отдельными API-модулями реализованы stateful
adaptation/UCB, безопасный NumPy MLP, behavior cloning, исправленный Double DQN и strict-JSON
LLM adapter.

```bash
uv run poker-benchmark catalog --suite extended
uv run poker-benchmark league --suite quick --pairs 10 \
  --output artifacts/exploratory-quick-p10
```

League пересоздаёт policy на каждый leg и предназначена для exploratory screening; adaptive
match сохраняет state между руками и требует отдельного sequential protocol. Полный каталог,
команды, lifecycle, M3/MPS defaults и ограничения: [docs/PLAYER_ZOO.md](docs/PLAYER_ZOO.md).
League публикует только descriptive mean/std/SE и two-sided/bootstrap CI; one-sided p-value и
`significant_win` намеренно отсутствуют и остаются частью отдельного preregistered benchmark.
Если league завершилась с любой pair error, её summaries и leaderboard считаются невалидными
для inference; сохранённые partial artifacts нужны только для диагностики.

Обучаемые baselines требуют `uv sync --locked --extra training`:

```bash
uv run poker-benchmark imitate \
  --checkpoint artifacts/checkpoints/behavior-cloning-v1.npz --device auto
uv run poker-benchmark train-dqn \
  --initial-checkpoint artifacts/checkpoints/behavior-cloning-v1.npz \
  --policy-name behavior_cloning_then_dqn_v1 \
  --output artifacts/training/behavior-cloning-then-dqn-v1 --device auto
uv run poker-benchmark league --suite quick \
  --checkpoint artifacts/training/behavior-cloning-then-dqn-v1/policy.npz --pairs 10 \
  --output artifacts/exploratory-quick-plus-neural
```

`--initial-checkpoint` — опциональный BC→DQN warm-start; без него DQN начинает со seeded
Xavier weights. Оба варианта являются exploratory training conditions и требуют отдельного
сравнения на нескольких seeds.

## Быстрый старт

Нужен [uv](https://docs.astral.sh/uv/) и Python 3.12.

```bash
uv sync --locked --extra dev --extra game-theory
uv run ruff check poker_research tests/research tests/legacy_compat.py
uv run mypy poker_research
uv run pytest
```

Эти команды являются обязательным clean gate; актуальное число тестов фиксируется в CI
конкретного коммита, а не поддерживается вручную в README.

Pilot:

```bash
uv run poker-benchmark run \
  configs/pilot_equity_vs_calling_station.toml \
  --output artifacts/my-pilot
```

Текущий confirmatory-файл — новый v2 protocol с ранее не сыгранными seeds. Его нельзя
запускать до принятия v2 pilot, clean commit и окончательного freeze:

```bash
uv run poker-benchmark run \
  configs/confirmatory_equity_vs_calling_station.toml \
  --output artifacts/my-confirmatory
```

Замороженный safe neural `.npz` поддерживается тем же runner. В новом preregistered TOML
нужно указать отдельный schedule seed и checkpoint для нужной роли:

```toml
[experiment]
mode = "confirmatory"
policy_schedule_seed = 2026071502

[hero]
policy = "behavior_cloning_then_dqn_v1"
checkpoint = "../artifacts/training/behavior-cloning-then-dqn-v1/policy.npz"
seed = 101
```

Это фрагмент полного strict-schema config: confirmatory-файл также обязан явно задавать
`name`, `master_seed`, `pair_count`, practical margin и bootstrap settings; неизвестные
секции/ключи отвергаются. Policy name должен совпадать с именем внутри checkpoint.
Относительный путь считается от TOML-файла; SHA-256 checkpoint входит в run hash.

Runner отказывается перезаписывать непустой каталог. Он сохраняет:

- `preregistered_config.toml`;
- `pairs.csv` и `hands.csv`;
- decision-level `decisions.jsonl`;
- replayable `hands.phhs`;
- `summary.json`, пустой/непустой `errors.jsonl`;
- optional `hero_checkpoint.npz`/`opponent_checkpoint.npz`, если роль задана через safe
  checkpoint;
- `manifest.json` с config, Git/runtime/dependency provenance и SHA-256 каждого артефакта.

Stochastic action schedule отделён от колоды и audit `hand_id`: runner фиксирует
`policy_schedule_seed` и пишет `policy_rng_key`. Для воспроизводимого сравнения нельзя менять
ни deal seed, ни policy schedule seed после freeze.

Независимая проверка готового каталога:

```bash
uv run poker-benchmark validate-run artifacts/my-confirmatory
```

Оценка требуемого числа duplicate-пар:

```bash
uv run poker-benchmark power \
  --standard-deviation 1302 \
  --effect 100 \
  --alpha 0.05 \
  --power 0.90
```

Exact game-theory sanity check:

```bash
uv run --extra game-theory poker-benchmark theory \
  --game kuhn_poker \
  --algorithm cfr_plus \
  --iterations 10000 \
  --report-every 500 \
  --output artifacts/cfr-plus-kuhn.csv
```

Reference run достиг exact exploitability `9.63e-6` на 10 000 итерациях CFR+.

## Структура

```text
poker_research/
  arena.py          PokerKit HU arena, action abstraction, invariants, PHH
  types.py          engine-independent Observation / Decision / Policy
  equity.py         deterministic PHEvaluator equity
  policies.py       frozen calling-station и equity/value baselines
  baselines.py      controls, sizing/position/street/SPR probes, fish/TAG/LAG heuristics
  preflop.py        explainable Chen-style chart policies
  catalog.py        versioned policy descriptors, factories и suites
  league.py         stateless exploratory duplicate round-robin
  adaptive.py       mixtures, opponent stats, adaptive/UCB policies
  adaptive_match.py sequential stateful duplicate matches
  neural.py         public encoder и safe NumPy `.npz` inference
  imitation.py      public-only behavior cloning pipeline
  dqn.py            corrected Double DQN training/export baseline
  llm_policy.py     provider-neutral strict-JSON LLM adapter
  experiment.py     duplicate runner, manifests, CSV/JSONL/PHH
  statistics.py     confidence intervals, bootstrap, power
  game_theory.py    OpenSpiel CFR/exploitability verification
  cli.py            poker-benchmark CLI

configs/             frozen pilot и confirmatory protocols
tests/research/      research/invariant/e2e tests без вручную зафиксированного счётчика
docs/                roadmap и обзор первичной литературы
reports/             portable technical report
pypokerengine/       legacy; не authoritative для новых HU claims
examples/players/    legacy DDQN/LLM agents
```

## Исследовательская программа

Реалистичная тема магистерской:

> Нейросетевое минимизирование сожаления и безопасная адаптация к ограниченно рациональным
> соперникам в heads-up no-limit Texas hold'em.

Порядок работ:

1. CFR/DCFR в Kuhn и Leduc с exact exploitability против OpenSpiel oracle.
2. Deep CFR/SD-CFR blueprint и минимум 5 независимых training seeds.
3. HUNL action abstraction, population self-play и held-out duplicate grid.
4. Opponent model + confidence-gated exploit module.
5. Fish suite, LBR/RL-BR, latency/memory/illegal-action guardrails и ablations.
6. LLM-соперник через strict JSON policy adapter: vanilla → structured → tools →
   blueprint-assisted.

Обоснование методологии и источники: [дорожная карта](docs/RESEARCH_ROADMAP.md) и
[обзор литературы](docs/LITERATURE_REVIEW.md). Постоянные инструкции для следующих Codex
итераций находятся в [AGENTS.md](AGENTS.md).

## Docker

```bash
docker compose build
docker compose run --rm poker-research \
  uv run poker-benchmark run configs/pilot_equity_vs_calling_station.toml \
  --output artifacts/docker-pilot
```

`artifacts/` намеренно исключён из Git. Для диссертационных запусков raw artifacts следует
публиковать отдельно (release/object storage/DVC) вместе с `manifest.json`; агрегированный
evidence snapshot уже встроен в HTML-отчёт.
