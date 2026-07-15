# Инструкции агентам проекта

Этот файл действует на весь репозиторий. Цель любых новых изменений — не «получить красивый
график», а сохранить воспроизводимую цепочку от правил игры и политики до статистического
вывода.

## 1. Что прочитать перед работой

1. `docs/RESEARCH_ROADMAP.md` — гипотезы, протокол и этапы.
2. `docs/LITERATURE_REVIEW.md` — научный фундамент и границы claims.
3. `pyproject.toml` и соответствующий код в `poker_research/`.
4. Preregistered TOML, если задача касается эксперимента.

Если код и документация расходятся, не выбирай удобную версию молча: зафиксируй расхождение,
исправь его до запуска либо пометь результат exploratory.

## 2. Границы архитектуры

- `poker_research/` — канонический исследовательский слой.
- `pypokerengine/`, `examples/players/` и исторические таблицы/графики — legacy бакалаврской
  работы. Не используй их для новых claims и не переписывай без явно поставленной migration
  задачи.
- PokerKit — authoritative state machine и источник выплат/PHH.
- PHEvaluator — hand rank/equity helper, но не движок ставок.
- OpenSpiel — reference для CFR, best response и exploitability в малых играх.
- Политика работает только через `Observation` и возвращает `Decision`; передавать ей raw
  PokerKit state, deck, карты соперника или будущие карты запрещено.

Сохраняй пользовательские незакоммиченные изменения. Не удаляй исторические артефакты и не
используй destructive Git-команды.

## 3. Среда и обязательные проверки

Используй Python 3.12 и `uv.lock`:

```bash
uv sync --locked --extra dev --extra analysis
uv run ruff check poker_research tests/research
uv run mypy poker_research
uv run pytest --cov=poker_research --cov-report=term-missing
```

Для game-theory задач:

```bash
uv sync --locked --extra dev --extra game-theory
```

Не добавляй зависимость, если задача решается stdlib или уже установленной библиотекой.
Новая core-зависимость требует: обоснования, фиксированной версии/диапазона, проверки
лицензии, обновления lockfile и CI.

## 4. Правила реализации политики

Новая политика обязана:

1. реализовать `Policy` из `poker_research/types.py`;
2. выбирать только действие из `observation.legal_actions`;
3. возвращать вероятность для каждого и только каждого legal action;
4. возвращать конечные неотрицательные вероятности с суммой `1`;
5. не изменять weights, replay, prompt или checkpoint во время evaluation;
6. иметь стабильное имя с версией, например `sd_cfr_v1`;
7. использовать явный seed для любой случайности;
8. записывать только компактную audit metadata, необходимую для воспроизведения решения;
9. проходить тесты на отсутствие hidden-information leakage и illegal actions.
10. использовать `observation.rng_key` вместе с явным policy seed для stochastic draw; не
    привязывать policy RNG к deal seed, config-derived `hand_id` или скрытому state.

Размеры ставок являются частью версии action abstraction. Изменение `0.5 pot` на `0.75 pot`
создаёт новую игру/версию политики и требует отдельного benchmark.

### 4.1. Каталог, league и stateful policies

- Для новой stateless scripted policy обнови descriptor, factory и нужный suite в
  `poker_research/catalog.py`; каталог и фактическое имя фабрики должны совпадать.
- `button_bully_v1`, street probes, `spr_jammer_v1` и `geometric_sizer_v1` — card-blind
  diagnostic interventions. Не называй их сильными игроками и не объединяй их результаты с
  fish/regular population без заранее заданной роли; они измеряют отдельную чувствительность
  к position/street/SPR/sizing.
- `run_league` создаёт свежую policy на каждый leg и является exploratory screening. Не
  включай туда adaptive/UCB policy так, будто её match-long state сохраняется.
- Safe neural `.npz` подключай к exploratory league через `checkpoints=`/`--checkpoint`;
  embedded policy name должен быть уникальным. SHA-256 пишется в metadata каждого neural
  decision, но league не копирует `.npz` и не регистрирует его отдельным artifact manifest;
  сохраняй исходный checkpoint вместе с exploratory result.
- Любой league run с непустым `errors.jsonl`, `complete_with_errors` или failed pair целиком
  невалиден для inference/ranking. Partial artifacts хранятся только для диагностики; не
  цитируй surviving summaries, CI или leaderboard как evidence.
- League serialization/CSV/CLI оставляй descriptive: mean, std, SE, two-sided t-CI и
  bootstrap CI. Не публикуй one-sided p-value или `significant_win`; `summary.json` должен
  явно содержать `inferential_statistics = "not_reported_exploratory_round_robin"`.
- Stateful adaptation запускай через `run_adaptive_match`; его последовательные pair results
  не i.i.d. и пока не подходят под обычный confirmatory t-test frozen policy. В текущем
  runner физический порядок legs чередуется AB/BA и пишется в `play_order`, но decision-time
  state из первого сыгранного leg всё равно может повлиять на второй, несмотря на delayed
  terminal callbacks.
- Trainer checkpoint `.pt/.pth` загружай только из доверенного локального источника. Для
  league/evaluation экспортируй immutable `.npz`, зафиксируй SHA-256 и используй greedy
  inference без mutation.
- Для confirmatory neural evaluation укажи `.npz` в `hero.checkpoint` или
  `opponent.checkpoint` TOML. Runner обязан проверить совпадение embedded policy name,
  включить SHA-256 в run hash, скопировать checkpoint в output и добавить его в manifest.
- Актуальная карта готовых игроков, lifecycle и known limitations находится в
  `docs/PLAYER_ZOO.md`; обновляй её вместе с публичным policy API.

## 5. Режимы эксперимента

Каждый запуск заранее получает один статус:

- **debug** — проверка кода; результаты не сравниваются;
- **pilot/exploratory** — разрешены tuning и оценка дисперсии; нельзя выдавать за
  подтверждение гипотезы;
- **confirmatory** — config, policy/checkpoint, seeds, horizon и test rule заморожены до
  запуска.

TOML loader использует закрытую schema: `experiment.mode` обязателен, неизвестные
секции/ключи и нестрогие типы отвергаются. Для `confirmatory` явно указывай practical margin,
bootstrap resamples/seed и `policy_schedule_seed`, не полагайся на defaults.

Перед confirmatory run:

1. проверь clean tests и data validator;
2. рассчитай sample size на отдельных pilot seeds;
3. зафиксируй config и hash checkpoint в Git/manifest;
4. создай новый output directory;
5. не просматривай промежуточный p-value и не останавливай run по результату.

Deal `master_seed` и `policy_schedule_seed` — разные части протокола. Первый задаёт колоды,
второй — stochastic policy draws через `policy_rng_key`; замораживай и логируй оба.

Runner намеренно не перезаписывает непустые каталоги. Не обходи эту защиту. Failed run и
его `errors.jsonl` сохраняются; нельзя удалять неудачные руки и анализировать остаток как
полный confirmatory sample.

Базовые команды:

```bash
uv run poker-benchmark run \
  configs/pilot_equity_vs_calling_station.toml \
  --output artifacts/pilot-equity-v1

uv run poker-benchmark power \
  --standard-deviation 120 \
  --effect 10 \
  --alpha 0.05 \
  --power 0.90

uv run poker-benchmark run \
  configs/confirmatory_equity_vs_calling_station.toml \
  --output artifacts/confirmatory-equity-v1
```

Confirmatory-команду не запускай, если после последнего pilot менялись policy thresholds,
game rules, dependencies или analysis code без нового freeze.

## 6. Статистика и данные

- Независимая единица HUNL-анализа — duplicate-пара: одна колода, swap seats.
- Primary estimand — pair-level `bb/100`.
- Primary test — фиксированный one-sided t-test; успех, если lower 95% bound выше
  preregistered margin.
- Bootstrap — sensitivity analysis, а не способ выбрать более выгодный интервал.
- Не считай отдельные решения, руки или legs независимыми наблюдениями.
- Не объединяй training и evaluation, pilot и confirmatory, разные правила или версии
  политики в одну primary таблицу.
- Для нескольких secondary comparisons используй preregistered multiplicity correction.
- Для обучаемого агента показывай минимум несколько независимых training seeds и разброс
  между ними.
- Отчёт всегда содержит эффект и uncertainty, а не только p-value.

Проверяй до анализа:

- два legs на каждый `pair_id`;
- одинаковые `deal_seed` и `deck_hash`, противоположные seats;
- наличие ожидаемого `policy_rng_key` из замороженного schedule seed для каждого leg;
- `sum(payoffs) == 0` и conservation of chips;
- отсутствие duplicate/missing rows;
- полную цепочку legal action → requested decision → executed action;
- ноль parse, illegal-action, timeout и fallback событий для обычных локальных политик.

Не редактируй CSV/JSONL вручную. Любая производная таблица должна создаваться скриптом из
raw artifacts и ссылаться на manifest/checksums.

## 7. Обучение CFR/нейросетей

- Начинай с Kuhn/Leduc и сверяй exploitability с OpenSpiel.
- Разделяй RNG streams для traversal, network initialization, replay sampling и evaluation
  decks.
- BC→DQN warm-start через safe `.npz` является отдельным training condition. Проверяй
  совпадение encoder schema, hidden size, starting stack и big blind; сохраняй initial
  checkpoint SHA-256 и сравнивай с random initialization при одинаковом budget и нескольких
  seeds. Для random-init ablation с тем же frozen encoder используй `--encoder-checkpoint`,
  а для warm-start weights — взаимоисключающий `--initial-checkpoint`.
- Checkpoint содержит config hash, code/Git SHA, weights, optimizer state и RNG states.
- Train/eval action abstraction и information-state schema должны совпадать и иметь версию.
- Не выбирай «лучший seed» для итогового результата.
- Не называй HUNL strategy равновесной без измеримой exploitability; для HUNL указывай
  empirical opponent pool и LBR/RL-BR proxy.
- Любая safe-exploitation работа сравнивает target gain и safety degradation относительно
  одного замороженного blueprint.

## 8. LLM-агент

LLM adapter подчиняется обычному `Policy` contract и дополнительно обязан:

- использовать strict JSON Schema, фиксированную версию prompt и список legal actions;
- требовать конкретные provider/model и непустой model revision ответа; при необходимости
  фиксировать `expected_model_revision`, temperature, seed, timeout и output limits;
- валидировать полный probability vector до исполнения;
- иметь bounded attempts и общий `max_total_calls` с deterministic fallback; рекурсивные
  retry запрещены;
- логировать per-attempt и aggregate audit: prompt hash/version, полученный JSON, revision,
  parse/deadline status, latency, usage и fallback reason. Произвольные exception messages,
  ключи и секреты не логировать;
- требовать, чтобы backend сам обеспечивал transport timeout: adapter не может безопасно
  прервать произвольный blocking synchronous call;
- не отправлять hidden cards/deck/internal state;
- не менять prompt после просмотра confirmatory hands;
- записывать фактически исполненное действие отдельно от ответа модели.

Для сравнения LLM используй ablation: vanilla, structured observation, tools, tools +
blueprint. Стоимость tools, token usage, latency и fallback rate входят в отчёт. Accuracy на
offline question benchmark не заменяет online duplicate `bb/100`.

## 9. Научная добросовестность

- Для текущих claims ищи первичные источники: статьи, официальную документацию и официальные
  репозитории. Обновляй `docs/LITERATURE_REVIEW.md`, указывая дату среза.
- Отделяй peer-reviewed results от arXiv/OpenReview preprints.
- Не копируй формулировку «superhuman» на собственную модель без эквивалентного протокола.
- Отрицательный результат не скрывается: он фиксируется вместе с config и uncertainty.
- Любая найденная утечка, contamination или ошибка evaluator обнуляет соответствующий claim
  до повторного запуска.
- Calling station — один конкретный opponent archetype, не синоним всех слабых игроков.

## 10. Definition of done

Изменение завершено, когда:

1. код и schema versioned;
2. добавлены тесты на happy path и критические failure modes;
3. lint, mypy и pytest проходят;
4. seeds и policy state не мутируют во время evaluation;
5. создан полный audit trail либо явно указано, что запуск не выполнялся;
6. документация и команды соответствуют коду;
7. итоговое сообщение перечисляет изменённые файлы, проведённые проверки и ограничения;
8. claim сформулирован не шире, чем позволяет evidence.
