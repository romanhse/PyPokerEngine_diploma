# Neural pilot S1: BC and matched Double-DQN initialization ablation

Status: **pilot / exploratory**. This report records model selection; none of its league
comparisons is a confirmatory claim.

## Training conditions

Behavior cloning used `equity_value_v1` as teacher against five opponents
(`calling_station_v1`, `maniac_v1`, `loose_passive_v1`, `tight_passive_v1`, `tag_v1`).
The 500 deal seeds were each played from both teacher seats, producing 3,305 public-only
decision demonstrations. The split kept complete deal seeds together: 400 train and 100
validation deals. After 12 MPS epochs, held-out action accuracy was 76.26%, with zero illegal
predictions.

Two Double-DQN branches then used the same public encoder, opponent order, master seed,
2,000 training hands, 250 held-out diagnostic hands and optimization budget:

| Branch | Initialization | Held-out mean, BB/hand |
|---|---|---:|
| `dqn_random_same_encoder_s1_v1` | seeded Xavier; BC encoder only | -0.636 |
| `dqn_bc_warm_s1_v1` | BC encoder and BC weights | +1.862 |

The held-out values alternate seats and use separate decks, but are not duplicate estimates.
They motivated a common duplicate-poker screen; they do not establish that warm-start is
better.

## Common duplicate league screen

The screen used 9 policies, 36 matchups and 30 duplicate pairs per matchup: 1,080 independent
pair observations / 2,160 hands. All pairs completed; `errors.jsonl` was empty. League output
is descriptive and intentionally omits one-sided p-values and significance labels.

| Policy | Pooled mean bb/100 | Descriptive 95% t-CI |
|---|---:|---:|
| `equity_value_v1` | +751.35 | [+478.82, +1023.89] |
| `tag_v1` | +259.06 | [+0.09, +518.03] |
| `bc_equity_population_s1_v1` | -43.02 | [-282.37, +196.33] |
| `dqn_bc_warm_s1_v1` | -62.50 | [-415.84, +290.84] |
| `dqn_random_same_encoder_s1_v1` | -112.81 | [-473.26, +247.63] |

The pooled intervals mix different opponents and are only screening summaries. Target-specific
calling-station rows were:

| Candidate | Mean bb/100 vs calling station | Descriptive 95% t-CI |
|---|---:|---:|
| `equity_value_v1` | +1138.33 | [+607.02, +1669.65] |
| `dqn_random_same_encoder_s1_v1` | +927.50 | [+63.67, +1791.33] |
| `dqn_bc_warm_s1_v1` | +405.00 | [-486.61, +1296.61] |
| `bc_equity_population_s1_v1` | +241.67 | [-25.45, +508.79] |

Warm-start beat random initialization head-to-head by +346.67 bb/100, but its interval
[-1199.13, +1892.46] is uninformative. The data therefore do **not** establish a warm-start
benefit. Random-DQN was selected only as the next calling-station candidate; this selection is
fully disclosed and all subsequent evidence must use fresh frozen seeds.

## Frozen artifacts and next test

| Artifact | SHA-256 |
|---|---|
| BC safe inference `.npz` | `aca46db789943b1520688993d6119bb97e028017fe1ffb4f1d385e284af1c4db` |
| random-DQN safe inference `.npz` | `5c157906c1ce5a23c0422e4abc01ebc1d897334ba9faa4c6aafbc5ea66312815` |
| warm-DQN safe inference `.npz` | `82f26e5221ef374d6b9c514811f5502c82cb14d6d084427056994b8fecffdc1d` |
| league manifest | `e89b1824ff80c4f6d72b8827b1ea61c93d8413457c4dec98587cd0d1031ecb4a` |
| league matchup table | `e8c2638878495bd3fad87e1a35a0694964c64b5c7b74f41a09f0d0702c559ad6` |

Using the target pilot standard deviation 2313.38 bb/100 and a planning effect of
+400 bb/100, the one-sided alpha=.05, power=.90 calculation requires 287 duplicate pairs.
The frozen follow-up rounds this to 300 pairs, uses a +5 bb/100 practical margin, fresh deal
and policy-schedule seeds, and is defined in
`configs/confirmatory_dqn_random_s1_vs_calling_station.toml`.

Trainer `.pt` files remain local because they are pickle-based and trusted-local-only. The
three safe evaluation `.npz` files and JSON training reports are versioned; exact MPS training
is deterministic in schedule but bitwise CPU/MPS identity is not claimed.
