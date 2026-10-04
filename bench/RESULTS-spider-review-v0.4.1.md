# Spider review-response record: repair ablations, uncertainty, policy bypass

### Qwen/Qwen2.5-0.5B-Instruct — dev split, n = 1034

Device `mps` (torch.float16), greedy, max_tokens 128, seed 0; host: local dev: Apple M3 Max, 36 GB (unpinned); torch 2.13.0.

| arm | syntax-valid | executes | EX [95% CI] | retry fired | tok/query |
|:---|---:|---:|---:|---:|---:|
| unconstrained | 32.4% | 32.3% | **16.7%** [14.6, 19.1] | — | 55.3 |
| unconstrained+repair | 42.6% | 42.6% | **19.8%** [17.5, 22.4] | 67.6% | 99.4 |
| grid | 60.9% | 60.8% | **30.5%** [27.7, 33.3] | — | 33.5 |
| grid+repair | 61.7% | 61.6% | **30.8%** [28.0, 33.6] | 30.6% | 47.2 |
| grid+repair-generic | 61.7% | 61.6% | **30.8%** [28.0, 33.6] | 30.6% | 47.4 |

Paired comparisons (same questions; delta in percentage points):

| arm vs. baseline | metric | delta [95% bootstrap CI] | arm-only / base-only | McNemar p |
|:---|:---|---:|---:|---:|
| unconstrained+repair vs. unconstrained | executes | +10.3 [+8.4, +12.2] | 106 / 0 | <0.001 |
| unconstrained+repair vs. unconstrained | EX | +3.1 [+2.1, +4.2] | 32 / 0 | <0.001 |
| grid vs. unconstrained | executes | +28.5 [+25.0, +32.1] | 364 / 69 | <0.001 |
| grid vs. unconstrained | EX | +13.7 [+10.7, +16.6] | 203 / 61 | <0.001 |
| grid+repair vs. unconstrained | executes | +29.3 [+25.7, +32.8] | 371 / 68 | <0.001 |
| grid+repair vs. unconstrained | EX | +14.0 [+11.0, +16.9] | 206 / 61 | <0.001 |
| grid+repair vs. unconstrained+repair | executes | +19.1 [+15.4, +22.7] | 301 / 104 | <0.001 |
| grid+repair vs. unconstrained+repair | EX | +10.9 [+7.9, +13.9] | 191 / 78 | <0.001 |
| grid+repair vs. grid | executes | +0.8 [+0.3, +1.4] | 8 / 0 | 0.008 |
| grid+repair vs. grid | EX | +0.3 [+0.0, +0.7] | 3 / 0 | 0.250 |
| grid+repair-generic vs. grid | executes | +0.8 [+0.3, +1.4] | 8 / 0 | 0.008 |
| grid+repair-generic vs. grid | EX | +0.3 [+0.0, +0.7] | 3 / 0 | 0.250 |
| grid+repair vs. grid+repair-generic | executes | +0.0 [-0.5, +0.5] | 3 / 3 | 1.000 |
| grid+repair vs. grid+repair-generic | EX | +0.0 [-0.4, +0.4] | 2 / 2 | 1.000 |

Failure residue of `grid+repair` (716 of 1034 questions not EX-correct):

| bucket | count | share of all questions |
|:---|---:|---:|
| unknown / unbound column | 348 | 33.7% |
| executes, wrong result (semantic) | 319 | 30.9% |
| aggregate misuse | 23 | 2.2% |
| budget reached (reserve-completed or cut) | 19 | 1.8% |
| ambiguous column | 6 | 0.6% |
| other execution error | 1 | 0.1% |

### Qwen/Qwen2.5-7B-Instruct — dev split, n = 1034

Device `mps` (torch.float16), greedy, max_tokens 128, seed 0; host: local dev: Apple M3 Max, 36 GB (unpinned); torch 2.13.0.

| arm | syntax-valid | executes | EX [95% CI] | retry fired | tok/query |
|:---|---:|---:|---:|---:|---:|
| unconstrained | 91.0% | 91.0% | **52.7%** [49.7, 55.7] | — | 33.1 |
| unconstrained+repair | 95.5% | 95.5% | **54.4%** [51.4, 57.5] | 9.1% | 37.8 |
| grid | 91.1% | 91.1% | **53.5%** [50.4, 56.5] | — | 35.1 |
| grid+repair | 94.7% | 94.7% | **55.2%** [52.2, 58.2] | 7.6% | 39.4 |
| grid+repair-generic | 94.1% | 94.1% | **54.9%** [51.9, 57.9] | 7.6% | 39.2 |

Paired comparisons (same questions; delta in percentage points):

| arm vs. baseline | metric | delta [95% bootstrap CI] | arm-only / base-only | McNemar p |
|:---|:---|---:|---:|---:|
| unconstrained+repair vs. unconstrained | executes | +4.4 [+3.3, +5.8] | 46 / 0 | <0.001 |
| unconstrained+repair vs. unconstrained | EX | +1.7 [+1.0, +2.6] | 18 / 0 | <0.001 |
| grid vs. unconstrained | executes | +0.1 [-1.6, +1.8] | 40 / 39 | 1.000 |
| grid vs. unconstrained | EX | +0.8 [-0.8, +2.3] | 37 / 29 | 0.389 |
| grid+repair vs. unconstrained | executes | +3.7 [+1.9, +5.4] | 62 / 24 | <0.001 |
| grid+repair vs. unconstrained | EX | +2.5 [+1.0, +4.1] | 46 / 20 | 0.002 |
| grid+repair vs. unconstrained+repair | executes | -0.8 [-2.3, +0.8] | 27 / 35 | 0.374 |
| grid+repair vs. unconstrained+repair | EX | +0.8 [-0.7, +2.3] | 34 / 26 | 0.366 |
| grid+repair vs. grid | executes | +3.6 [+2.5, +4.7] | 37 / 0 | <0.001 |
| grid+repair vs. grid | EX | +1.7 [+1.0, +2.6] | 18 / 0 | <0.001 |
| grid+repair-generic vs. grid | executes | +3.0 [+2.0, +4.1] | 31 / 0 | <0.001 |
| grid+repair-generic vs. grid | EX | +1.5 [+0.8, +2.2] | 15 / 0 | <0.001 |
| grid+repair vs. grid+repair-generic | executes | +0.6 [-0.3, +1.5] | 13 / 7 | 0.263 |
| grid+repair vs. grid+repair-generic | EX | +0.3 [-0.3, +0.9] | 7 / 4 | 0.549 |

Failure residue of `grid+repair` (463 of 1034 questions not EX-correct):

| bucket | count | share of all questions |
|:---|---:|---:|
| executes, wrong result (semantic) | 408 | 39.5% |
| unknown / unbound column | 36 | 3.5% |
| ambiguous column | 10 | 1.0% |
| budget reached (reserve-completed or cut) | 7 | 0.7% |
| aggregate misuse | 2 | 0.2% |

### Qwen/Qwen2.5-0.5B-Instruct — test split, n = 2147

Device `mps` (torch.float16), greedy, max_tokens 128, seed 0; host: local dev: Apple M3 Max, 36 GB (unpinned); torch 2.13.0.

| arm | syntax-valid | executes | EX [95% CI] | retry fired | tok/query |
|:---|---:|---:|---:|---:|---:|
| unconstrained | 32.2% | 32.2% | **15.5%** [14.0, 17.1] | — | 55.0 |
| grid | 56.9% | 56.7% | **27.1%** [25.2, 29.0] | — | 36.4 |
| grid+repair | 58.2% | 58.0% | **27.2%** [25.4, 29.2] | 34.9% | 53.8 |

Paired comparisons (same questions; delta in percentage points):

| arm vs. baseline | metric | delta [95% bootstrap CI] | arm-only / base-only | McNemar p |
|:---|:---|---:|---:|---:|
| grid vs. unconstrained | executes | +24.5 [+22.0, +27.0] | 689 / 163 | <0.001 |
| grid vs. unconstrained | EX | +11.6 [+9.5, +13.6] | 375 / 127 | <0.001 |
| grid+repair vs. unconstrained | executes | +25.8 [+23.3, +28.3] | 710 / 156 | <0.001 |
| grid+repair vs. unconstrained | EX | +11.7 [+9.7, +13.7] | 377 / 125 | <0.001 |
| grid+repair vs. grid | executes | +1.3 [+0.8, +1.8] | 28 / 0 | <0.001 |
| grid+repair vs. grid | EX | +0.2 [+0.0, +0.4] | 4 / 0 | 0.125 |

Failure residue of `grid+repair` (1562 of 2147 questions not EX-correct):

| bucket | count | share of all questions |
|:---|---:|---:|
| unknown / unbound column | 775 | 36.1% |
| executes, wrong result (semantic) | 661 | 30.8% |
| budget reached (reserve-completed or cut) | 62 | 2.9% |
| aggregate misuse | 35 | 1.6% |
| ambiguous column | 22 | 1.0% |
| other execution error | 7 | 0.3% |

### Qwen/Qwen2.5-7B-Instruct — test split, n = 2147

Device `mps` (torch.float16), greedy, max_tokens 128, seed 0; host: local dev: Apple M3 Max, 36 GB (unpinned); torch 2.13.0.

| arm | syntax-valid | executes | EX [95% CI] | retry fired | tok/query |
|:---|---:|---:|---:|---:|---:|
| unconstrained | 92.9% | 92.9% | **53.7%** [51.6, 55.8] | — | 33.0 |
| grid | 90.5% | 90.5% | **52.7%** [50.6, 54.8] | — | 34.8 |
| grid+repair | 93.7% | 93.7% | **54.3%** [52.1, 56.4] | 7.1% | 38.4 |

Paired comparisons (same questions; delta in percentage points):

| arm vs. baseline | metric | delta [95% bootstrap CI] | arm-only / base-only | McNemar p |
|:---|:---|---:|---:|---:|
| grid vs. unconstrained | executes | -2.4 [-3.6, -1.2] | 59 / 111 | <0.001 |
| grid vs. unconstrained | EX | -1.0 [-2.1, +0.0] | 58 / 80 | 0.073 |
| grid+repair vs. unconstrained | executes | +0.8 [-0.4, +2.1] | 106 / 88 | 0.222 |
| grid+repair vs. unconstrained | EX | +0.5 [-0.7, +1.7] | 84 / 73 | 0.425 |
| grid+repair vs. grid | executes | +3.3 [+2.5, +4.0] | 70 / 0 | <0.001 |
| grid+repair vs. grid | EX | +1.5 [+1.0, +2.0] | 33 / 0 | <0.001 |

Failure residue of `grid+repair` (982 of 2147 questions not EX-correct):

| bucket | count | share of all questions |
|:---|---:|---:|
| executes, wrong result (semantic) | 847 | 39.5% |
| unknown / unbound column | 87 | 4.1% |
| ambiguous column | 21 | 1.0% |
| budget reached (reserve-completed or cut) | 18 | 0.8% |
| aggregate misuse | 5 | 0.2% |
| other execution error | 4 | 0.2% |

### Qwen/Qwen2.5-0.5B-Instruct — policy experiment (20 databases, n = 1034)

Role: SELECT-only, one forbidden table per database (the table its dev gold queries read most often). Violation = sqlite's authorizer observes a READ of the forbidden table while compiling the output. Attempted = the table appears as a FROM/JOIN/qualifier token in an output that does not compile.

| arm | subset | n | violations [95% CI] | attempted | compiles | EX |
|:---|:---|---:|---:|---:|---:|---:|
| unconstrained-instr | needs forbidden table | 654 | **34.1%** [30.6, 37.8] | 90.4% | 35.0% | 19.7% |
| unconstrained-instr | does not need it | 380 | **8.9%** [6.5, 12.2] | 40.0% | 34.7% | 20.0% |
| unconstrained-instr | all | 1034 | **24.9%** [22.3, 27.6] | 71.9% | 34.9% | 19.8% |
| unconstrained-hidden | needs forbidden table | 654 | **0.5%** [0.2, 1.3] | 7.3% | 15.1% | 1.5% |
| unconstrained-hidden | does not need it | 380 | **0.0%** [0.0, 1.0] | 0.3% | 42.6% | 20.5% |
| unconstrained-hidden | all | 1034 | **0.3%** [0.1, 0.8] | 4.7% | 25.2% | 8.5% |
| grid-hidden | needs forbidden table | 654 | **0.0%** [0.0, 0.6] | 0.0% | 43.3% | 2.8% |
| grid-hidden | does not need it | 380 | **0.0%** [0.0, 1.0] | 0.0% | 64.5% | 36.8% |
| grid-hidden | all | 1034 | **0.0%** [0.0, 0.4] | 0.0% | 51.1% | 15.3% |

### Qwen/Qwen2.5-7B-Instruct — policy experiment (20 databases, n = 1034)

Role: SELECT-only, one forbidden table per database (the table its dev gold queries read most often). Violation = sqlite's authorizer observes a READ of the forbidden table while compiling the output. Attempted = the table appears as a FROM/JOIN/qualifier token in an output that does not compile.

| arm | subset | n | violations [95% CI] | attempted | compiles | EX |
|:---|:---|---:|---:|---:|---:|---:|
| unconstrained-instr | needs forbidden table | 654 | **86.4%** [83.6, 88.8] | 96.3% | 87.8% | 50.2% |
| unconstrained-instr | does not need it | 380 | **5.5%** [3.6, 8.3] | 6.3% | 93.7% | 54.7% |
| unconstrained-instr | all | 1034 | **56.7%** [53.6, 59.7] | 63.2% | 89.9% | 51.8% |
| unconstrained-hidden | needs forbidden table | 654 | **22.5%** [19.4, 25.8] | 46.0% | 50.9% | 9.3% |
| unconstrained-hidden | does not need it | 380 | **1.3%** [0.6, 3.0] | 1.3% | 96.1% | 55.5% |
| unconstrained-hidden | all | 1034 | **14.7%** [12.7, 17.0] | 29.6% | 67.5% | 26.3% |
| grid-hidden | needs forbidden table | 654 | **0.0%** [0.0, 0.6] | 0.0% | 61.9% | 4.9% |
| grid-hidden | does not need it | 380 | **0.0%** [0.0, 1.0] | 0.0% | 92.9% | 55.3% |
| grid-hidden | all | 1034 | **0.0%** [0.0, 0.4] | 0.0% | 73.3% | 23.4% |

