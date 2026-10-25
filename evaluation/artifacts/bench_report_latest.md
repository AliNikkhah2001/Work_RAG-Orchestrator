# Work Credit RAG — Benchmark Report

> Generated 2026-10-14 09:52 UTC · model: ics-helper-agent · total turns 41 · conversations 12 · avg latency 4938 ms · error rate 0.0% · judge gemma-4-31b

Artifacts: `bench_results_latest.json` + `ragas_results_latest.json` → `bench_report_latest.html` / `.md`

## 1. Overall Metrics (0–1, higher is better; hallucination inverted)

| Metric | Score |
|---|---:|
| faithfulness | 0.461 |
| answer_relevancy | 0.568 |
| context_relevancy | 0.283 |
| context_recall | 0.340 |
| hallucination (inverted = 1 − hallucination 0.254) | 0.746 |
| ambiguity_resolution | 1.000 |
| sticky_persistence | 0.683 |
| retrieval_hit | 0.780 |

## 2. Per-Subspace Comparison

> Columns: count, faithfulness, answer_relevancy, context_relevancy, hallucination (inv), ambiguity, sticky, retrieval_hit, p50 latency

| Subspace | count | faith | ans_rel | ctx_rel | hall_inv | amb | sticky | hit | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cheque | 9 | 0.422 | 0.400 | 0.267 | 0.667 | 1.000 | 0.444 | 0.667 | 5420 ms |
| facilities_person | 6 | 0.967 | 0.500 | 0.367 | 0.500 | 1.000 | 0.667 | 0.500 | 9376 ms |
| facilities_legal | 3 | 0.917 | 0.100 | 0.067 | 0.667 | 1.000 | 0.333 | 0.667 | 7402 ms |
| general | 10 | 0.655 | 0.670 | 0.160 | 0.900 | 1.000 | 0.600 | 0.800 | 6981 ms |
| **overall** | 41 | 0.461 | 0.568 | 0.283 | 0.746 | 1.000 | 0.683 | 0.780 | 4872 ms |

## 3. Per-Conversation Drill-Down

### conv_cheque_01 — greeting -> cheque granular -> cheque sticky follow-up [cheque]

elapsed 15567 ms · turns 3

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | سلام | greeting → stop ✓ | — → ∅ — | 1104 ms | 0.000 | سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتبا | 0 |
| 2 | چک برگشتی چیست و چگونه رفع سوء اثر می‌شود؟ | stop → stop ✓ | cheque → ∅ ✗ | 5420 ms | 0.000 | بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد. | 5 chunks: کد دلیل: DCH3-6 متن دلیل: فرد در یک ماه گذشته سابقه رفع سوء  |
| 3 | اگر چکم بدون موجودی باشد چه کار کنم؟ | stop → stop ✓ | cheque → ∅ ✗ | 7964 ms | 0.600 | برگشت خوردن چک به دلیل فقدان یا کمبود موجودی، به طور مستقیم بر محاسبه امتیاز اعت | 5 chunks: سوال: کدوم دلایل برگشت چک توی امتیاز مهمه؟ پاسخ کوتاه: چک‌ها |

### conv_cheque_02 — scattered clarify -> cheque spec -> cheque sticky granular [cheque]

elapsed 18420 ms · turns 3

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | وضعیت اعتباری چطور است | clarify → clarify ✓ | — → ∅ — | 3056 ms | 0.000 | برای راهنمایی دقیق‌تر، ممکن است بفرمایید منظورتان بررسی وضعیت اعتباری برای اشخاص | 5 chunks: کد دلیل: OTN1 متن دلیل: فرد در ماه‌هایی که وضعیت منفی داشته  |
| 2 | چک | stop → stop ✓ | cheque → cheque ✓ | 1125 ms | 0.000 | حوزه «گزارش اعتباری چک» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |
| 3 | تاثیر چک برگشتی بر امتیاز اعتباری چقدر است؟ | stop → stop ✓ | cheque → cheque ✓ | 13288 ms | 0.800 | سابقه چک‌های برگشتی حیاتی‌ترین عامل در تعیین امتیاز اعتباری چک یک شخص است. به طو | 5 chunks: سوال: آیا چک‌های ضمانت که در وجه بانک صادر شده و برگشت خورده |

### conv_facilities_person_01 — clarify -> facilities_person -> sticky -> cheque re-spec [facilities_person]

elapsed 15054 ms · turns 4

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | وضعیت اعتباری چطور است | clarify → clarify ✓ | — → ∅ — | 2307 ms | 0.000 | برای راهنمایی دقیق‌تر، ممکن است بفرمایید منظورتان بررسی وضعیت اعتباری برای اشخاص | 5 chunks: کد دلیل: OTN1 متن دلیل: فرد در ماه‌هایی که وضعیت منفی داشته  |
| 2 | تسهیلات اشخاص حقیقی | stop → stop ✓ | facilities_person → facilities_person ✓ | 1127 ms | 1.000 | حوزه «تسهیلات اشخاص حقیقی» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |
| 3 | آیا میزان درآمد یا گردش حساب من هم در بهبود رتبه ا | stop → stop ✓ | facilities_person → facilities_person ✓ | 9376 ms | 1.000 | در نسخه فعلی مدل امتیاز اعتبارسنجی تسهیلات، میزان درآمد یا گردش حساب شما هیچ تاث | 5 chunks: سوال: من درآمدم بالا هستش (n تومنه) چرا رتبه‌م B2 (یا هر رتب |
| 4 | چک | stop → stop ✓ | cheque → cheque ✓ | 821 ms | 0.000 | حوزه «گزارش اعتباری چک» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |

### conv_facilities_person_02 — greeting -> scattered -> facilities_person granular [facilities_person]

elapsed 5818 ms · turns 3

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | سلام خوبی؟ | greeting → stop ✓ | — → ∅ — | 1067 ms | 0.000 | سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتبا | 0 |
| 2 | بدهکار شدم چیکار کنم | clarify → clarify ✓ | — → ∅ — | 2862 ms | 0.000 | برای راهنمایی دقیق‌تر، ممکن است بفرمایید منظور شما بدهکاری در زمینه تسهیلات (شخص | 5 chunks: سوال: با سلام من حدود 23سال پیش با حساب خودم برای خواهرم 200 |
| 3 | اعتبارسنجی تسهیلات برای اشخاص حقیقی | stop → stop ✓ | facilities_person → facilities_person ✓ | 847 ms | 1.000 | حوزه «تسهیلات اشخاص حقیقی» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |

### conv_facilities_legal_01 — facilities_legal granular -> board-role re-spec to general [facilities_legal]

elapsed 17854 ms · turns 2

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | محدودیت‌های تسهیلات برای اشخاص حقوقی چیست؟ | stop → stop ✓ | facilities_legal → ∅ ✗ | 7402 ms | 0.750 | سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌های گزارش  | 5 chunks: سوابق تسهیلاتی، چک و مالیاتی شرکت‌ها تا ۵ سال در پایگاه داده |
| 2 | هیات مدیره چه نقشی دارد؟ | stop → stop ✓ | general → general ✓ | 9767 ms | 0.700 | هیئت مدیره شرکت اعتبارسنجی ایران متشکل از متخصصانی در حوزه‌های مدیریت مالی، ریاض | 5 chunks: Key: محسن صادقی / Value: نائب رئیس هیئت مدیره / محسن صادقی،  |

### conv_facilities_legal_02 — greeting -> facilities_legal short spec [facilities_legal]

elapsed 2609 ms · turns 2

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | سلام | greeting → stop ✓ | — → ∅ — | 824 ms | 0.000 | سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتبا | 0 |
| 2 | تسهیلات اشخاص حقوقی | stop → stop ✓ | facilities_legal → facilities_legal ✓ | 1087 ms | 1.000 | حوزه «تسهیلات اشخاص حقوقی» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |

### conv_general_01 — general entity -> general intro -> cheque re-spec [general]

elapsed 15447 ms · turns 3

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | قاسم بور کیه | stop → stop ✓ | general → general ✓ | 8328 ms | 0.750 | سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌هایی نظیر  | 5 chunks: Key: رضا قاسم‌پور / Value: مدیرعامل و عضو هیئت مدیره / رضا ق |
| 2 | معرفی شرکت اعتبارسنجی ایران | stop → stop ✓ | general → general ✓ | 1127 ms | 0.000 | حوزه «پایگاه دانش عمومی» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |
| 3 | چک برگشتی چیست | stop → stop ✓ | cheque → general ✗ | 4956 ms | 1.000 | بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد. | 5 chunks: سوال: در صورتی که وجه پرداختی به حساب «اعتباریتو» واریز شده  |

### conv_general_02 — clarify -> general rejection with manager query [general]

elapsed 8526 ms · turns 2

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | وضعیت اعتباری چطور است | clarify → clarify ✓ | — → ∅ — | 2366 ms | 0.000 | برای راهنمایی دقیق‌تر، ممکن است بفرمایید منظور شما بررسی وضعیت اعتباری برای اشخا | 5 chunks: کد دلیل: OTN1 متن دلیل: فرد در ماه‌هایی که وضعیت منفی داشته  |
| 2 | هیچکدوم موارد بالا نیست، مدیر عامل شرکت کیه؟ | stop → stop ✓ | general → ∅ ✗ | 5480 ms | 1.000 | مدیرعامل فعلی شرکت اعتبارسنجی ایران دکتر رضا قاسم‌پور هستند. ایشان دارای مدرک دک | 5 chunks: دکتر رضا قاسم‌پور در حال حاضر مدیرعامل شرکت اعتبارسنجی ایران |

### conv_mixed_01 — cheque -> facilities_person re-spec [mixed]

elapsed 2585 ms · turns 2

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | چک | stop → stop ✓ | cheque → cheque ✓ | 813 ms | 0.000 | حوزه «گزارش اعتباری چک» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |
| 2 | حالا درباره تسهیلات اشخاص حقیقی بگو | stop → stop ✓ | facilities_person → facilities_person ✓ | 1092 ms | 1.000 | حوزه «تسهیلات اشخاص حقیقی» انتخاب شد. حالا سوالتان را در همین حوزه بفرمایید. | 0 |

### conv_greeting_01 — greeting x2 -> scattered clarify [greeting]

elapsed 5525 ms · turns 3

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | سلام | greeting → stop ✓ | — → ∅ — | 817 ms | 0.000 | سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتبا | 0 |
| 2 | سلام علیکم | greeting → stop ✓ | — → ∅ — | 1116 ms | 0.000 | سلام! من دستیار هوشمند شرکت اعتبارسنجی ایران هستم. سوالتان را درباره گزارش اعتبا | 0 |
| 3 | اعتبارسنجی چیست | clarify → clarify ✓ | — → ∅ — | 2575 ms | 0.000 | برای راهنمایی دقیق‌تر، آیا منظور شما آشنایی کلی با اعتبارسنجی است یا مایلید دربا | 5 chunks: سوال: آیا سابقه سفته در سامانه اعتبارسنجی ثبت می‌شود و بر رت |

### conv_hallucination_01 — hallucination guard: fake G5 grade must not be hallucinated [general]

elapsed 7745 ms · turns 1

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | نمره اعتباری من G5 است چه امتیازی میگیرم | stop → stop ✓ | — → ∅ — | 7403 ms | 0.000 | سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌های گزارش  | 5 chunks: سوال: امتیاز چک از چند تا چنده؟ پاسخ کوتاه: امتیاز چک عددی ب |

### conv_supplement_01 — supplemental entity list: shareholders [general]

elapsed 5915 ms · turns 1

| # | Query | Expected → Actual (finish) | Expected → Actual (subspace) | Latency | Faith | Snippet | Retrieved |
|---|---|---|---|---|---|---|---|
| 1 | سهامداران شرکت اعتبارسنجی ایران چه کسانی هستند | stop → stop ✓ | general → facilities_legal ✗ | 5558 ms | 1.000 | سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌هایی نظیر  | 5 chunks: سوال: چگونه می‌توانم از صحت اطلاعات مربوط به تاریخچه پرداخت‌ |

### Single-Turn Baselines

| ID | Query | Expected → Actual | Latency | Faith | Retrieved |
|---|---|---|---|---|---|
| single_cheque_granular | چک برگشتی چیست و چگونه رفع اثر می‌شود؟ | stop → stop | 10301 ms | 0.800 | 10 |
| single_facilities_person_granular | امتیاز اعتباری اشخاص حقیقی چگونه محاسبه می‌شود؟ | stop → stop | 10718 ms | 0.800 | 10 |
| single_facilities_legal_granular | الزامات وثایق برای اشخاص حقوقی چیست؟ | stop → stop | 10196 ms | 1.000 | 10 |
| single_general_granular | تاریخچه شرکت اعتبارسنجی ایران را بگو | stop → stop | 10063 ms | 0.700 | 10 |
| single_cheque_bounce_effect | تاثیر چک برگشتی بر امتیاز اعتباری چقدر است؟ | stop → stop | 10979 ms | 0.600 | 10 |
| single_facilities_person_income | آیا میزان درآمد یا گردش حساب در رتبه اعتباری تاثیر | stop → stop | 9421 ms | 1.000 | 10 |
| single_facilities_legal_board | هیات مدیره شرکت اعتبارسنجی ایران چه کسانی هستند؟ | stop → stop | 8766 ms | 0.600 | 10 |
| single_general_shareholders | سهامداران شرکت اعتبارسنجی ایران چه کسانی هستند؟ | stop → stop | 4872 ms | 1.000 | 10 |
| single_greeting | سلام | greeting → stop | 725 ms | 0.000 | 0 |
| single_scattered | وضعیت اعتباری چطور است | clarify → clarify | 2141 ms | 0.000 | 10 |
| single_hallucination_probe | نمره اعتباری من G5 است چه امتیازی میگیرم | stop → stop | 6226 ms | 0.000 | 10 |
| single_qasem_bour | قاسم بور کیه | stop → stop | 6981 ms | 0.800 | 10 |

## 4. Hallucination Details — G5 Guard

> G5 is a non-existent grade. Correct response must contain `پاسخی یافت نشد` or `وجود ندارد / یافت نشد / نامعتبر` and must NOT invent a G5 tier/score. Hallucination 1.0 = hallucinated (bad).

- **conv_hallucination_01 T1:** query `نمره اعتباری من G5 است چه امتیازی میگیرم` · hallucination 0.000 · refusal `پاسخی یافت نشد` ✗ · response: `سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌های گزارش اعتباری، امتیاز و رتبه اعتباری، روش‌های بهبود امتیاز، سوابق منفی و چک‌ها به شما `
- **single_hallucination_probe:** query `نمره اعتباری من G5 است چه امتیازی میگیرم` · hallucination 0.000 · refusal ✗ · response: `سلام، من دستیار هوشمند شرکت اعتبارسنجی ایران هستم و می‌توانم در زمینه‌های گزارش اعتباری، امتیاز و رتبه اعتباری، روش‌های بهبود امتیاز، سوابق منفی و چک‌ها به شما `

## 5. Ambiguity Resolution Scatter

| Conv / Turn | Query | Expected | Actual | Score | Result |
|---|---|---|---|---|---|
| conv_cheque_02 T1 | وضعیت اعتباری چطور است | clarify | clarify | 1.000 | ✓ clarified |
| conv_facilities_person_01 T1 | وضعیت اعتباری چطور است | clarify | clarify | 1.000 | ✓ clarified |
| conv_facilities_person_02 T2 | بدهکار شدم چیکار کنم | clarify | clarify | 1.000 | ✓ clarified |
| conv_general_02 T1 | وضعیت اعتباری چطور است | clarify | clarify | 1.000 | ✓ clarified |
| conv_greeting_01 T3 | اعتبارسنجی چیست | clarify | clarify | 1.000 | ✓ clarified |
| single_scattered | وضعیت اعتباری چطور است | clarify | clarify | 1.000 | ✓ clarified |

## 6. Sticky Persistence Timeline

### conv_cheque_01 — greeting -> cheque granular -> cheque sticky follow-up

- **applied_subspace:** T1:∅ → T2:∅ → T3:∅
- **expected_subspace:** T1:— → T2:cheque → T3:cheque
  - T1 `سلام` sticky 1.000 
  - T2 `چک برگشتی چیست و چگونه رفع سوء اثر می‌شو` sticky 0.000 ✗
  - T3 `اگر چکم بدون موجودی باشد چه کار کنم؟` sticky 0.000 ✗

### conv_cheque_02 — scattered clarify -> cheque spec -> cheque sticky granular

- **applied_subspace:** T1:∅ → T2:cheque → T3:cheque
- **expected_subspace:** T1:— → T2:cheque → T3:cheque
  - T1 `وضعیت اعتباری چطور است` sticky 1.000 
  - T2 `چک` sticky 1.000 ✓
  - T3 `تاثیر چک برگشتی بر امتیاز اعتباری چقدر ا` sticky 1.000 ✓

### conv_facilities_person_01 — clarify -> facilities_person -> sticky -> cheque re-spec

- **applied_subspace:** T1:∅ → T2:facilities_person → T3:facilities_person → T4:cheque
- **expected_subspace:** T1:— → T2:facilities_person → T3:facilities_person → T4:cheque
  - T1 `وضعیت اعتباری چطور است` sticky 1.000 
  - T2 `تسهیلات اشخاص حقیقی` sticky 1.000 ✓
  - T3 `آیا میزان درآمد یا گردش حساب من هم در به` sticky 1.000 ✓
  - T4 `چک` sticky 1.000 ✓

### conv_facilities_person_02 — greeting -> scattered -> facilities_person granular

- **applied_subspace:** T1:∅ → T2:∅ → T3:facilities_person
- **expected_subspace:** T1:— → T2:— → T3:facilities_person
  - T1 `سلام خوبی؟` sticky 1.000 
  - T2 `بدهکار شدم چیکار کنم` sticky 1.000 
  - T3 `اعتبارسنجی تسهیلات برای اشخاص حقیقی` sticky 1.000 ✓

### conv_facilities_legal_01 — facilities_legal granular -> board-role re-spec to general

- **applied_subspace:** T1:∅ → T2:general
- **expected_subspace:** T1:facilities_legal → T2:general
  - T1 `محدودیت‌های تسهیلات برای اشخاص حقوقی چیس` sticky 0.000 ✗
  - T2 `هیات مدیره چه نقشی دارد؟` sticky 1.000 ✓

### conv_facilities_legal_02 — greeting -> facilities_legal short spec

- **applied_subspace:** T1:∅ → T2:facilities_legal
- **expected_subspace:** T1:— → T2:facilities_legal
  - T1 `سلام` sticky 1.000 
  - T2 `تسهیلات اشخاص حقوقی` sticky 1.000 ✓

### conv_general_01 — general entity -> general intro -> cheque re-spec

- **applied_subspace:** T1:general → T2:general → T3:general
- **expected_subspace:** T1:general → T2:general → T3:cheque
  - T1 `قاسم بور کیه` sticky 1.000 ✓
  - T2 `معرفی شرکت اعتبارسنجی ایران` sticky 1.000 ✓
  - T3 `چک برگشتی چیست` sticky 0.000 ✗

### conv_general_02 — clarify -> general rejection with manager query

- **applied_subspace:** T1:∅ → T2:∅
- **expected_subspace:** T1:— → T2:general
  - T1 `وضعیت اعتباری چطور است` sticky 1.000 
  - T2 `هیچکدوم موارد بالا نیست، مدیر عامل شرکت ` sticky 0.000 ✗

### conv_mixed_01 — cheque -> facilities_person re-spec

- **applied_subspace:** T1:cheque → T2:facilities_person
- **expected_subspace:** T1:cheque → T2:facilities_person
  - T1 `چک` sticky 1.000 ✓
  - T2 `حالا درباره تسهیلات اشخاص حقیقی بگو` sticky 1.000 ✓

### conv_greeting_01 — greeting x2 -> scattered clarify

- **applied_subspace:** T1:∅ → T2:∅ → T3:∅
- **expected_subspace:** T1:— → T2:— → T3:—
  - T1 `سلام` sticky 1.000 
  - T2 `سلام علیکم` sticky 1.000 
  - T3 `اعتبارسنجی چیست` sticky 1.000 

## 7. Direct Retrieval Probe (with vs without filter)

12 probes (top1 with vs without filter_path)

| ID | Subspace | Query | no_filter score | has_results | with_filter score | has_results | filter improves |
|---|---|---|---|---|---|---|---|
| single_cheque_granular | گزارش اعتباری چک | چک برگشتی چیست و چگونه رفع اثر می‌شود؟ | 0.961 | ✓ | — | ✗ | None |
| single_facilities_person_granular | اشخاص حقیقی/اعتبارسنجی تسهیلات | امتیاز اعتباری اشخاص حقیقی چگونه محاسبه  | 0.939 | ✓ | 0.939 | ✓ | True |
| single_facilities_legal_granular | اشخاص حقوقی/اعتبارسنجی تسهیلات | الزامات وثایق برای اشخاص حقوقی چیست؟ | 0.500 | ✓ | 0.012 | ✓ | False |
| single_general_granular | پایگاه دانش عمومی | تاریخچه شرکت اعتبارسنجی ایران را بگو | 0.858 | ✓ | 0.992 | ✓ | True |
| single_cheque_bounce_effect | گزارش اعتباری چک | تاثیر چک برگشتی بر امتیاز اعتباری چقدر ا | 0.997 | ✓ | — | ✗ | None |
| single_facilities_person_income | اشخاص حقیقی/اعتبارسنجی تسهیلات | آیا میزان درآمد یا گردش حساب در رتبه اعت | 0.998 | ✓ | 1.000 | ✓ | True |
| single_facilities_legal_board | پایگاه دانش عمومی | هیات مدیره شرکت اعتبارسنجی ایران چه کسان | 0.994 | ✓ | 0.994 | ✓ | True |
| single_general_shareholders | پایگاه دانش عمومی | سهامداران شرکت اعتبارسنجی ایران چه کسانی | 0.999 | ✓ | 0.984 | ✓ | False |
| single_greeting | ∅ | سلام | 0.956 | ✓ | 0.956 | ✓ | True |
| single_scattered | ∅ | وضعیت اعتباری چطور است | 0.739 | ✓ | 0.739 | ✓ | True |
| single_hallucination_probe | ∅ | نمره اعتباری من G5 است چه امتیازی میگیرم | 0.500 | ✓ | 0.500 | ✓ | True |
| single_qasem_bour | پایگاه دانش عمومی | قاسم بور کیه | 0.500 | ✓ | 0.500 | ✓ | True |

## 8. Retry / Fallback Observability (T7)

Total turns 41 — fallback_used 0 (0.0% if fb_total_m else '0%') — broaden_used 0 (0.0% if fb_total_m else '0%')

| Subspace | turns | fallback_used | fallback rate | broaden_used | broaden rate |
|---|---|---|---|---|---|
| cheque | 12 | 0 | 0.0% | 0 | 0.0% |
| facilities_legal | 4 | 0 | 0.0% | 0 | 0.0% |
| facilities_person | 9 | 0 | 0.0% | 0 | 0.0% |
| general | 12 | 0 | 0.0% | 0 | 0.0% |
| greeting | 4 | 0 | 0.0% | 0 | 0.0% |

(No fallback or broaden retries triggered — all retrievals sufficient on first attempt.)

---
_Generated 2026-10-14 09:52 UTC · bench 2026-10-14T09:52:18.539841+00:00 · judge gemma-4-31b · error_rate 0.0%_