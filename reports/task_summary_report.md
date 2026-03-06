# Task Summary Report

**Task 1 — Data Ingestion & Initial Cleaning:**  
This step established a reproducible ingestion pipeline for utility sensor telemetry and outage logs, standardized schemas across both sources, normalized timestamp and text formats, removed duplicates, and handled missing numeric values through asset-level forward/backward filling with median fallback. Basic integrity checks were included to validate schema completeness, duplicate rates, negative outage durations, and timestamp parsing quality, ensuring the downstream workflow starts from a consistent and auditable data foundation.

**Task 2 — Sensor Data Preprocessing & Feature Engineering:**  
The raw telemetry was transformed into a model-ready feature layer by clipping outliers, smoothing noisy sensor streams, and deriving predictive indicators such as rolling means, rolling volatility, multi-hour trends, load change rates, power proxies, and frequency-domain vibration energy. These engineered features convert raw operational measurements into interpretable health signals that better reflect thermal stress, mechanical wear, electrical instability, and sudden operational changes that typically precede asset degradation.

**Task 3 — Outage Log Analysis & Label Generation:**  
Historical outage events were profiled to summarize failure type, root-cause distribution, severity patterns, and average outage durations, then aligned with time-indexed sensor data to create supervised learning labels. Clear business labeling rules were applied so that normal periods remain class 0, the 24-hour period before an outage becomes a pre-failure anomaly label, and the outage interval itself becomes a failure window label, producing a practical training target that mirrors early-warning utility maintenance workflows.

**Task 4 — ML Model Selection & Initial Training:**  
Multiple classification families were benchmarked, including Random Forest, Extra Trees, and Logistic Regression, using the processed feature set and generated labels. Each model was wrapped in a consistent preprocessing pipeline, evaluated with train/test splitting and light hyperparameter tuning, and compared on accuracy, precision, recall, weighted F1, classification reports, and confusion matrices to identify the strongest baseline candidate for predictive failure detection.

**Task 5 — Model Optimization & Validation:**  
The baseline model was refined through randomized hyperparameter search, cross-validation, and an ensemble comparison to improve generalization and reduce operational risk. Validation extended beyond headline accuracy to include weighted F1, risky-vs-normal false positive and false negative rates, cross-validation stability, and practical documentation of limitations such as synthetic-data bias, class imbalance, seasonal drift, and the simplified proxy used for time-to-failure estimation.

**Task 6 — Market Mood Index (Asset Health Regime) Definition:**  
Model outputs were translated into an interpretable 0–100 Asset Health Index by combining risk probabilities with entropy-based confidence, then smoothing the result with EWMA to reduce short-term noise. Business rules, persistence thresholds, hysteresis, and immediate escalation logic were used to assign Normal, Warning, and Critical regimes, producing a decision-oriented asset health signal that is easier for operators and maintenance planners to consume than raw class probabilities alone.
