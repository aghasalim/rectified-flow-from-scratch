-- Recompute the two published tables from the per seed rows.
--
-- Both README tables are medians over three seeds, taken by
-- scripts/check_numbers.py in Python. That is one implementation, and the
-- figures read the same CSV, so a wrong group-by would agree with itself
-- everywhere. This derives the same cells in SQL from results/nfe-quality.csv
-- and results/straightness.csv. verify/verify.sh diffs this output against the
-- Go and JavaScript recomputations.
--
-- Run: sqlite3 -init verify/medians.sql :memory: "" < /dev/null

.mode csv
.headers off
.import --csv results/nfe-quality.csv nfe
.import --csv results/straightness.csv straight

-- Median of an odd or even count without a median aggregate: average the one
-- or two middle order statistics.
CREATE TEMP VIEW w2 AS
    SELECT dataset, model, CAST(nfe AS INTEGER) AS nfe, CAST(sliced_w2 AS REAL) AS v
    FROM nfe WHERE sampler = 'euler';

CREATE TEMP VIEW w2_ranked AS
    SELECT dataset, model, nfe, v,
           ROW_NUMBER() OVER (PARTITION BY dataset, model, nfe ORDER BY v) AS rn,
           COUNT(*)     OVER (PARTITION BY dataset, model, nfe)            AS c
    FROM w2;

CREATE TEMP VIEW s_ranked AS
    SELECT dataset, model,
           CAST(straightness_S AS REAL) AS s,
           CAST(path_length_ratio_mean AS REAL) AS r,
           ROW_NUMBER() OVER (PARTITION BY dataset, model
                              ORDER BY CAST(straightness_S AS REAL)) AS rn_s,
           ROW_NUMBER() OVER (PARTITION BY dataset, model
                              ORDER BY CAST(path_length_ratio_mean AS REAL)) AS rn_r,
           COUNT(*)     OVER (PARTITION BY dataset, model) AS c
    FROM straight;

SELECT 'w2' AS kind, dataset, model, printf('%03d', nfe) AS key,
       printf('%.10f', AVG(v)) AS value
FROM w2_ranked
WHERE rn IN ((c + 1) / 2, (c + 2) / 2) AND nfe IN (1, 2, 4, 8, 128)
GROUP BY dataset, model, nfe
UNION ALL
SELECT 'straightness_S', dataset, model, 'median', printf('%.10f', AVG(s))
FROM s_ranked WHERE rn_s IN ((c + 1) / 2, (c + 2) / 2)
GROUP BY dataset, model
UNION ALL
SELECT 'path_length_ratio_mean', dataset, model, 'median', printf('%.10f', AVG(r))
FROM s_ranked WHERE rn_r IN ((c + 1) / 2, (c + 2) / 2)
GROUP BY dataset, model;
