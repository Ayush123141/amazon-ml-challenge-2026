# Amazon ML Challenge 2026 — Data Report

## 1. Purpose

This document records the initial Exploratory Data Analysis (EDA) performed on the Business Entity Resolution datasets. The analysis covers dataset structure, schema, missing values, duplicates, entity ID integrity, country distribution, text-field characteristics, and training ground-truth match counts.

## 2. Dataset Overview


| dataset            |    rows |   columns | column_names                                        |
|:-------------------|--------:|----------:|:----------------------------------------------------|
| Train Source 1     | 2206821 |         4 | entity_id, business_name, business_address, country |
| Train Source 2     | 5034616 |         4 | entity_id, business_name, business_address, country |
| Train Source 3     | 5285603 |         4 | entity_id, business_name, business_address, country |
| Train Ground Truth | 2206821 |         2 | source1_entity_id, matched_entity_ids               |
| Test Source 1      | 1732544 |         4 | entity_id, business_name, business_address, country |
| Test Source 2      | 4887273 |         4 | entity_id, business_name, business_address, country |
| Test Source 3      | 5082316 |         4 | entity_id, business_name, business_address, country |


## 3. Schema and Data Types


| dataset            | column             | dtype   |   non_empty_values |   unique_values |
|:-------------------|:-------------------|:--------|-------------------:|----------------:|
| Train Source 1     | entity_id          | str     |            2206821 |         2206821 |
| Train Source 1     | business_name      | str     |            2206821 |         1539229 |
| Train Source 1     | business_address   | str     |            2206821 |         2130606 |
| Train Source 1     | country            | str     |            2206821 |               2 |
| Train Source 2     | entity_id          | str     |            5034616 |         5034616 |
| Train Source 2     | business_name      | str     |            5034616 |         4402009 |
| Train Source 2     | business_address   | str     |            4865649 |         4337262 |
| Train Source 2     | country            | str     |            5034616 |               2 |
| Train Source 3     | entity_id          | str     |            5285603 |         5285603 |
| Train Source 3     | business_name      | str     |            5285603 |         4651609 |
| Train Source 3     | business_address   | str     |            5109687 |         4632765 |
| Train Source 3     | country            | str     |            5285603 |               2 |
| Train Ground Truth | source1_entity_id  | str     |            2206821 |         2206821 |
| Train Ground Truth | matched_entity_ids | str     |            2083574 |         2083575 |
| Test Source 1      | entity_id          | str     |            1732544 |         1732544 |
| Test Source 1      | business_name      | str     |            1732544 |         1238867 |
| Test Source 1      | business_address   | str     |            1732544 |         1677483 |
| Test Source 1      | country            | str     |            1732544 |               3 |
| Test Source 2      | entity_id          | str     |            4887273 |         4887273 |
| Test Source 2      | business_name      | str     |            4887273 |         4311041 |
| Test Source 2      | business_address   | str     |            4757865 |         4224784 |
| Test Source 2      | country            | str     |            4887273 |               3 |
| Test Source 3      | entity_id          | str     |            5082316 |         5082316 |
| Test Source 3      | business_name      | str     |            5082316 |         4521929 |
| Test Source 3      | business_address   | str     |            4946218 |         4456436 |
| Test Source 3      | country            | str     |            5082316 |               3 |


## 4. Missing-Value Analysis


### Train Source 1


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                    0 |
| business_name    |              0 |             0 |               0 |                    0 |
| business_address |              0 |             0 |               0 |                    0 |
| country          |              0 |             0 |               0 |                    0 |


### Train Source 2


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                 0    |
| business_name    |              0 |             0 |               0 |                 0    |
| business_address |         168967 |             0 |          168967 |                 3.36 |
| country          |              0 |             0 |               0 |                 0    |


### Train Source 3


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                 0    |
| business_name    |              0 |             0 |               0 |                 0    |
| business_address |         175916 |             0 |          175916 |                 3.33 |
| country          |              0 |             0 |               0 |                 0    |


### Train Ground Truth


| column             |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-------------------|---------------:|--------------:|----------------:|---------------------:|
| source1_entity_id  |              0 |             0 |               0 |                 0    |
| matched_entity_ids |         123247 |             0 |          123247 |                 5.58 |


### Test Source 1


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                    0 |
| business_name    |              0 |             0 |               0 |                    0 |
| business_address |              0 |             0 |               0 |                    0 |
| country          |              0 |             0 |               0 |                    0 |


### Test Source 2


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                 0    |
| business_name    |              0 |             0 |               0 |                 0    |
| business_address |         129408 |             0 |          129408 |                 2.65 |
| country          |              0 |             0 |               0 |                 0    |


### Test Source 3


| column           |   blank_values |   null_values |   missing_total |   missing_percentage |
|:-----------------|---------------:|--------------:|----------------:|---------------------:|
| entity_id        |              0 |             0 |               0 |                 0    |
| business_name    |              0 |             0 |               0 |                 0    |
| business_address |         136098 |             0 |          136098 |                 2.68 |
| country          |              0 |             0 |               0 |                 0    |


## 5. Duplicate-Row Analysis


| dataset            |   duplicate_rows |   duplicate_percentage |
|:-------------------|-----------------:|-----------------------:|
| Train Source 1     |                0 |                      0 |
| Train Source 2     |                0 |                      0 |
| Train Source 3     |                0 |                      0 |
| Train Ground Truth |                0 |                      0 |
| Test Source 1      |                0 |                      0 |
| Test Source 2      |                0 |                      0 |
| Test Source 3      |                0 |                      0 |


## 6. Entity-ID Integrity


| dataset            | entity_id_present   |    unique_ids |   duplicate_ids |   empty_ids |
|:-------------------|:--------------------|--------------:|----------------:|------------:|
| Train Source 1     | True                |   2.20682e+06 |               0 |           0 |
| Train Source 2     | True                |   5.03462e+06 |               0 |           0 |
| Train Source 3     | True                |   5.2856e+06  |               0 |           0 |
| Train Ground Truth | False               | nan           |             nan |         nan |
| Test Source 1      | True                |   1.73254e+06 |               0 |           0 |
| Test Source 2      | True                |   4.88727e+06 |               0 |           0 |
| Test Source 3      | True                |   5.08232e+06 |               0 |           0 |


## 7. Country Distribution


### Train Source 1


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| US        | 1323633 |        59.98 |
| India     |  883188 |        40.02 |


### Train Source 2


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| US        | 3016817 |        59.92 |
| India     | 2017799 |        40.08 |


### Train Source 3


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| US        | 3170056 |        59.98 |
| India     | 2115547 |        40.02 |


### Test Source 1


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| India     |  809986 |        46.75 |
| US        |  663106 |        38.27 |
| France    |  259452 |        14.98 |


### Test Source 2


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| India     | 2312565 |        47.32 |
| US        | 1871330 |        38.29 |
| France    |  703378 |        14.39 |


### Test Source 3


| country   |   count |   percentage |
|:----------|--------:|-------------:|
| India     | 2405000 |        47.32 |
| US        | 1945701 |        38.28 |
| France    |  731615 |        14.4  |


## 8. Business Name and Address Length Statistics


| dataset        | field            |   minimum |   maximum |   mean |   median |   p25 |   p75 |
|:---------------|:-----------------|----------:|----------:|-------:|---------:|------:|------:|
| Train Source 1 | business_name    |         3 |       105 |  24.03 |       24 |    18 |    30 |
| Train Source 1 | business_address |        11 |       256 |  52.07 |       41 |    33 |    70 |
| Train Source 2 | business_name    |         2 |       104 |  25.1  |       25 |    19 |    31 |
| Train Source 2 | business_address |         0 |       249 |  46.23 |       37 |    30 |    61 |
| Train Source 3 | business_name    |         2 |       123 |  25.2  |       25 |    18 |    31 |
| Train Source 3 | business_address |         0 |       240 |  46.71 |       42 |    35 |    54 |
| Test Source 1  | business_name    |         3 |        92 |  23.84 |       24 |    18 |    29 |
| Test Source 1  | business_address |        11 |       268 |  57.21 |       50 |    36 |    74 |
| Test Source 2  | business_name    |         2 |       102 |  25.7  |       25 |    19 |    32 |
| Test Source 2  | business_address |         0 |       269 |  50.41 |       43 |    32 |    67 |
| Test Source 3  | business_name    |         2 |       103 |  25.66 |       25 |    19 |    32 |
| Test Source 3  | business_address |         0 |       267 |  48.74 |       43 |    35 |    59 |


## 9. Training Ground-Truth Match Distribution


- Ground-truth rows: **2206821**

- Entities with zero matches: **123247**

- Entities with exactly one match: **119157**

- Entities with multiple matches: **1964417**

- Maximum matches for one Source 1 entity: **11**



|   match_count |   source1_entities |   percentage |
|--------------:|-------------------:|-------------:|
|             0 |             123247 |         5.58 |
|             1 |             119157 |         5.4  |
|             2 |             375212 |        17    |
|             3 |             530841 |        24.05 |
|             4 |             484115 |        21.94 |
|             5 |             321957 |        14.59 |
|             6 |             164868 |         7.47 |
|             7 |              63968 |         2.9  |
|             8 |              18680 |         0.85 |
|             9 |               4205 |         0.19 |
|            10 |                534 |         0.02 |
|            11 |                 37 |         0    |


## 10. Source 1 / Ground-Truth Coverage


- Train Source 1 rows: **2206821**

- Ground-truth rows: **2206821**

- Unique Train Source 1 IDs: **2206821**

- Unique ground-truth Source 1 IDs: **2206821**

- Source 1 IDs missing from ground truth: **0**

- Ground-truth IDs not present in Source 1: **0**


## 11. Test Source 1 Coverage


- Test Source 1 rows: **1732544**

- Unique Test Source 1 entity IDs: **1732544**

- Every Test Source 1 entity must appear in the final matching output, including entities with no matches.


## 12. Initial EDA Observations


The observations below are generated directly from the loaded datasets. No normalization, blocking, fuzzy matching, or ML scoring has been applied at this stage.

- Train Source 1 contains **2** distinct country values.

- Train Source 1 business-name length: minimum **3**, maximum **105**, median **24.0** characters.

- Train Source 1 business-address length: minimum **11**, maximum **256**, median **41.0** characters.

- Ground truth contains **123247** entities with zero matches, **119157** with exactly one match, and **1964417** with multiple matches.


## 13. Next Step

The next M1 stage should use this EDA to design the normalization pipeline. This should include business-name normalization, address normalization, punctuation and whitespace handling, abbreviation handling, country-aware processing, and preservation of information that may be useful for later blocking and matching.
