# M3.1 Feature Report

- Candidate pairs: 1,799,071
- Source-1 entities represented: 85,940
- Positives: 172,163
- Negatives: 1,626,908
- Positive rate: 9.569550%
- Source-1 limit: all candidate rows
- Duplicate pair rows: 0
- Missing feature values: 0

## Mean feature values by label

```csv
,0,1
label,0.0,1.0
country_exact,1.0,1.0
country_normalized_exact,1.0,1.0
name_token_count_s1,3.078628,3.512276
name_token_count_target,3.118905,3.474283
address_token_count_s1,7.881072,8.281303
address_token_count_target,7.57896,7.830666
name_length_s1,21.00076,23.45551
name_length_target,21.450443,23.372647
address_length_s1,47.174596,50.197023
address_length_target,44.442897,46.203575
retrieved_by_exact_name,0.554811,0.43859
retrieved_by_exact_address,0.003826,0.164925
retrieved_by_exact_name_address,0.0,0.025389
retrieved_by_rare_token,1.0,1.0
matching_rare_token_count,2.820949,9.163578
name_exact,0.554811,0.43859
name_token_jaccard,0.603551,0.725178
name_token_overlap_count,1.740983,2.765734
name_token_containment,0.638752,0.833175
name_char_similarity,0.71665,0.84445
name_edit_similarity,0.71665,0.84445
name_length_difference_normalized,0.130253,0.105702
name_length_difference_raw,4.022826,2.963842
name_token_count_difference,0.503931,0.447187
name_prefix_similarity,0.58069,0.649691
name_suffix_similarity,0.574974,0.51839
address_exact,0.003826,0.164925
address_token_jaccard,0.073653,0.667345
address_token_overlap_count,1.043547,6.458879
address_token_containment,0.126066,0.805837
address_char_similarity,0.332498,0.796377
address_edit_similarity,0.332498,0.796377
address_length_difference_normalized,0.22828,0.143385
address_length_difference_raw,13.166352,7.735832
address_token_count_difference,2.113994,1.081156
address_numeric_token_overlap,0.126801,1.217271
address_digit_agreement,0.0696,0.767648
```
