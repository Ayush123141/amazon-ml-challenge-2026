# M3.1 Feature Report

- Candidate pairs: 3,956,525
- Source-1 entities represented: 95,876
- Positives: 227,853
- Negatives: 3,728,672
- Positive rate: 5.758917%
- Source-1 limit: all candidate rows
- Duplicate pair rows: 0
- Missing feature values: 0

## Mean feature values by label

```csv
,0,1
label,0.0,1.0
country_exact,1.0,1.0
country_normalized_exact,1.0,1.0
name_token_count_s1,3.191111,3.533133
name_token_count_target,3.19459,3.472599
address_token_count_s1,7.832935,8.198878
address_token_count_target,7.352319,7.610301
name_length_s1,21.912794,23.609375
name_length_target,21.836836,23.377041
address_length_s1,46.554636,49.281045
address_length_target,42.646017,44.496109
retrieved_by_exact_name,0.242077,0.331393
retrieved_by_exact_address,0.001669,0.124615
retrieved_by_exact_name_address,0.0,0.019183
retrieved_by_rare_token,1.0,1.0
matching_rare_token_count,2.811914,8.95846
name_exact,0.242077,0.331393
name_token_jaccard,0.627214,0.726849
name_token_overlap_count,2.184735,2.811782
name_token_containment,0.781854,0.85834
name_char_similarity,0.786033,0.838893
name_edit_similarity,0.786033,0.838893
name_length_difference_normalized,0.161508,0.123309
name_length_difference_raw,4.39819,3.381909
name_token_count_difference,0.630698,0.519901
name_prefix_similarity,0.622008,0.635626
name_suffix_similarity,0.275708,0.402649
address_exact,0.001669,0.124615
address_token_jaccard,0.043043,0.646762
address_token_overlap_count,0.6267,6.212584
address_token_containment,0.076946,0.790775
address_char_similarity,0.295163,0.775206
address_edit_similarity,0.295163,0.775206
address_length_difference_normalized,0.2453,0.158691
address_length_difference_raw,14.137156,8.549003
address_token_count_difference,2.255778,1.208494
address_numeric_token_overlap,0.064695,1.194647
address_digit_agreement,0.034758,0.748181
```
