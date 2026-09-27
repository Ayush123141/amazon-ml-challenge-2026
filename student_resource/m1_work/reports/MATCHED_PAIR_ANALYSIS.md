# M1 — Matched-Pair Variation Analysis

## 1. Objective

This analysis examines positive Source 1 → Source 2/Source 3 entity-resolution pairs from the training ground truth. The purpose is to identify observed variations in business names and addresses before implementing the normalization pipeline.

**Important:** No normalization rules are applied to the source datasets in this analysis. The transformations measured below are diagnostic comparisons only.


## 2. Dataset Summary


| dataset                |    rows |
|:-----------------------|--------:|
| Train Source 1         | 2206821 |
| Train Source 2         | 5034616 |
| Train Source 3         | 5285603 |
| Positive matched pairs | 7638365 |


## 3. Ground-Truth Reference Validation


- Positive ground-truth pairs analyzed: **7,638,365**

- Invalid Source 1 references encountered: **0**

- Invalid matched references encountered: **0**


## 4. Matched Pair Distribution by Source


| matched_source   |   positive_pairs |   name_exact |   address_exact |   country_exact |   name_token_order_equal |   address_token_order_equal |   name_exact_pct |   address_exact_pct |   country_exact_pct |
|:-----------------|-----------------:|-------------:|----------------:|----------------:|-------------------------:|----------------------------:|-----------------:|--------------------:|--------------------:|
| S2               |          3693619 |       174617 |               5 |         3693619 |                   983375 |                      645779 |             4.73 |                0    |                 100 |
| S3               |          3944746 |       179501 |          170069 |         3944746 |                  1072212 |                      243542 |             4.55 |                4.31 |                 100 |


## 5. Matched Pair Distribution by Source 1 Country


| s1_country   |   positive_pairs |   name_exact |   address_exact |   country_exact |   name_token_order_equal |   address_token_order_equal |   name_exact_pct |   address_exact_pct |   country_exact_pct |
|:-------------|-----------------:|-------------:|----------------:|----------------:|-------------------------:|----------------------------:|-----------------:|--------------------:|--------------------:|
| India        |          3059843 |        82949 |           65377 |         3059843 |                   624967 |                      318283 |             2.71 |                2.14 |                 100 |
| US           |          4578522 |       271169 |          104697 |         4578522 |                  1430620 |                      571038 |             5.92 |                2.29 |                 100 |


## 6. Exact Equality Analysis


| field            |   exact_pairs |   percentage |
|:-----------------|--------------:|-------------:|
| Business name    |        354118 |         4.64 |
| Business address |        170074 |         2.23 |
| Country          |       7638365 |       100    |


## 7. Business Name Transformation Coverage


| transformation         |   pairs_equal |   percentage |
|:-----------------------|--------------:|-------------:|
| Exact                  |        354118 |         4.64 |
| Casefold               |        821025 |        10.75 |
| Whitespace normalized  |       1204650 |        15.77 |
| Punctuation normalized |       1843780 |        24.14 |
| Alphanumeric only      |       1752581 |        22.94 |
| Token order normalized |       2055587 |        26.91 |
| Digits equal           |       7322746 |        95.87 |


## 8. Business Address Transformation Coverage


| transformation         |   pairs_equal |   percentage |
|:-----------------------|--------------:|-------------:|
| Exact                  |        170074 |         2.23 |
| Casefold               |        553282 |         7.24 |
| Whitespace normalized  |        566260 |         7.41 |
| Punctuation normalized |        633281 |         8.29 |
| Alphanumeric only      |        635952 |         8.33 |
| Token order normalized |        889321 |        11.64 |
| Digits equal           |       4739981 |        62.05 |


## 9. Business Name Variation Categories


| variation_category   |   pairs |   percentage |
|:---------------------|--------:|-------------:|
| Other                | 5202443 |        68.11 |
| Punctuation          |  639130 |         8.37 |
| Case only            |  466907 |         6.11 |
| Token order          |  386779 |         5.06 |
| Whitespace           |  383625 |         5.02 |
| Exact                |  354118 |         4.64 |
| Formatting / symbols |  205363 |         2.69 |


## 10. Business Address Variation Categories


| variation_category   |   pairs |   percentage |
|:---------------------|--------:|-------------:|
| Other                | 6744501 |        88.3  |
| Case only            |  383208 |         5.02 |
| Token order          |  257902 |         3.38 |
| Exact                |  170074 |         2.23 |
| Punctuation          |   67021 |         0.88 |
| Whitespace           |   12978 |         0.17 |
| Formatting / symbols |    2681 |         0.04 |


## 11. Business Name Difference Types


| difference_type      |   pairs |   percentage |
|:---------------------|--------:|-------------:|
| Content variation    | 5589222 |        73.17 |
| Punctuation          |  639130 |         8.37 |
| Case                 |  466907 |         6.11 |
| Whitespace           |  383625 |         5.02 |
| Exact                |  354118 |         4.64 |
| Symbols / formatting |  205363 |         2.69 |


## 12. Business Address Difference Types


| difference_type      |   pairs |   percentage |
|:---------------------|--------:|-------------:|
| Content variation    | 7002403 |        91.67 |
| Case                 |  383208 |         5.02 |
| Exact                |  170074 |         2.23 |
| Punctuation          |   67021 |         0.88 |
| Whitespace           |   12978 |         0.17 |
| Symbols / formatting |    2681 |         0.04 |


## 13. Similarity Distribution


Similarity statistics below are calculated on a deterministic sample of **100,000 positive pairs** rather than the complete positive-pair population. This avoids the extreme computational cost of running Python-level string similarity over millions of rows.


| metric                      |   mean |   median |    p10 |    p25 |    p75 |    p90 |   minimum |
|:----------------------------|-------:|---------:|-------:|-------:|-------:|-------:|----------:|
| Name sequence similarity    | 0.7815 |   0.8649 | 0.4582 | 0.7143 | 0.9474 | 1      |    0.0377 |
| Address sequence similarity | 0.7492 |   0.8254 | 0.4407 | 0.6422 | 0.918  | 0.9752 |    0      |
| Name character Jaccard      | 0.8031 |   0.8889 | 0.4667 | 0.75   | 1      | 1      |    0.0222 |
| Address character Jaccard   | 0.8313 |   0.9    | 0.625  | 0.7727 | 0.963  | 1      |    0      |
| Name token Jaccard          | 0.6147 |   0.6667 | 0      | 0.5    | 1      | 1      |    0      |
| Address token Jaccard       | 0.5975 |   0.625  | 0.25   | 0.4286 | 0.7857 | 1      |    0      |


## 14. Empty Address Analysis


| condition             |   count |   percentage |
|:----------------------|--------:|-------------:|
| S1 address empty      |       0 |         0    |
| Matched address empty |  337018 |         4.41 |
| Both addresses empty  |       0 |         0    |
| Either address empty  |  337018 |         4.41 |


## 15. Lowest-Similarity Positive Name Pairs


These are genuine ground-truth matches with comparatively low character-level name similarity within the analyzed similarity sample.


| source1_entity_id   | matched_entity_id   | matched_source   | s1_country   | s1_business_name                          | matched_business_name   |   name_sequence_similarity |   name_token_jaccard |
|:--------------------|:--------------------|:-----------------|:-------------|:------------------------------------------|:------------------------|---------------------------:|---------------------:|
| S1-317238493        | S3-266932261        | S3               | India        | International Infrastructure              | इंटरनेशनल इंफ्रास्ट्रक्चर        |                  0.0377358 |                    0 |
| S1-317238493        | S2-455875431        | S2               | India        | International Infrastructure              | इंटरनेशनल इंफ्रास्ट्रक्चर        |                  0.0377358 |                    0 |
| S1-198788972        | S3-333904297        | S3               | India        | Kakinada (Urban) Skincare Private Limited | Beloumbra               |                  0.04      |                    0 |
| S1-380645956        | S3-266454179        | S3               | India        | Faridabad Publishers Private Limited      | Nylálumlyra             |                  0.0425532 |                    0 |
| S1-179304425        | S3-211843016        | S3               | India        | Universal Enterprises Private Limited     | Fayeonyx                |                  0.0444444 |                    0 |
| S1-211616414        | S3-968495590        | S3               | US           | Pandya, Osorio and Fluke Health, LLC      | Onyxumbra               |                  0.0444444 |                    0 |
| S1-217869217        | S2-513975759        | S2               | India        | Digital Infrastructure                    | डिजिटल इंफ्रास्ट्रक्चर         |                  0.0454545 |                    0 |
| S1-878107316        | S2-337702070        | S2               | US           | Integrated Materials Consultants Inc      | Keloorbi                |                  0.0454545 |                    0 |
| S1-201575633        | S2-268146607        | S2               | India        | United Constructions                      | யுனைடெட் கன்ஸ்ட்ரக்ஷன்ஸ்         |                  0.0454545 |                    0 |
| S1-789759210        | S2-962396946        | S2               | India        | International Industries                  | इंटरनेशनल इंडस्ट्रीज           |                  0.0454545 |                    0 |
| S1-789759210        | S2-663458560        | S2               | India        | International Industries                  | इंटरनेशनल इंडस्ट्रीज           |                  0.0454545 |                    0 |
| S1-714565486        | S3-542282760        | S3               | India        | Panchwati Solutions Private Limited       | Ectovera                |                  0.0465116 |                    0 |
| S1-497482988        | S2-865292743        | S2               | India        | Pune Servicesprivate Private Limited      | Lumzeph                 |                  0.0465116 |                    0 |
| S1-123121467        | S2-439543058        | S2               | India        | Labdhi Vani of Machilipatnam Pvt Ltd      | Korfaye                 |                  0.0465116 |                    0 |
| S1-294347693        | S2-854162346        | S2               | India        | Creative Investments                      | क्रिएटिव इन्वेस्टमेंट्स         |                  0.0465116 |                    0 |
| S1-59267573         | S2-747607074        | S2               | India        | International Consulting                  | इंटरनेशनल कंसल्टिंग           |                  0.0465116 |                    0 |
| S1-59267573         | S3-89170150         | S3               | India        | International Consulting                  | इंटरनेशनल कंसल्टिंग           |                  0.0465116 |                    0 |
| S1-771632417        | S2-146956111        | S2               | India        | International Impex                       | ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಇಂಪೆಕ್ಸ್         |                  0.047619  |                    0 |
| S1-771632417        | S2-181566078        | S2               | India        | International Impex                       | ಇಂಟರ್‌ನ್ಯಾಷನಲ್ ಇಂಪೆಕ್ಸ್         |                  0.047619  |                    0 |
| S1-476763791        | S3-431727797        | S3               | India        | Best Constructions                        | బెస్ట్ కన్‌స్ట్రక్షన్స్             |                  0.0487805 |                    0 |
| S1-760435535        | S2-570246674        | S2               | India        | Creative Healthcare                       | ಕ್ರಿಯೇಟಿವ್ ಹೆಲ್ತ್‌ಕೇರ್           |                  0.0487805 |                    0 |
| S1-760435535        | S2-776583526        | S2               | India        | Creative Healthcare                       | ಕ್ರಿಯೇಟಿವ್ ಹೆಲ್ತ್‌ಕೇರ್           |                  0.0487805 |                    0 |
| S1-760435535        | S3-959124977        | S3               | India        | Creative Healthcare                       | ಕ್ರಿಯೇಟಿವ್ ಹೆಲ್ತ್‌ಕೇರ್           |                  0.0487805 |                    0 |
| S1-885485250        | S2-144030669        | S2               | India        | Shivam Constructions                      | శివం కన్‌స్ట్రక్షన్స్            |                  0.0487805 |                    0 |
| S1-476763791        | S2-693154291        | S2               | India        | Best Constructions                        | బెస్ట్ కన్‌స్ట్రక్షన్స్             |                  0.0487805 |                    0 |
| S1-960007747        | S2-494862771        | S2               | India        | Supreme Constructions                     | सुप्रीम कंस्ट्रक्शंस             |                  0.0487805 |                    0 |
| S1-250398308        | S2-894520069        | S2               | India        | Black Investments                         | பிளாக் இன்வெஸ்ட்மெண்ட்ஸ்           |                  0.0487805 |                    0 |
| S1-476763791        | S3-158221878        | S3               | India        | Best Constructions                        | బెస్ట్ కన్‌స్ట్రక్షన్స్             |                  0.0487805 |                    0 |
| S1-207246905        | S2-47216329         | S2               | US           | Almeta Parker Creative Compass Inc        | Wexmira                 |                  0.0487805 |                    0 |
| S1-250398308        | S3-137313073        | S3               | India        | Black Investments                         | பிளாக் இன்வெஸ்ட்மெண்ட்ஸ்           |                  0.0487805 |                    0 |
| S1-103056427        | S3-867330132        | S3               | India        | Anand Investments                         | ఆనంద్ ఇన్వెస్ట్‌మెంట్స్          |                  0.05      |                    0 |
| S1-103056427        | S2-316855458        | S2               | India        | Anand Investments                         | ఆనంద్ ఇన్వెస్ట్‌మెంట్స్          |                  0.05      |                    0 |
| S1-254057674        | S2-290829449        | S2               | India        | Swastik Technologies                      | स्वस्तिक टेक्नोलॉजीज          |                  0.05      |                    0 |
| S1-11945426         | S3-958248137        | S3               | India        | Devi Services Private Limited             | Synyumagild             |                  0.05      |                    0 |
| S1-254057674        | S3-742176042        | S3               | India        | Swastik Technologies                      | स्वस्तिक टेक्नोलॉजीज          |                  0.05      |                    0 |
| S1-144239108        | S3-172928912        | S3               | India        | Mumbai Marketing Private Limited          | Onyxcira                |                  0.05      |                    0 |
| S1-665326577        | S2-413386268        | S2               | US           | 2820 Lincoln Plaza Management LLC         | NEXGILD                 |                  0.05      |                    0 |
| S1-672431182        | S2-878343619        | S2               | India        | Sunrise Investments                       | सनराइज इन्वेस्टमेंट्स         |                  0.05      |                    0 |
| S1-672431182        | S3-830716356        | S3               | India        | Sunrise Investments                       | सनराइज इन्वेस्टमेंट्स         |                  0.05      |                    0 |
| S1-429134233        | S3-901667372        | S3               | India        | Whitestone Farmers Private Limited        | Evoavi                  |                  0.05      |                    0 |
| S1-108816955        | S2-387015647        | S2               | US           | Primary Care Center of Islip Inc          | Cirazeph                |                  0.05      |                    0 |
| S1-103056427        | S3-932402112        | S3               | India        | Anand Investments                         | ఆనంద్ ఇన్వెస్ట్‌మెంట్స్          |                  0.05      |                    0 |
| S1-13148852         | S2-293373135        | S2               | India        | Gujarat Technologies                      | గుజరాత్ టెక్నాలజీస్            |                  0.0512821 |                    0 |
| S1-13148852         | S3-506670624        | S3               | India        | Gujarat Technologies                      | గుజరాత్ టెక్నాలజీస్            |                  0.0512821 |                    0 |
| S1-996196567        | S2-144760741        | S2               | India        | Swastik Consultants                       | स्वस्तिक कंसल्टेंट्स            |                  0.0512821 |                    0 |
| S1-996196567        | S3-941021301        | S3               | India        | Swastik Consultants                       | स्वस्तिक कंसल्टेंट्स            |                  0.0512821 |                    0 |
| S1-13148852         | S3-476520454        | S3               | India        | Gujarat Technologies                      | గుజరాత్ టెక్నాలజీస్            |                  0.0512821 |                    0 |
| S1-146904342        | S2-920673079        | S2               | India        | Indian Constructions                      | ਇੰਡੀਅਨ ਕੰਸਟ੍ਰਕਸ਼ਨਜ਼          |                  0.0512821 |                    0 |
| S1-160773323        | S2-761258285        | S2               | India        | Laxmi Enterprises                         | লক্ষ্মী এন্টারপ্রাইজেস          |                  0.0512821 |                    0 |
| S1-761468356        | S3-632260632        | S3               | India        | Premier Engineering                       | प्रीमियर इंजीनियरिंग        |                  0.0512821 |                    0 |


## 16. Lowest-Similarity Positive Address Pairs


These are genuine ground-truth matches with comparatively low character-level address similarity within the analyzed similarity sample.


| source1_entity_id   | matched_entity_id   | matched_source   | s1_country   | s1_business_address                                                                                                                                | matched_business_address   |   address_sequence_similarity |   address_token_jaccard |
|:--------------------|:--------------------|:-----------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------|------------------------------:|------------------------:|
| S1-322389605        | S2-256585223        | S2               | US           | 328 Spring Street, Ossining, NY                                                                                                                    |                            |                             0 |                       0 |
| S1-66721445         | S3-139061677        | S3               | US           | 2424 John Cox Place, City Of El Paso, TX                                                                                                           |                            |                             0 |                       0 |
| S1-204791742        | S3-684974903        | S3               | India        | 23, Nagathamman Koil St 12Th Avenue, Ashok Nagar, Chennai, Tamil Nadu                                                                              |                            |                             0 |                       0 |
| S1-275239484        | S2-810206705        | S2               | India        | 1725 Dariba Kalan Dariba, Kalan Delhi Central, Delhi, North Delhi, Delhi                                                                           |                            |                             0 |                       0 |
| S1-641817890        | S2-289543055        | S2               | US           | 16669 2493, Tyler, TX                                                                                                                              |                            |                             0 |                       0 |
| S1-65871194         | S3-32400007         | S3               | US           | Arverne, 109 56 Place, NY                                                                                                                          |                            |                             0 |                       0 |
| S1-392134615        | S2-60325419         | S2               | US           | Murfreesboro, Unit D, TN, 2434 Willowbrook Drive                                                                                                   |                            |                             0 |                       0 |
| S1-925840718        | S3-810104532        | S3               | US           | 12788 177th Avenue, Goodyear, AZ                                                                                                                   |                            |                             0 |                       0 |
| S1-426790673        | S3-745660534        | S3               | India        | Plot No.- 52, Saheed Nagar, Bhubaneswar, Khordha, Orissa                                                                                           |                            |                             0 |                       0 |
| S1-607278049        | S3-761894336        | S3               | US           | 305 Collin Circle, Bloomingdale, IL                                                                                                                |                            |                             0 |                       0 |
| S1-886012300        | S2-335725953        | S2               | US           | 4805 Autumnwood Drive, Henrico County, VA                                                                                                          |                            |                             0 |                       0 |
| S1-606804763        | S2-199887157        | S2               | India        | Shop No 3, Giridarshan Bldg, Old Agra Road, Thane, Maharashtra                                                                                     |                            |                             0 |                       0 |
| S1-216195205        | S2-18187593         | S2               | US           | 305 Briarwood Lane, Palatine, IL                                                                                                                   |                            |                             0 |                       0 |
| S1-324165385        | S3-167350825        | S3               | India        | House No 234 Kh. No 11/23/2 & 11/24/2, Ground Floor Vardhman Enclave Blk-F Village Karala Landmark Near Water Tank, Delhi, North West Delhi, Delhi |                            |                             0 |                       0 |
| S1-322389605        | S2-448232659        | S2               | US           | 328 Spring Street, Ossining, NY                                                                                                                    |                            |                             0 |                       0 |
| S1-65263544         | S2-998270769        | S2               | US           | 1 Greenbrier Drive, Kimberling City, MO                                                                                                            |                            |                             0 |                       0 |
| S1-706930694        | S3-461361668        | S3               | US           | 8343 B Greensboro Drive, Unit CND 4, Fairfax County, VA                                                                                            |                            |                             0 |                       0 |
| S1-261697610        | S2-509844651        | S2               | US           | 5335 Hydraulic Avenue, Fl 0, Wichita, KS                                                                                                           |                            |                             0 |                       0 |
| S1-58997889         | S2-34159244         | S2               | India        | H.No.61, Basai Enclave, Part-Ii, Near Green Field Public School, Garauli Road, Railway Road, Gurgaon, Haryana                                      |                            |                             0 |                       0 |
| S1-428577205        | S2-778640327        | S2               | India        | Plot No 2 Flat No 67Unique Group Housing Society Sector-13 Rohini, Delhi, North Delhi, Delhi                                                       |                            |                             0 |                       0 |
| S1-338580819        | S3-882461134        | S3               | US           | 19 Norvel Road, Norwalk, CT                                                                                                                        |                            |                             0 |                       0 |
| S1-836630456        | S2-279932629        | S2               | India        | Unit No F8- K, Lobby Level At Grand Hyatt Hotel, Mumbai, Maharashtra                                                                               |                            |                             0 |                       0 |
| S1-836630456        | S2-310966922        | S2               | India        | Unit No F8- K, Lobby Level At Grand Hyatt Hotel, Mumbai, Maharashtra                                                                               |                            |                             0 |                       0 |
| S1-859172960        | S3-923547406        | S3               | US           | 125 Keats Avenue, Greenburgh, NY                                                                                                                   |                            |                             0 |                       0 |
| S1-357814676        | S2-586689814        | S2               | India        | Sree Seetha Palace, Plot No.113 And 114, F. No.303 Madhavi Cooperative Society, Kukatpally, Hyderabad, Telangana                                   |                            |                             0 |                       0 |
| S1-357814676        | S3-69858689         | S3               | India        | Sree Seetha Palace, Plot No.113 And 114, F. No.303 Madhavi Cooperative Society, Kukatpally, Hyderabad, Telangana                                   |                            |                             0 |                       0 |
| S1-449586730        | S3-919074199        | S3               | India        | 808, 8Th Floor Vijaya Building 17 Barakhamba Road, Delhi, East Delhi, Delhi                                                                        |                            |                             0 |                       0 |
| S1-163556256        | S3-261433142        | S3               | India        | A-11, The Amin Co.Op. Housing Society, Bage Nishat Society, Opp.Sonal Cinema, Ve, Jalpur, Ahmedabad, Gujarat                                       |                            |                             0 |                       0 |
| S1-139380525        | S3-906822043        | S3               | US           | 4828 A Delridge Way, Seattle, WA                                                                                                                   |                            |                             0 |                       0 |
| S1-29845983         | S3-588502663        | S3               | US           | 33 Sleepy Hollow Drive, Danbury, CT                                                                                                                |                            |                             0 |                       0 |
| S1-965667           | S3-860443364        | S3               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                                                   |                            |                             0 |                       0 |
| S1-685349968        | S3-762259318        | S3               | US           | 341 Ellerman Street, Piqua, OH                                                                                                                     |                            |                             0 |                       0 |
| S1-952882718        | S2-824408943        | S2               | US           | 6119 Glass Peak Lane, Richmond, TX                                                                                                                 |                            |                             0 |                       0 |
| S1-937665695        | S2-969898980        | S2               | US           | 98 Locust Street, Unit 9, Normal, IL                                                                                                               |                            |                             0 |                       0 |
| S1-276513891        | S2-343915465        | S2               | US           | Louisville, TN, 138 Fine Lane                                                                                                                      |                            |                             0 |                       0 |
| S1-245891332        | S3-197649025        | S3               | US           | 611 Bordeaux Street, Chadron, NE                                                                                                                   |                            |                             0 |                       0 |
| S1-840162906        | S2-129529678        | S2               | US           | 95 Forest Edge Drive, Eads, TN                                                                                                                     |                            |                             0 |                       0 |
| S1-686866467        | S2-979927445        | S2               | India        | A 192 A Lajpat Nagar, Ghaziabad, Uttar Pradesh                                                                                                     |                            |                             0 |                       0 |
| S1-786926871        | S3-46362087         | S3               | US           | 4702 Jade Street, Marion County, OR                                                                                                                |                            |                             0 |                       0 |
| S1-227070667        | S3-270311318        | S3               | US           | 8402 Heatherwood Lane, Fl 0, Pasadena, MD                                                                                                          |                            |                             0 |                       0 |
| S1-267781387        | S3-171810891        | S3               | US           | 89 Rockridge Road, Paynesville, WV                                                                                                                 |                            |                             0 |                       0 |
| S1-297300907        | S3-414739166        | S3               | India        | 1-3-11/2, Mamidipalem Ongole, Ongole, Prakasam, Andhra Pradesh                                                                                     |                            |                             0 |                       0 |
| S1-318247042        | S2-151135165        | S2               | India        | Villa No. 23, Vaswani Whispering Palms, Outer Ring Road, Marathahalli, Bangalore, Karnataka                                                        |                            |                             0 |                       0 |
| S1-991945738        | S3-530041253        | S3               | India        | East Delhi, 95 B Village Patpar Ganj, East Delhi, Delhi                                                                                            |                            |                             0 |                       0 |
| S1-923836709        | S2-909024870        | S2               | India        | 22, Ramdev Industrial Estate Opp Shahwadi Bus Stand, B/H Ashish Narsu, Ry, Narol, Ahmedabad, Gujarat                                               |                            |                             0 |                       0 |
| S1-291511476        | S2-250528724        | S2               | US           | 1230 45th Street, Vancouver, WA                                                                                                                    |                            |                             0 |                       0 |
| S1-125940535        | S3-707623358        | S3               | US           | 555 Newell Street, Unit 42, Bellefontaine, OH                                                                                                      |                            |                             0 |                       0 |
| S1-771630457        | S2-659021954        | S2               | US           | Chapel Hill, 114 Salix Street, NC                                                                                                                  |                            |                             0 |                       0 |
| S1-198921555        | S3-789244764        | S3               | US           | TX, Houston, 7122 Pavilion Drive                                                                                                                   |                            |                             0 |                       0 |
| S1-198921555        | S3-232308764        | S3               | US           | TX, Houston, 7122 Pavilion Drive                                                                                                                   |                            |                             0 |                       0 |


## 17. Representative Non-Exact Business Name Pairs


| source1_entity_id   | matched_entity_id   | matched_source   | s1_country   | s1_business_name                         | matched_business_name                 | name_variation_category   |
|:--------------------|:--------------------|:-----------------|:-------------|:-----------------------------------------|:--------------------------------------|:--------------------------|
| S1-965667           | S2-681193310        | S2               | US           | Maure Williams Colombier Inc             | Maure Wilblims Colombier Inc          | Other                     |
| S1-965667           | S2-743505751        | S2               | US           | Maure Williams Colombier Inc             | Maure Williams Colombier              | Other                     |
| S1-965667           | S3-775321672        | S3               | US           | Maure Williams Colombier Inc             | Dréxkor                               | Other                     |
| S1-965667           | S3-11291185         | S3               | US           | Maure Williams Colombier Inc             | maurewilliamscolombier.com            | Other                     |
| S1-965667           | S3-860443364        | S3               | US           | Maure Williams Colombier Inc             | Maure Williams Inc Center             | Other                     |
| S1-55344266         | S2-249013014        | S2               | India        | Raj Investments LLP                      | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி                      | Other                     |
| S1-55344266         | S3-478195123        | S3               | India        | Raj Investments LLP                      | Raj Investments எல்எல்பி                 | Other                     |
| S1-55344266         | S3-384364074        | S3               | India        | Raj Investments LLP                      | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி                      | Other                     |
| S1-343815751        | S2-790675320        | S2               | US           | Dahlia Power Reliable Scientific LLC     | Dahlia Power Reliable                 | Other                     |
| S1-343815751        | S2-479876582        | S2               | US           | Dahlia Power Reliable Scientific LLC     | Dahlia Power Reliable Scientific      | Other                     |
| S1-343815751        | S3-878454467        | S3               | US           | Dahlia Power Reliable Scientific LLC     | Dahlia Ponr Reliable Scientific LLC   | Other                     |
| S1-656753428        | S2-153058913        | S2               | India        | Ss Food Private Limited                  | एसएस फूड प्राइवेट लिमिटेड                  | Other                     |
| S1-656753428        | S2-24659151         | S2               | India        | Ss Food Private Limited                  | एसएस फूड प्राइवेट लिमिटेड                  | Other                     |
| S1-656753428        | S3-679606215        | S3               | India        | Ss Food Private Limited                  | एसएस फूड प्राइवेट लिमिटेड                  | Other                     |
| S1-102811957        | S2-478959098        | S2               | US           | Payne Enterprises                        | Payne Énterprises                     | Punctuation               |
| S1-102811957        | S2-553508714        | S2               | US           | Payne Enterprises                        | Payne Enterpires                      | Other                     |
| S1-102811957        | S2-625774905        | S2               | US           | Payne Enterprises                        | PAYNE-ENRTPRMISES                     | Other                     |
| S1-102811957        | S3-728090388        | S3               | US           | Payne Enterprises                        | Payne Etrepndiels                     | Other                     |
| S1-102811957        | S3-928796641        | S3               | US           | Payne Enterprises                        | Payne Énterprises                     | Punctuation               |
| S1-102811957        | S3-449308785        | S3               | US           | Payne Enterprises                        | Payne Enterprises  LLC                | Other                     |
| S1-18727616         | S2-755677256        | S2               | US           | Lumay Boral                              | Lumay Boral Inc.                      | Other                     |
| S1-18727616         | S3-187831601        | S3               | US           | Lumay Boral                              | Lumay Bóral                           | Punctuation               |
| S1-18727616         | S3-476250621        | S3               | US           | Lumay Boral                              | Lumay Bóral                           | Punctuation               |
| S1-318373630        | S2-660036492        | S2               | India        | Red Ventures Private Limited             | रेड वेंचर्स प्राइवेट लिमिटेड                  | Other                     |
| S1-318373630        | S3-804600254        | S3               | India        | Red Ventures Private Limited             | Red Ventures Private                  | Other                     |
| S1-86989137         | S3-274817120        | S3               | India        | Laxmi Golden Investments Private Limited | Laxmi Gbn lnvestments Private Limited | Other                     |
| S1-86989137         | S3-312496301        | S3               | India        | Laxmi Golden Investments Private Limited | Laxmi Golden Investments              | Other                     |
| S1-29845983         | S2-648035184        | S2               | US           | Hendricks and Flowers Inc                | Hendricks and  Flowers Inc            | Whitespace                |
| S1-29845983         | S3-588502663        | S3               | US           | Hendricks and Flowers Inc                | Hendricks and Inc Flowers             | Token order               |
| S1-789009573        | S2-383871912        | S2               | India        | Hotel Enterprises Limited                | होटल एंटरप्राइजेज लिमिटेड                  | Other                     |
| S1-789009573        | S3-74481402         | S3               | India        | Hotel Enterprises Limited                | Hotel Limited Services                | Other                     |
| S1-789009573        | S3-576451439        | S3               | India        | Hotel Enterprises Limited                | Hotel Énterprises Limited             | Punctuation               |
| S1-730934468        | S2-356983532        | S2               | US           | Orellana Investments LLC                 | Orellana Investments Investments Llc  | Other                     |
| S1-730934468        | S3-352439310        | S3               | US           | Orellana Investments LLC                 | LLC Orellana Invsmbens                | Other                     |
| S1-7293388          | S2-7028416          | S2               | India        | Chordia & Partners                       | Chordia & Partners Company            | Other                     |
| S1-7293388          | S2-442723188        | S2               | India        | Chordia & Partners                       | Chordia + Pagnters - 7306204978       | Other                     |
| S1-7293388          | S2-157073701        | S2               | India        | Chordia & Partners                       | Chordia &-Pártners Ltd                | Other                     |
| S1-7293388          | S3-523120965        | S3               | India        | Chordia & Partners                       | Smt Chordia  & Center                 | Other                     |
| S1-546142636        | S2-487600131        | S2               | US           | Crystal Staffing Solutions LLC           | Crystal Solutions LLC Partners        | Other                     |
| S1-546142636        | S2-582477216        | S2               | US           | Crystal Staffing Solutions LLC           | CRYSTAL STAFFING SOLUTIONS-L.L.C.     | Formatting / symbols      |
| S1-546142636        | S2-392804085        | S2               | US           | Crystal Staffing Solutions LLC           | LLC Crystal Sttfrifng Solutions       | Other                     |
| S1-546142636        | S3-200008747        | S3               | US           | Crystal Staffing Solutions LLC           | Llc Crystal Staffing Solutions        | Token order               |
| S1-546142636        | S3-729771680        | S3               | US           | Crystal Staffing Solutions LLC           | LLC Crystal Shaffing Solutions        | Other                     |
| S1-546142636        | S3-249331830        | S3               | US           | Crystal Staffing Solutions LLC           | Crystal                               | Other                     |
| S1-274126313        | S2-736616474        | S2               | US           | Obsidian, LLC                            | Obsidian,-LLC                         | Formatting / symbols      |
| S1-274126313        | S2-680265918        | S2               | US           | Obsidian, LLC                            | obsidian, llc                         | Case only                 |
| S1-274126313        | S2-51486805         | S2               | US           | Obsidian, LLC                            | Obsidian, LLC Center                  | Other                     |
| S1-274126313        | S3-925631694        | S3               | US           | Obsidian, LLC                            | Obsidian, Llc                         | Case only                 |
| S1-274126313        | S3-461175723        | S3               | US           | Obsidian, LLC                            | Obsidian, [[LLC]]                     | Punctuation               |
| S1-274126313        | S3-850112871        | S3               | US           | Obsidian, LLC                            | Korbrixx D.B.A. Obsidian, LLC         | Other                     |


## 18. Representative Non-Exact Business Address Pairs


| source1_entity_id   | matched_entity_id   | matched_source   | s1_country   | s1_business_address                                                                                                    | matched_business_address                                                                                           | address_variation_category   |
|:--------------------|:--------------------|:-----------------|:-------------|:-----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------|
| S1-965667           | S2-681193310        | S2               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                       |                                                                                                                    | Other                        |
| S1-965667           | S2-743505751        | S2               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                       |                                                                                                                    | Other                        |
| S1-965667           | S3-775321672        | S3               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                       | 85 Wanye Avenue, Ticonderoga Townshiip, New York                                                                   | Other                        |
| S1-965667           | S3-11291185         | S3               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                       | Wayne Ave, Ticonderoga Townshiip, New York                                                                         | Other                        |
| S1-965667           | S3-860443364        | S3               | US           | 85 Wayne Avenue, Ticonderoga, NY                                                                                       |                                                                                                                    | Other                        |
| S1-55344266         | S2-249013014        | S2               | India        | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu                                                      | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu                                                  | Case only                    |
| S1-55344266         | S2-197070651        | S2               | India        | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu                                                      | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu                                                  | Case only                    |
| S1-55344266         | S3-478195123        | S3               | India        | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu                                                      | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN                                                          | Other                        |
| S1-55344266         | S3-384364074        | S3               | India        | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu                                                      | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு                                                     | Other                        |
| S1-343815751        | S2-790675320        | S2               | US           | 630 45th Terrace, Kansas City, MO                                                                                      | KANSAS CITY, MO, 630 45ND TERRACE, null                                                                            | Other                        |
| S1-343815751        | S2-479876582        | S2               | US           | 630 45th Terrace, Kansas City, MO                                                                                      | 45ND TERRACE, null, KANSAS CITY, MO                                                                                | Other                        |
| S1-343815751        | S3-878454467        | S3               | US           | 630 45th Terrace, Kansas City, MO                                                                                      | Missouri, 630 45th Terrace, Kansas City                                                                            | Other                        |
| S1-656753428        | S2-153058913        | S2               | India        | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh                           | AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश                          | Other                        |
| S1-656753428        | S2-24659151         | S2               | India        | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh                           | AF-0684, Uttar Pradesh, GHAZIABAD, 9487203                                                                         | Other                        |
| S1-656753428        | S3-679606215        | S3               | India        | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh                           | Af-684, Ghaziabad, UP                                                                                              | Other                        |
| S1-102811957        | S2-478959098        | S2               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | 3315 FREMONT ST, PEORIA, IL                                                                                        | Other                        |
| S1-102811957        | S2-553508714        | S2               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | 3315 FREMONT ST, PEORIA, IL                                                                                        | Other                        |
| S1-102811957        | S2-625774905        | S2               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | 3315 FREMONT SAINT, PEORIA, IL                                                                                     | Other                        |
| S1-102811957        | S3-728090388        | S3               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | 3315 Fremont St, Peoria, Illinois                                                                                  | Other                        |
| S1-102811957        | S3-928796641        | S3               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | 3315 Fremont Street, Peoria, Illinois                                                                              | Other                        |
| S1-102811957        | S3-449308785        | S3               | US           | 3315 Fremont Street, Peoria, IL                                                                                        | Fremont St, Peoria, Illinois                                                                                       | Other                        |
| S1-18727616         | S2-755677256        | S2               | US           | 1056 Belden Avenue, Akron, OH                                                                                          | 1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH                                                                       | Other                        |
| S1-18727616         | S3-187831601        | S3               | US           | 1056 Belden Avenue, Akron, OH                                                                                          | 1056c Belden Ave, AKON, Ohio                                                                                       | Other                        |
| S1-18727616         | S3-641489370        | S3               | US           | 1056 Belden Avenue, Akron, OH                                                                                          | 1056c Belden Ave, AKON, Ohio                                                                                       | Other                        |
| S1-18727616         | S3-476250621        | S3               | US           | 1056 Belden Avenue, Akron, OH                                                                                          | 1056c Belden Avenue, AKON, Ohio                                                                                    | Other                        |
| S1-318373630        | S2-660036492        | S2               | India        | Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1                                             | G-1, BANIPARK, JAIPUR, Rajasthan                                                                                   | Other                        |
| S1-318373630        | S3-804600254        | S3               | India        | Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1                                             | Doro No 316 G-1, Gokul Apartment, E-3a Kanti Chandra Road, Banipark, Subhash Nagar, RJ                             | Other                        |
| S1-86989137         | S3-274817120        | S3               | India        | New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka | New Bridge Bssiness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, ಕರ್ನಾಟಕ | Other                        |
| S1-86989137         | S3-312496301        | S3               | India        | New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka | New Bridge Buisness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, KA    | Other                        |
| S1-29845983         | S2-648035184        | S2               | US           | 33 Sleepy Hollow Drive, Danbury, CT                                                                                    | CT, SLEEPY HOLLOW DRIVE, DANBURY                                                                                   | Other                        |
| S1-29845983         | S3-588502663        | S3               | US           | 33 Sleepy Hollow Drive, Danbury, CT                                                                                    |                                                                                                                    | Other                        |
| S1-789009573        | S2-383871912        | S2               | India        | Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi         | WZ-187C SHOP NO.13, DELHI, WEST DELHI, Delhi                                                                       | Other                        |
| S1-789009573        | S3-74481402         | S3               | India        | Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi         | Block B-517 Wz-187c Shop No.13, Divreportingcircle, West Delhi, DL                                                 | Other                        |
| S1-789009573        | S3-576451439        | S3               | India        | Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi         | Block B-517 Wz-187c Shop No.13, South West Delhi, Delhi, DL                                                        | Other                        |
| S1-730934468        | S2-356983532        | S2               | US           | 728 A Quail Avenue, Fl Ground Floor, Geneva, IA                                                                        |                                                                                                                    | Other                        |
| S1-730934468        | S3-352439310        | S3               | US           | 728 A Quail Avenue, Fl Ground Floor, Geneva, IA                                                                        | 728 A Quail Avenue, Fl. Ground Floor, Geneva, Iowa                                                                 | Other                        |
| S1-7293388          | S2-7028416          | S2               | India        | Faridabad, 1038 Sector 9, Haryana                                                                                      | हरियाणा, 1038 SECTOR 9, FARIDABAD                                                                                  | Other                        |
| S1-7293388          | S2-442723188        | S2               | India        | Faridabad, 1038 Sector 9, Haryana                                                                                      | हरियाणा, DOOR NO 1038 SECTOR 9, FARIDABAD                                                                          | Other                        |
| S1-7293388          | S2-157073701        | S2               | India        | Faridabad, 1038 Sector 9, Haryana                                                                                      | H.NO 1038 SECTOR 9, FARIABAD, Haryana                                                                              | Other                        |
| S1-7293388          | S3-523120965        | S3               | India        | Faridabad, 1038 Sector 9, Haryana                                                                                      | #1038 Sector 9, Faridabad, हरियाणा                                                                                 | Other                        |
| S1-546142636        | S2-487600131        | S2               | US           | 8706 Kentucky Derby Drive, Waxhaw, NC                                                                                  | 8706 KENTUCKY DERBY DR, WAXHAW, NC                                                                                 | Other                        |
| S1-546142636        | S2-582477216        | S2               | US           | 8706 Kentucky Derby Drive, Waxhaw, NC                                                                                  | 8706 KENTUCKY DERBY DR, WAXHAW, NC                                                                                 | Other                        |
| S1-546142636        | S2-392804085        | S2               | US           | 8706 Kentucky Derby Drive, Waxhaw, NC                                                                                  | 8706 KENTUCKY DERBY DRIVE, WAXHAW, NC                                                                              | Case only                    |
| S1-546142636        | S3-200008747        | S3               | US           | 8706 Kentucky Derby Drive, Waxhaw, NC                                                                                  | 870 Kentucky Derby Drive, Waxhaw, North Carolina                                                                   | Other                        |
| S1-546142636        | S3-729771680        | S3               | US           | 8706 Kentucky Derby Drive, Waxhaw, NC                                                                                  | 870 Kentucky Derby Drive, Waxhaw, North Carolina                                                                   | Other                        |
| S1-274126313        | S2-736616474        | S2               | US           | 3907 Hamilton Road, Deer Park, WA                                                                                      | 3907 HAMILTON RD, DEER PARK CIYT, WA                                                                               | Other                        |
| S1-274126313        | S2-680265918        | S2               | US           | 3907 Hamilton Road, Deer Park, WA                                                                                      |                                                                                                                    | Other                        |
| S1-274126313        | S2-51486805         | S2               | US           | 3907 Hamilton Road, Deer Park, WA                                                                                      | 3907 HAMILTON RD, DEER PARK CIYT, WA                                                                               | Other                        |
| S1-274126313        | S3-925631694        | S3               | US           | 3907 Hamilton Road, Deer Park, WA                                                                                      | Deer Park, Washington, Hamilton Rd                                                                                 | Other                        |
| S1-274126313        | S3-461175723        | S3               | US           | 3907 Hamilton Road, Deer Park, WA                                                                                      |                                                                                                                    | Other                        |


## 19. Data-Driven Observations


- Exact business-name equality occurs in **4.64%** of positive pairs.

- Exact business-address equality occurs in **2.23%** of positive pairs.

- Case-insensitive business-name equality occurs in **10.75%** of positive pairs.

- Case-insensitive business-address equality occurs in **7.24%** of positive pairs.

- Token-order-normalized equality occurs in **26.91%** of business names.

- Token-order-normalized equality occurs in **11.64%** of addresses.

- Positive pairs with an empty Source 1 address: **0** (0.0%).

- Positive pairs with an empty matched address: **337,018** (4.41%).


## 20. Candidate Normalization Areas to Review


The following are analysis categories for the next M1 stage. They are not automatically approved normalization rules. Each should be validated against observed examples and false-match risk before implementation.


1. Unicode and case handling
2. Whitespace normalization
3. Punctuation handling
4. Symbol and separator handling
5. Token-order variation
6. Business-name abbreviation/legal-suffix variation
7. Address abbreviation variation
8. Numeric/address-component preservation
9. Empty-address handling
10. Country-independent normalization behavior
11. Transliteration or Unicode-script variation if observed
12. Conservative preservation of discriminative tokens


These areas must be evaluated using both positive matches and negative/non-match pairs before becoming part of the production normalizer.


## 21. Next M1 Stage


The next stage is to inspect the observed variation examples and implement `text_normalizer.py`. The normalizer should produce multiple representations where useful rather than destroying information through aggressive normalization. Its output will later be consumed by the blocking and candidate-generation stage.
