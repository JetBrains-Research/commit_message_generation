# Specific percentiles values

During "Filtering outliers" step in our dataset processing pipeline, we dropped examples out of [5% percentile, 95% percentile] range.
In this file, we provide specific values for these percentiles.

|         Feature         |5% percentile| 95% percentile |
|:-----------------------:|:-----------:|:--------------:|
| Messages: # characters  |     12      |      491       |	
|   Messages: # tokens    |      2      |       53       |
|  Diffs: # characterss   |     240     |     42 785     |	
|     Diffs: # tokens     |     20      |      3740      |
| Diffs: # modified files |      1      |       16       |