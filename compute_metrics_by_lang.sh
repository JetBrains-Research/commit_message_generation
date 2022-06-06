#!/bin/bash
for lang in TypeScript Java Go C++ JavaScript Python '"C#"' Swift C Ruby PHP Kotlin
do
python compute_metrics.py +language=$lang
done