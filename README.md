# Commit messages completion (and generation!)

This branch allows to evaluate models for both commit messages completion and generation tasks. 

## Usage

See steps 1-4 in main branch README. Note that there is an additional option for `model` part of the config:
* `actual_generation`: **true** if you want evaluate a model for generation task **false** otherwise

Next, to evaluate a model, define configuration at `conf/config.yaml` and run the following command:
 
```
python eval.py
```