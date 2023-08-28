# NewsVerify

## Conda 
Gen env 
```shell
conda env export > environment.yml
```

Create env 
```shell
conda env create -f environment.yml
```

## Use 

```shell
conda activate huggingface
python main.py
```

```shell
python -m crawl_evidences.nli_fever_wiki_evidences --verbose 2
```