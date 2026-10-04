## OMDD-NET

### Datasets

For the BE setting, we directly follow the datasets of OBDD-NET. The dataset archives are provided by the OBDD-NET repository as `datasets/large_datasets_part.tar.gz.*` and `datasets/small_datasets.tar.gz`. Download these files and extract them into the `./datasets` directory of this project:

```bash
cd OMDD-NET
cat datasets/large_datasets_part.tar.gz.* | tar -xzvf - -C ./datasets
tar -xzvf datasets/small_datasets.tar.gz -C ./datasets
```

For the AP setting, the datasets are already prepared in this repository as `datasets/large_datasets_ap_part.tar.gz.*` and `datasets/small_datasets_ap.tar.gz`. Extract them into `./datasets`, and the processed CSV files will be placed under `datasets/small_datasets/processed` and `datasets/large_datasets/processed`.

```bash
cd OMDD-NET
tar -xzvf datasets/small_datasets_ap.tar.gz -C ./datasets
cat datasets/large_datasets_ap_part.tar.gz.* | tar -xzvf - -C ./datasets
```

### Requirements

The experiments were conducted with `Python 3.11`, and the requirements are summarized in `requirements.txt`.

```bash
pip install -r requirements.txt
```


### Code

Conduct experiment for 5-fold cross-validation on a small dataset `anneal-un.csv` under the BE setting:

```bash
python ./experiments/base_exper.py --mode 5-fold --timeout 900 --interval_num 2 --step_num 10000 --net_depth 6 --train_file ./datasets/small_datasets/anneal-un.csv --device cuda:0 
```

Conduct experiment for 5-fold cross-validation on a large dataset `adult_bin.csv` under the BE setting:

```bash
python ./experiments/base_exper.py --mode 5-fold --timeout 1800 --interval_num 2 --step_num 20000 --net_depth 6 --train_file ./datasets/large_datasets/adult_bin.csv --device cuda:0 
```

---

Conduct experiment for 5-fold cross-validation on a small dataset `anneal_processed.csv` under the AP setting with $D=3$:

```bash
python ./experiments/base_exper.py --mode k-fold --kfold 5 --timeout 900 --interval_num 3 --step_num 10000 --net_depth 6 --train_file ./datasets/small_datasets/processed/anneal_processed.csv --device cuda:0
```

Conduct experiment for 5-fold cross-validation on a large dataset `adult_processed.csv` under the AP setting with $D=3$:

```bash
python ./experiments/base_exper.py --mode k-fold --kfold 5 --timeout 1800 --interval_num 3 --step_num 20000 --net_depth 6 --train_file ./datasets/large_datasets/processed/adult_processed.csv --device cuda:0
```

Replace `--train_file` with another CSV file in the same setting to run the remaining datasets. The experimental results will be stored in the `results` folder.


