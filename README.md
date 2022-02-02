# ML-augmented screening algorithm for diabetes detection

## Code structure
```
.
├── lxh_prediction  # defined the models, configs, common functions for plots
│   └── models
├── scripts         # scripts to generat the figures and experimental results
│   ├── 10.supp_figure_5.ipynb
│   ├── 11.supp_table_delong_test.ipynb
│   ├── 12.supp_table_missing_data.R
│   ├── 1.data_preprocessing.py
│   ├── 2.figure_3.ipynb
│   ├── 3.figure_4.ipynb
│   ├── 4.figure_5.ipynb
│   ├── 5.figure_6.ipynb
│   ├── 6.supp_figure_1.ipynb
│   ├── 7.supp_figure_2.ipynb
│   ├── 8.supp_figure_3.ipynb
│   └── 9.supp_figure_4.ipynb
├── experiments     # configs for parameter tuning using TPE and NNI
│   ├── ann
│   └── lightgbm
├── README.md
├── requirements.txt
└── setup.py
```

## Usage
### Install dependencies
```bash
git clone https://github.com/longcw/ml-for-diabetes-screening.git
cd ml-for-diabetes-screening
pip install -r requirements.txt
python setup.py develop
```

### Run experiments
Run script in `scripts/` to generate the corresponding results.