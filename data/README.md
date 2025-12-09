# Data Directory

This directory contains the raw data files required to run the Marketing Mix Optimization system.

## Data Source

**Dataset:** DT MART: Market Mix Modeling

**Source:** Kaggle

**URL:** https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling/data

**License:** Please refer to Kaggle for dataset license and usage terms.

## Required Files

Download all files from the Kaggle dataset and place them in the `data/raw/` folder:

```
data/
└── raw/
    ├── MediaInvestment.csv
    ├── MonthlyNPSscore.csv
    ├── ProductList.csv
    ├── Sales.csv
    ├── Secondfile.csv
    ├── SpecialSale.csv
    └── firstfile.csv
```

## File Descriptions

| File | Description |
|------|-------------|
| MediaInvestment.csv | Monthly marketing spend across channels (TV, Digital, Sponsorship, etc.) |
| MonthlyNPSscore.csv | Net Promoter Score data by month |
| ProductList.csv | Product catalog with product details |
| Sales.csv | Transaction-level sales data with GMV and units |
| Secondfile.csv | Aggregated monthly data with channel spend, revenue, and NPS |
| SpecialSale.csv | Promotional calendar with sale event dates |
| firstfile.csv | Additional aggregated marketing data |

## Primary Files Used

The system primarily uses these two files:

| File | Purpose |
|------|---------|
| Secondfile.csv | Main input file containing monthly channel spend, total GMV, and NPS |
| SpecialSale.csv | Promotional calendar to identify sale periods |

## Secondfile.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| month | Date | Month in format "Mon YYYY" (e.g., "Jan 2016") |
| TV | Numeric | TV advertising spend |
| Digital | Numeric | Digital advertising spend |
| Sponsorship | Numeric | Sponsorship spend |
| Content.Marketing | Numeric | Content marketing spend |
| Online.marketing | Numeric | Online marketing spend |
| Affiliates | Numeric | Affiliate marketing spend |
| SEM | Numeric | Search engine marketing spend |
| Radio | Numeric | Radio advertising spend |
| Other | Numeric | Other marketing spend |
| total_gmv | Numeric | Total Gross Merchandise Value (revenue) |
| NPS | Numeric | Net Promoter Score |
| Date | Date | Date in format YYYY-MM-DD |

## SpecialSale.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| Date | Date | Date of the sale event |
| Sales Name | String | Name of the promotional event |

## How to Download

1. Visit https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling/data
2. Click "Download" button (requires Kaggle account)
3. Extract the downloaded ZIP file
4. Copy all CSV files to `data/raw/` folder

Alternatively, using Kaggle CLI:

```bash
kaggle datasets download -d datatattle/dt-mart-market-mix-modeling
unzip dt-mart-market-mix-modeling.zip -d data/raw/
```

## Data Validation

After placing the files, verify the data by running:

```bash
python -c "from data_processor import DataProcessor; dp = DataProcessor(); dp.load_data(); print('Data loaded successfully')"
```

## Citation

If using this dataset, please cite the original source:

```
DT MART: Market Mix Modeling
Source: Kaggle (https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling)
Author: DataTattle
```

## Notes

- Ensure all CSV files are placed in `data/raw/` folder
- The system expects the exact column names as specified above
- Data files are excluded from version control via .gitignore