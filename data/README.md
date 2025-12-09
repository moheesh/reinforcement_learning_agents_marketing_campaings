# Data Directory

This directory contains the raw data files required to run the Marketing Mix Optimization system.

## Required Files

Place the following files in the `data/raw/` folder:

| File | Description |
|------|-------------|
| SecondFile.csv | Main marketing data with channel spend and revenue |
| SpecialSale.csv | Promotional calendar with sale events |

## Data Source

The dataset used in this project is a marketing analytics dataset containing monthly marketing spend across multiple channels and corresponding revenue figures.

### Option 1: Use Sample Data

Sample data files are available in the project submission. Copy them to:

```
data/
└── raw/
    ├── SecondFile.csv
    └── SpecialSale.csv
```

### Option 2: Create Your Own Data

If using your own data, ensure the CSV files match the expected format below.

## File Formats

### SecondFile.csv

Main marketing data file with the following columns:

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

Example row:
```
month,TV,Digital,Sponsorship,Content.Marketing,Online.marketing,Affiliates,SEM,Radio,Other,total_gmv,NPS,Date
Jan 2016,44000000,5000000,42000000,9000000,229000000,74000000,42000000,27000000,271000000,387210245,47.1,2016-01-01
```

### SpecialSale.csv

Promotional calendar file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Date | Date | Date of the sale event |
| Sales Name | String | Name of the promotional event |

Example rows:
```
Date,Sales Name
2015-10-01,Diwali Sale
2015-12-25,Christmas Sale
2016-01-26,Republic Day Sale
```

## Data Validation

After placing the files, verify the data by running:

```bash
python -c "from data_processor import DataProcessor; dp = DataProcessor(); dp.load_data(); print('Data loaded successfully')"
```

## Notes

- Ensure all numeric columns contain valid numbers (no text or missing values)
- Date formats should be consistent
- Channel names must match exactly as specified in config.yaml
- The system expects at least 6 months of data for meaningful results

## Privacy

The data files are excluded from version control via .gitignore to protect potentially sensitive business information. Do not commit actual business data to public repositories.