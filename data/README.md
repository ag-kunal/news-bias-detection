# Dataset

The datasets used for the News Bias Detection research project are stored separately from this GitHub repository because of their size.

## Dataset Access

- **Shared download link:**[ https://drive.google.com/file/d/1_ABeC4Ly4jwbRXusJYHR9XUojUIdUgCR/view?usp=sharing](https://drive.google.com/file/d/1_ABeC4Ly4jwbRXusJYHR9XUojUIdUgCR/view?usp=sharing)
- **Dataset version:** Current cleaned dataset
- **Access:** Project team members

## Expected Files

Download the approved files into this directory:

- `Akshit_Dataset.csv`
- `Deep_Dataset.csv`
- `Khoshang_dataset.csv`
- `Kunal_Dataset.csv`
- `Combined_Dataset.csv`

## Dataset Schema

| Column     | Description                                     |
| ---------- | ----------------------------------------------- |
| `Title`  | News article title                              |
| `Text`   | News article body                               |
| `Bias`   | Target label:`left`, `center`, or `right` |
| `Source` | News publisher or source                        |

## Usage Guidelines

1. Use the approved cleaned datasets for feature extraction and experiments.
2. Do not modify the original CSV files directly.
3. Do not independently remove rows or change labels.
4. Preserve the original article records when generating feature datasets.
5. Keep generated feature CSVs and experiment outputs out of GitHub unless explicitly required.
6. If the dataset is updated, communicate the change to the team and update the dataset version above.

Dataset files are excluded from Git tracking. Each team member should download the files separately before running scripts.
