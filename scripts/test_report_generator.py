import json
from modules.report_generator import generate_dataset_report, generate_model_report, save_report

def main():
    profile = {"columns": ["a", "b"], "rows": 3}
    cleaning = {"drops": [], "notes": "none"}

    dataset_report = generate_dataset_report(profile, cleaning)
    model_report = generate_model_report("regression", "LinearRegression", {"LinearRegression": {"r2": 0.9}})

    ds_path = save_report(dataset_report, "smoke_test_dataset_report")
    mr_path = save_report(model_report, "smoke_test_model_report")

    print("dataset_report_path:", ds_path)
    print("model_report_path:", mr_path)

    # Print small snippets of saved files
    with open(ds_path, "r", encoding="utf-8") as f:
        print("dataset_report_json:", f.read()[:500])

    with open(mr_path, "r", encoding="utf-8") as f:
        print("model_report_json:", f.read()[:500])

if __name__ == "__main__":
    main()
