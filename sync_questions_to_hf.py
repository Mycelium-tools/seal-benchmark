"""
Sync questions from Google Sheets → Local CSV → HuggingFace

Setup (one-time):
1. In Google Sheets: File → Share → Publish to web
2. Select your sheet, choose "Comma-separated values (.csv)"
3. Copy the URL and paste it below as GOOGLE_SHEETS_URL
4. Make sure HF_TOKEN is set in the .env file

Usage:
    python sync_questions_to_hf.py

This will:
1. Download the latest questions from Google Sheets
2. Save to dataset/seal_questions.csv (with a canary column injected)
3. Upload to HuggingFace
4. Regenerate samples.json
"""

import os
import csv
import subprocess
import requests
from datasets import load_dataset
from huggingface_hub import create_repo, upload_file, login
from dotenv import load_dotenv

from canary import CANARY

load_dotenv()

# Configuration — paste the published-CSV URL of the SEAL Google Sheet here.
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRww74qa899_6LaZqtngTnbSohqkT6Euch6iJa4yahMel6B-u0aOAQGRs7cSW2l8Qxb2hm5UyqYM7IO/pub?gid=0&single=true&output=csv"
LOCAL_CSV = "dataset/seal_questions.csv"
HF_CSV = "seal_questions.csv"          # filename as stored in the HF repo
HF_DATASET = "mycelium-ai/seal-benchmark-questions"


def normalize_csv(path):
    """Rewrite the CSV with leading all-blank rows removed and no BOM.

    Google Sheets' published CSV can emit a blank spacer row above the header
    (e.g. ",,,," before "id,question_1,..."). csv.DictReader would then treat that
    blank row as the header and every lookup ("id", "question_1") would KeyError.
    Stripping leading blank rows makes the real header row 0 for all downstream readers.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    first = next((i for i, r in enumerate(rows) if any(c.strip() for c in r)), None)
    if first is None:
        return  # empty file; leave as-is
    if first > 0:
        print(f"⚠️  Stripped {first} leading blank row(s) from {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows[first:])


def get_existing_ids():
    """Read question IDs from the current local CSV before overwriting it.

    Tolerant by design: a missing, empty, or malformed local file yields an empty
    set (every downloaded row is then reported as new) rather than crashing.
    """
    if not os.path.exists(LOCAL_CSV):
        return set()
    try:
        with open(LOCAL_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "id" not in reader.fieldnames:
                return set()
            return {row["id"] for row in reader if row.get("id")}
    except (OSError, csv.Error):
        return set()


def print_new_questions(old_ids):
    """Compare current local CSV against old_ids and print any new rows."""
    new_rows = []
    with open(LOCAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] not in old_ids:
                new_rows.append(row)

    print("\n" + "=" * 60)
    if not new_rows:
        print("ℹ️  No new questions were added in this sync.")
    else:
        print(f"🆕 {len(new_rows)} new question(s) added this sync:")
        print("=" * 60)
        for row in new_rows:
            q = row.get("question_1") or row.get("question") or ""
            dims = row.get("dimensions") or row.get("tags", "")
            print(f"  ID {row['id']} | Dimensions: {dims} | {q[:120]}{'...' if len(q) > 120 else ''}")
    print("=" * 60)


def download_from_google_sheets():
    """Download CSV from the published Google Sheet."""
    if not GOOGLE_SHEETS_URL:
        print("❌ Error: GOOGLE_SHEETS_URL is not set in sync_questions_to_hf.py")
        print("\nSteps:")
        print("1. Open your SEAL Google Sheet")
        print("2. File → Share → Publish to web")
        print("3. Select 'Comma-separated values (.csv)'")
        print("4. Copy the URL")
        print("5. Paste it as GOOGLE_SHEETS_URL in this file")
        print("\n(Until then, build samples.json from the local CSV with:")
        print("    python sample_questions.py --local )")
        return False

    print("📥 Downloading from Google Sheets...")
    try:
        response = requests.get(GOOGLE_SHEETS_URL, allow_redirects=True)
        response.raise_for_status()
        with open(LOCAL_CSV, "wb") as f:
            f.write(response.content)
        # Strip any leading blank spacer rows Sheets may emit above the header.
        normalize_csv(LOCAL_CSV)
        with open(LOCAL_CSV, "r", encoding="utf-8") as f:
            num_questions = len(f.readlines()) - 1  # -1 for header
        print(f"✅ Downloaded {num_questions} questions to {LOCAL_CSV}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def inject_canary():
    """Append a canary column to every CSV row so all published copies carry it."""
    with open(LOCAL_CSV, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("❌ Error: CSV is empty, skipping canary injection")
        return False
    header, data = rows[0], rows[1:]
    if "canary" not in header:
        header.append("canary")
    canary_idx = header.index("canary")
    for row in data:
        while len(row) <= canary_idx:
            row.append("")
        row[canary_idx] = CANARY
    with open(LOCAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"✅ Injected canary string into {len(data)} rows")
    return True


def upload_to_huggingface():
    """Upload the CSV and a dataset card to the HuggingFace dataset repo."""
    if not os.path.exists(LOCAL_CSV):
        print(f"❌ Error: {LOCAL_CSV} not found")
        return False

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file")
        return False

    login(token=hf_token)

    print(f"📤 Uploading to HuggingFace ({HF_DATASET})...")
    try:
        create_repo(repo_id=HF_DATASET, repo_type="dataset", exist_ok=True, private=False)
        upload_file(
            path_or_fileobj=LOCAL_CSV,
            path_in_repo=HF_CSV,
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message="Sync from Google Sheets",
        )
        readme = (
            "# SEAL Benchmark Questions\n\n"
            "Factual accuracy on animal sentience — question dataset.\n\n"
            f"{CANARY}\n"
        )
        upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{HF_DATASET}")

        print("🔍 Verifying...")
        dataset = load_dataset(HF_DATASET, data_files=HF_CSV, split="train")
        print(f"✅ Verified: {len(dataset)} questions loaded")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\n💡 Make sure HF_TOKEN is set in .env file")
        return False


def main():
    print("=" * 60)
    print("Google Sheets → Local CSV → HuggingFace Sync")
    print("=" * 60)

    old_ids = get_existing_ids()

    if not download_from_google_sheets():
        return
    if not inject_canary():
        return
    if not upload_to_huggingface():
        return

    print_new_questions(old_ids)

    print("\n" + "=" * 60)
    print("✅ Sync complete!")
    print("=" * 60)
    print("\n📊 Your questions are now:")
    print(f"   • Local: {LOCAL_CSV}")
    print(f"   • HuggingFace: https://huggingface.co/datasets/{HF_DATASET}")

    print("\n" + "=" * 60)
    print("Regenerating samples.json...")
    print("=" * 60)
    subprocess.run(["python3", "sample_questions.py"], check=True)

    print("\nNext time you update your Google Sheet, just run:")
    print("   python sync_questions_to_hf.py")


if __name__ == "__main__":
    main()
