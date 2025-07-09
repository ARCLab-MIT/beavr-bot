#!/usr/bin/env python3
"""
tests/scripts/extract_actions.py

Fetches a slice of rows from your Hugging Face dataset (train split),
automatically paging in chunks of 100 to respect the API limit,
extracts the “action” vectors, and writes them to actions.json.
"""

import requests
import json
import sys
import time
import argparse


BASE_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=aposadasn%2Flx7r_pickup_test3_dataset"
    "&config=default"
    "&split=train"
)
OUTPUT_FILE = "./tests/scripts/actions_test3.json"
PAGE_SIZE = 100
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds

def fetch_with_retry(url):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url)
        if resp.status_code == 429:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            print(f"429 Too Many Requests — retry {attempt}/{MAX_RETRIES} in {backoff}s...",
                  file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded maximum retries")

def extract_actions(rows):
    return [
        row["row"]["action"][3:6]
        for row in rows
        if "row" in row and "action" in row["row"]
    ]

def main():
    p = argparse.ArgumentParser(
        description="Download a slice of actions from the HF dataset"
    )
    p.add_argument(
        "--offset", "-o", type=int, default=0,
        help="Row index to start from (default: 0)"
    )
    p.add_argument(
        "--length", "-n", type=int, default=100,
        help="Total number of rows to fetch (default: 100)"
    )
    args = p.parse_args()

    start = args.offset
    end = args.offset + args.length
    print(f"Requesting rows {start}–{end-1} (in pages of {PAGE_SIZE})…")

    all_actions = []
    for page_start in range(start, end, PAGE_SIZE):
        page_length = min(PAGE_SIZE, end - page_start)
        url = f"{BASE_URL}&offset={page_start}&length={page_length}"
        try:
            data = fetch_with_retry(url)
        except Exception as e:
            print(f"\nError fetching rows {page_start}–{page_start+page_length-1}: {e}", file=sys.stderr)
            sys.exit(1)

        rows = data.get("rows", [])
        actions = extract_actions(rows)
        all_actions.extend(actions)

        print(f"  → got {len(actions)} rows ({page_start}–{page_start+len(actions)-1})", end="\r")

    if not all_actions:
        print("No action vectors found!", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(all_actions, fp, indent=2)

    print(f"\nSaved {len(all_actions)} action vectors (rows {start}–{end-1}) to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
