import argparse
import datetime
import pickle as pkl
from collections import Counter

import polars as pl
import regex as re


def preprocess(input_file, include_metaAI=False):
    with open(input_file, "r", encoding="utf-8") as f:
        chat_str = f.read()

    info = {}

    # Count media types and call time
    media_counts = Counter()
    call_time = Counter()
    media_pattern = r"^\u200e(.*?)\u200e(\w+)\s\w+$"
    call_pattern = r": \u200e(\w+)\. \u200e\u200e(\d+)\s(\w+)\.\s\u2022.*$"

    def replace_media(match):
        key = match.group(2)
        media_counts[key] += 1
        return f"{match.group(1)}<{key}>"

    def replace_call(match):
        key = match.group(1)
        call_time[match.group(3)] += int(match.group(2))
        media_counts[key] += 1
        return f": <{key}>"

    chat_str = re.sub(media_pattern, replace_media, chat_str, flags=re.MULTILINE)
    chat_str = re.sub(call_pattern, replace_call, chat_str, flags=re.MULTILINE)

    total_time_called = int(
        datetime.timedelta(
            hours=call_time.get("Std", 0),
            minutes=call_time.get("Min", 0),
            seconds=call_time.get("Sek", 0),
        ).total_seconds()
    )
    H = total_time_called // 3600
    M = (total_time_called % 3600) // 60

    # remove LTR/RTL markers
    chat_str = re.sub(r"[\u2066-\u2069]", "", chat_str)
    chat_str = re.sub(r"[\p{Cf}]", "", chat_str)

    # [dd.mm.yy, hh:mm:ss] author: message
    message_mask = (
        r"\[(\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2})\] (.*?): ?(.*?)(?=\n\[|$)"
    )
    messages = re.findall(message_mask, chat_str, re.DOTALL)

    df = (
        pl.DataFrame(messages, schema=["datetime", "author", "message"], orient="row")
        .with_columns(
            datetime=pl.col("datetime").str.strptime(pl.Datetime, "%d.%m.%y, %H:%M:%S"),
            message_length=pl.col("message").str.len_chars(),
        )
        .sort("datetime")
    )

    # filter out: First author = chatname, Meta AI messages if flag is not set
    # either group-> multiple authors + metaai, first message is group name
    # or one-on-one chat -> 2 authors + metaai, first author is other person name

    chat_name = df["author"].to_list()[0]
    author_unique = set(df["author"].unique().to_list()) - {"Meta AI"}

    # if authors without meta ai >2 -> group chat
    if len(author_unique) > 2:
        is_group = True
    else:
        is_group = False

    author_filter = []
    if is_group:
        author_filter.append(chat_name)
    if not include_metaAI:
        author_filter.append("Meta AI")
    df = df.filter(~pl.col("author").is_in(author_filter))

    info["chat_name"] = chat_name
    info["is_group"] = is_group
    info["df"] = df
    info["media_counts"] = media_counts
    info["total_call_time"] = {"h": H, "m": M}

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chat data")
    parser.add_argument("input_file", help="Path to the input chat file")
    parser.add_argument(
        "--include_metaAI", action="store_true", help="Include messages from Meta AI"
    )
    args = parser.parse_args()

    info = preprocess(args.input_file, args.include_metaAI)

    # export to pickle
    pkl.dump(
        info, open(f"{args.input_file.replace('.txt', '')}_preprocessed.pkl", "wb")
    )
