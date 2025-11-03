import argparse
import pickle as pkl

import polars as pl
import regex as re


def preprocess(input_file, include_metaAI=False):
    with open(input_file, "r", encoding="utf-8") as f:
        chat_str = f.read()
    chat_str = re.sub(r"[\p{Cf}]", "", chat_str)
    media_types = [
        "Video",
        "Audio",
        "Bild",
        "Sticker",
        "GIF",
        "Dokument",
    ]
    # make "medium weggelassen" to "<medium>" if medium in medium list
    for m in media_types:
        chat_str = re.sub(rf"\b{m} weggelassen\b", f"<{m}>", chat_str)

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
    chat_name = df["author"].to_list()[0]
    author_filter = [chat_name]
    if not include_metaAI:
        author_filter.append("Meta AI")
    df = df.filter(~pl.col("author").is_in(author_filter))

    info = {"chat_name": chat_name, "df": df}

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
