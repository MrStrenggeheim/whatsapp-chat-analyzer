# %%
from analyze import analyze, plot_charts
from preprocess import preprocess
# auto reload
%load_ext autoreload
%autoreload 2

info = preprocess("./chats/p2p.txt")
info = analyze(info)
# plot_charts(info)
chat_name = info["chat_name"]
df = info["df"]

# %%
info


# TODO analysis on topics of messages and conversations