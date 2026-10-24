"""Clean / Dirty video index: search, retrieve and play videos by their index label, see when dirty places appear,
and index a new video.  Run:  python app.py   (needs outputs/index from code.ipynb, section 11)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import vi_data as D
import vi_train as TR
import vi_eval as E

ROOT = Path(__file__).resolve().parent
IDX = ROOT / "outputs" / "index"
MODELS = ["GRU", "LSTM", "RNN"]
FPS, T, W = 2.0, 16, 4                     # video model: 16 frames at 2 fps; timestamp model: 2-s windows


class VideoIndex:
    def __init__(self):
        self.df = pd.read_csv(IDX / "video_index.csv")
        self.timelines = json.loads((IDX / "timelines.json").read_text())
        self._models = {}

    # ------------------------------------------------------------------ search
    def search(self, category="Dirty", model="GRU", min_conf=0.5, query=""):
        """Videos whose index (by `model`) equals `category` ('All' = every video), most confident first."""
        df = self.df.copy()
        df["confidence"] = np.where(df[f"index_{model}"] == "Dirty", df[f"p_dirty_{model}"], 1 - df[f"p_dirty_{model}"])
        if category != "All":
            df = df[df[f"index_{model}"] == category]
        df = df[df.confidence >= min_conf]
        if query:
            df = df[df.name.str.contains(query, case=False) | df.scene.str.contains(query, case=False)]
        return df.sort_values("confidence", ascending=False)

    def segments(self, vid, model):
        return json.loads(self.df.set_index("vid").loc[vid, f"segments_{model}"])

    # ------------------------------------------------------------------ new videos
    def models(self, cell):
        if cell not in self._models:
            self._models[cell] = (TR.load_model(f"main_{cell}_s0")[0], TR.load_model(f"win_{cell}_s0")[0])
        return self._models[cell]

    @torch.no_grad()
    def index_video(self, path, cell="GRU"):
        """Decodes a new video at 2 fps, extracts ResNet-50 features and returns (P(dirty), label, times, timeline, segments)."""
        import cv2
        c = cv2.VideoCapture(str(path))
        fps = c.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        times, idx = D.sample_indices(n, fps, FPS)
        want, frames = set(idx.tolist()), []
        for i in range(int(idx.max()) + 1):
            ok, fr = c.read() if i in want else (c.grab(), None)
            if not ok:
                break
            if fr is not None:
                frames.append(cv2.cvtColor(D.resize_short(fr), cv2.COLOR_BGR2RGB))
        c.release()
        x = torch.from_numpy(D.frame_features(frames)).cuda()
        vid_model, win_model = self.models(cell)
        clips = [x[s] for s in TR.eval_clips(len(x), D.BASE_FPS, T)]   # x is already at 2 fps -> step 1
        xb, L = TR.pad(clips)
        p = float(vid_model(xb, L).softmax(-1)[:, 1].mean())
        tl = E.timeline(win_model, x, W)
        t = times[:len(x)]
        return p, D.CLASSES[int(p >= 0.5)], t, tl, E.segments(tl >= 0.5, t)


def timeline_figure(t, p, segs, title):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.fill_between(t, p, step="mid", alpha=0.35, color="#c0392b")
    ax.plot(t, p, color="#c0392b", lw=1.2, drawstyle="steps-mid")
    for s, e in segs:
        ax.axvspan(s, e, color="#c0392b", alpha=0.12)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set(ylim=(0, 1), xlabel="time (s)", ylabel="P(dirty)", title=title)
    fig.tight_layout()
    return fig


def fmt_segments(segs):
    return ", ".join(f"{int(s // 60)}:{s % 60:04.1f}–{int(e // 60)}:{e % 60:04.1f}" for s, e in segs) or "none"


def build_app():
    import gradio as gr
    index = VideoIndex()
    thumbs = ROOT / "outputs" / "index" / "thumbs"

    def do_search(category, model, min_conf, query):
        df = index.search(category, model, min_conf, query)
        gal = [(str(thumbs / f"{v}.jpg"), f"{v} · {lab} ({c:.2f})") for v, lab, c in zip(df.vid, df[f"index_{model}"], df.confidence)]
        table = df[["vid", "name", "scene", "duration_s", f"index_{model}", "confidence", "true_class"]].round(3)
        return gal, table, gr.update(choices=list(df.vid), value=df.vid.iloc[0] if len(df) else None)

    def do_show(vid, model):
        if not vid:
            return None, None, ""
        r = index.df.set_index("vid").loc[vid]
        tl = index.timelines[model][vid]
        segs = index.segments(vid, model)
        fig = timeline_figure(np.array(tl["t"]), np.array(tl["p"]), segs, f"{vid}: index = {r[f'index_{model}']}")
        info = (f"**{r['name']}**  \nindex ({model}): **{r[f'index_{model}']}**, P(dirty) = {r[f'p_dirty_{model}']:.3f}  \n"
                f"dirty segments: {fmt_segments(segs)}  \nground truth: {r['true_class']}")
        return str(ROOT / "outputs" / "previews" / f"{vid}.mp4"), fig, info

    def do_index(path, model):
        if not path:
            return None, ""
        p, lab, t, tl, segs = index.index_video(path, model)
        return timeline_figure(t, tl, segs, f"new video: {lab} (P(dirty) = {p:.2f})"), f"index: **{lab}**  \ndirty segments: {fmt_segments(segs)}"

    with gr.Blocks(title="Clean / Dirty video index") as demo:
        gr.Markdown("## Clean / Dirty video index\nSearch the indexed videos by label, play one, and see when dirty places appear.")
        with gr.Tab("Search"):
            with gr.Row():
                cat = gr.Radio(["Dirty", "Clean", "All"], value="Dirty", label="category")
                mdl = gr.Radio(MODELS, value="GRU", label="indexing model")
                conf = gr.Slider(0.5, 1.0, 0.5, step=0.05, label="min. confidence")
                q = gr.Textbox(label="text filter (file name / scene)")
            btn = gr.Button("Search", variant="primary")
            gal = gr.Gallery(columns=6, height=260, label="results")
            tab = gr.Dataframe(label="results", interactive=False)
            pick = gr.Dropdown(label="open video", choices=[])
            with gr.Row():
                vid = gr.Video(label="video", height=320)
                with gr.Column():
                    plot = gr.Plot(label="dirty timeline")
                    info = gr.Markdown()
            btn.click(do_search, [cat, mdl, conf, q], [gal, tab, pick])
            pick.change(do_show, [pick, mdl], [vid, plot, info])
        with gr.Tab("Index a new video"):
            up = gr.Video(label="upload a video")
            mdl2 = gr.Radio(MODELS, value="GRU", label="indexing model")
            go = gr.Button("Index", variant="primary")
            plot2, info2 = gr.Plot(), gr.Markdown()
            go.click(do_index, [up, mdl2], [plot2, info2])
    return demo


if __name__ == "__main__":
    plt.switch_backend("Agg")
    build_app().launch(allowed_paths=[str(ROOT / "outputs")])
