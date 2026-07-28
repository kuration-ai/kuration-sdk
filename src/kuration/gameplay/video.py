"""Notebook-friendly, dependency-light gameplay video comparison."""
from __future__ import annotations

import base64
from dataclasses import asdict
from hashlib import sha256
from html import escape
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np

from ..core.io import write_text


def _is_local(source: str) -> bool:
    parsed = urlparse(source)
    return not parsed.scheme or (len(parsed.scheme) == 1 and source[1:2] == ":")


def _validated_source(source: str | Path) -> str:
    value = str(source)
    if not value:
        raise ValueError("video source must be a non-empty path or URL")
    if _is_local(value) and not Path(value).is_file():
        raise ValueError(f"video source does not exist: {value}")
    return value


def _label(source: str) -> str:
    path = urlparse(source).path
    return Path(path).name or source


def _mime(source: str) -> str:
    return mimetypes.guess_type(urlparse(source).path)[0] or "video/mp4"


def _frame_caption(ref, *, timed: bool) -> str:
    """Label one contact-sheet frame in the unit its reference value actually carries.

    Without a duration or an average rate a provider has no timebase, so the reference value
    falls back to the frame index; rendering that as seconds would misreport it.
    """
    return f"{ref.timestamp:.2f}s" if timed else f"frame {ref.frame_index}"


class VideoComparison:
    """A side-by-side gameplay video review with synchronized playback."""

    def __init__(
        self,
        left: str | Path,
        right: str | Path,
        *,
        labels: tuple[str, str] | None = None,
        provider=None,
    ) -> None:
        self.left = _validated_source(left)
        self.right = _validated_source(right)
        if labels is None:
            labels = (_label(self.left), _label(self.right))
        if (
            not isinstance(labels, (tuple, list))
            or len(labels) != 2
            or any(not isinstance(value, str) or not value for value in labels)
        ):
            raise ValueError("labels must be a pair of non-empty strings")
        self.labels = (labels[0], labels[1])
        self._provider = provider

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "labels": list(self.labels),
        }

    def _source_url(
        self,
        source: str,
        *,
        embed: bool | str,
        max_embed_bytes: int,
    ) -> tuple[str, bool]:
        if not _is_local(source):
            return source, False
        path = Path(source).resolve()
        size = path.stat().st_size
        should_embed = embed is True or (embed == "auto" and size <= max_embed_bytes)
        if should_embed:
            if size > max_embed_bytes:
                raise ValueError(
                    f"{path.name} is {size} bytes, above max_embed_bytes={max_embed_bytes}"
                )
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{_mime(source)};base64,{encoded}", True
        return path.as_uri(), False

    def render(
        self,
        *,
        embed: bool | str = "auto",
        max_embed_bytes: int = 12 * 1024 * 1024,
        full_document: bool = False,
        title: str = "Gameplay video comparison",
    ) -> str:
        """Render synchronized video players as safe standalone or notebook HTML.

        ``embed="auto"`` makes small local videos notebook-portable while avoiding accidental
        expansion of large recordings. ``embed=True`` is explicit and still bounded by
        ``max_embed_bytes``.
        """
        if not isinstance(embed, bool) and embed != "auto":
            raise ValueError("embed must be true, false, or 'auto'")
        if (
            not isinstance(max_embed_bytes, int)
            or isinstance(max_embed_bytes, bool)
            or max_embed_bytes < 1
        ):
            raise ValueError("max_embed_bytes must be a positive integer")
        left, left_embedded = self._source_url(
            self.left, embed=embed, max_embed_bytes=max_embed_bytes
        )
        right, right_embedded = self._source_url(
            self.right, embed=embed, max_embed_bytes=max_embed_bytes
        )
        # A comparison may be displayed more than once in the same notebook document.
        # Every render needs a fresh DOM id or a later script binds to the first pair of players.
        identifier = "kur-video-" + sha256(
            f"{self.left}\0{self.right}\0{uuid4().hex}".encode()
        ).hexdigest()[:12]
        notices = []
        if _is_local(self.left) and not left_embedded:
            notices.append(f"{self.labels[0]} uses a local file URL")
        if _is_local(self.right) and not right_embedded:
            notices.append(f"{self.labels[1]} uses a local file URL")
        notice = (
            "<p class=\"kur-video-note\">"
            + escape("; ".join(notices))
            + ". Use <code>embed=True</code> for a portable report.</p>"
            if notices
            else ""
        )
        body = (
            f"<section id=\"{identifier}\" class=\"kur-video-compare\">"
            f"<h2>{escape(title)}</h2>"
            "<div class=\"kur-video-grid\">"
            f"<figure><figcaption>{escape(self.labels[0])}</figcaption>"
            f"<video controls preload=\"metadata\" src=\"{escape(left, quote=True)}\"></video>"
            "</figure>"
            f"<figure><figcaption>{escape(self.labels[1])}</figcaption>"
            f"<video controls preload=\"metadata\" src=\"{escape(right, quote=True)}\"></video>"
            "</figure></div>"
            "<label class=\"kur-video-sync\"><input type=\"checkbox\" checked> "
            "Synchronize playback and seeking</label>"
            f"{notice}</section>"
            "<script>(function(){"
            f"const root=document.getElementById('{identifier}');"
            "if(!root)return;const videos=[...root.querySelectorAll('video')];"
            "const enabled=root.querySelector('input');let active=false;"
            "const mirror=(source,event)=>{if(!enabled.checked||active)return;active=true;"
            "for(const video of videos){if(video===source)continue;"
            "if(event==='play')video.play().catch(()=>{});"
            "else if(event==='pause')video.pause();"
            "else video.currentTime=source.currentTime;}active=false;};"
            "for(const video of videos){video.addEventListener('play',()=>mirror(video,'play'));"
            "video.addEventListener('pause',()=>mirror(video,'pause'));"
            "video.addEventListener('seeked',()=>mirror(video,'seeked'));}})();</script>"
        )
        if not full_document:
            return (
                "<style>.kur-video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));"
                "gap:1rem}.kur-video-grid figure{margin:0}.kur-video-grid video{width:100%;"
                "background:#111;border-radius:.45rem}.kur-video-grid figcaption{font-weight:600;"
                "margin-bottom:.35rem}.kur-video-sync{display:block;margin-top:.7rem}"
                ".kur-video-note{opacity:.75;font-size:.9em}</style>"
                + body
            )
        style = (
            "body{max-width:1400px;margin:2rem auto;padding:0 1rem;font-family:system-ui,sans-serif}"
            ".kur-video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}"
            ".kur-video-grid figure{margin:0}.kur-video-grid video{width:100%;background:#111;"
            "border-radius:.45rem}.kur-video-grid figcaption{font-weight:600;margin-bottom:.35rem}"
            ".kur-video-sync{display:block;margin-top:.7rem}.kur-video-note{opacity:.75}"
            "@media(max-width:700px){.kur-video-grid{grid-template-columns:1fr}}"
        )
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{style}</style></head><body>{body}</body></html>"
        )

    def save(
        self,
        path: str | Path,
        *,
        embed: bool = False,
        max_embed_bytes: int = 64 * 1024 * 1024,
        title: str = "Gameplay video comparison",
    ) -> str:
        """Write a synchronized comparison page and return its path."""
        output = str(path)
        write_text(
            output,
            self.render(
                embed=embed,
                max_embed_bytes=max_embed_bytes,
                full_document=True,
                title=title,
            ),
        )
        return output

    def probe(self, *, provider=None) -> dict:
        """Return comparable video metadata, importing PyAV only when needed."""
        if provider is None:
            provider = self._provider
        if provider is None:
            from ..core.media import PyAVMediaProvider

            provider = PyAVMediaProvider()
        return {
            side: {
                "label": label,
                **asdict(provider.probe(source)),
            }
            for side, label, source in zip(
                ("left", "right"),
                self.labels,
                (self.left, self.right),
                strict=True,
            )
        }

    def plot(self, *, frames: int = 4, provider=None):
        """Return a Matplotlib contact sheet at matched relative positions."""
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
            raise ValueError("frames must be a positive integer")
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "video contact sheets need Matplotlib: pip install 'kuration-sdk[stats]'"
            ) from error
        if provider is None:
            provider = self._provider
        if provider is None:
            from ..core.media import PyAVMediaProvider

            provider = PyAVMediaProvider()

        rows = []
        captions = []
        for source in (self.left, self.right):
            metadata = provider.probe(source)
            count = metadata.frame_count
            # Without a duration or an average rate there is no timebase, so the reference
            # value below falls back to the frame index; a caption must then say "frame N"
            # rather than labelling an index as seconds.
            timed = True
            if count is not None and count > 0:
                from ..core.media import FrameRef

                indexes = np.linspace(
                    0, count - 1, num=min(frames, count), dtype=int
                )
                duration = metadata.duration
                timed = duration is not None or bool(metadata.average_rate)
                refs = [
                    FrameRef(
                        source,
                        source,
                        int(index),
                        (
                            float(duration) * index / max(1, count - 1)
                            if duration is not None
                            else index / metadata.average_rate
                            if metadata.average_rate
                            else float(index)
                        ),
                        stream_index=metadata.stream_index,
                    )
                    for index in indexes
                ]
            else:
                refs = list(provider.iter_refs(source, source_id=source))
            if not refs:
                raise ValueError(f"video {source!r} contains no frames")
            indexes = np.linspace(0, len(refs) - 1, num=min(frames, len(refs)), dtype=int)
            chosen = [refs[index] for index in indexes]
            rows.append(provider.decode_batch(chosen))
            captions.append([_frame_caption(ref, timed=timed) for ref in chosen])
        columns = min(len(rows[0]), len(rows[1]))
        figure, axes = plt.subplots(
            2,
            columns,
            figsize=(3.2 * columns, 6),
            squeeze=False,
            constrained_layout=True,
        )
        for row, label in enumerate(self.labels):
            for column in range(columns):
                axes[row, column].imshow(rows[row][column])
                axes[row, column].axis("off")
                axes[row, column].set_title(
                    f"{label}\n{captions[row][column]}"
                )
        return figure

    def _repr_html_(self) -> str:
        return self.render()


def compare_videos(
    left: str | Path,
    right: str | Path,
    *,
    labels: tuple[str, str] | None = None,
    provider=None,
) -> VideoComparison:
    """Create a synchronized gameplay video comparison for notebooks or HTML export."""
    return VideoComparison(left, right, labels=labels, provider=provider)
