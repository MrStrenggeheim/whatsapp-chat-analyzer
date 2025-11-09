# %%
import os

import polars as pl
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from svglib.svglib import svg2rlg

styles = getSampleStyleSheet()
styles.add(styles["Normal"].clone("Center", alignment=1))  # TA_CENTER


class PageGrid:
    """
    Flexible ReportLab layout helper.
    Supports text, SVG, PNG, and Polars DataFrames positioned
    on a fine-grained grid (n_rows x n_cols) with spanning via row/col ranges.
    """

    def __init__(
        self,
        c: canvas.Canvas,
        n_rows: int,
        n_cols: int,
        page_size=A4,
        margins=(0.5 * cm, 0.5 * cm, 0.5 * cm, 0.5 * cm),  # (left, right, top, bottom)
        padding=0.2 * cm,
        show_grid=False,
    ):
        self.c = c
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.page_width, self.page_height = page_size
        self.left_margin, self.right_margin, self.top_margin, self.bottom_margin = (
            margins
        )
        self.padding = padding
        self.show_grid = show_grid

        # Derived geometry
        self.cell_width = (
            self.page_width - self.left_margin - self.right_margin
        ) / n_cols
        self.cell_height = (
            self.page_height - self.top_margin - self.bottom_margin
        ) / n_rows

    # ------------------------------------------------------------------
    def _normalize_range(self, rng, max_val):
        if isinstance(rng, int):
            return rng, rng + 1
        if isinstance(rng, tuple) and len(rng) == 2:
            start, end = rng
            if end <= start:
                raise ValueError(f"Invalid range {rng}: end must be > start")
            return start, min(end, max_val)
        raise TypeError("Row/col must be int or (start, end) tuple")

    def _range_coordinates(self, row, col):
        row_start, row_end = self._normalize_range(row, self.n_rows)
        col_start, col_end = self._normalize_range(col, self.n_cols)

        x = self.left_margin + col_start * self.cell_width + self.padding
        y = (
            self.page_height
            - self.top_margin
            - row_end * self.cell_height
            + self.padding
        )
        width = (col_end - col_start) * self.cell_width - 2 * self.padding
        height = (row_end - row_start) * self.cell_height - 2 * self.padding
        return x, y, width, height

    def _aligned_position(
        self, cell_x, cell_y, cell_w, cell_h, obj_w, obj_h, halign, valign
    ):
        if halign == "center":
            x = cell_x + (cell_w - obj_w) / 2
        elif halign == "right":
            x = cell_x + (cell_w - obj_w)
        else:
            x = cell_x

        if valign == "middle":
            y = cell_y + (cell_h - obj_h) / 2
        elif valign == "top":
            y = cell_y + (cell_h - obj_h)
        else:
            y = cell_y
        return x, y

    # ------------------------------------------------------------------
    # === Content methods ===
    def get_frame(self, row, col):
        x, y, w, h = self._range_coordinates(row, col)
        frame = Frame(x, y, w, h, showBoundary=self.show_grid)
        return frame

    def add_text(self, text, row, col, style=None, halign="left", valign="top"):
        if style is None:
            style = self.styles["Normal"]
        frame = self.get_frame(row, col)
        if isinstance(text, str):
            p = Paragraph(text, style)
        elif isinstance(text, Paragraph):
            p = text
        else:
            raise TypeError("text must be str or Paragraph")
        frame.addFromList([p], self.c)
        return frame

    def add_svg(
        self, svg_path, row, col, preserve_aspect=True, halign="center", valign="middle"
    ):
        x, y, w, h = self._range_coordinates(row, col)
        drawing = svg2rlg(str(svg_path))
        if drawing is None:
            raise ValueError(f"Could not load SVG: {svg_path}")

        scale_x = w / drawing.width
        scale_y = h / drawing.height
        if preserve_aspect:
            scale = min(scale_x, scale_y)
            drawing.scale(scale, scale)
            dw, dh = drawing.width * scale, drawing.height * scale
        else:
            drawing.scale(scale_x, scale_y)
            dw, dh = w, h
        px, py = self._aligned_position(x, y, w, h, dw, dh, halign, valign)
        renderPDF.draw(drawing, self.c, px, py)
        if self.show_grid:
            self.c.rect(x, y, w, h, stroke=1, fill=0)

    def add_png(
        self, img_path, row, col, halign="center", valign="middle", preserve_aspect=True
    ):
        """Add a raster image (PNG/JPEG)."""
        x, y, w, h = self._range_coordinates(row, col)
        img = Image(str(img_path), useDPI=True)
        iw, ih = img.drawWidth, img.drawHeight

        scale_x = w / iw
        scale_y = h / ih
        if preserve_aspect:
            scale = min(scale_x, scale_y)
            iw, ih = iw * scale, ih * scale
        else:
            iw, ih = w, h
        px, py = self._aligned_position(x, y, w, h, iw, ih, halign, valign)
        img.drawWidth, img.drawHeight = iw, ih
        img.drawOn(self.c, px, py)
        if self.show_grid:
            self.c.rect(x, y, w, h, stroke=1, fill=0)

    def add_table(self, table, row, col, font_size=8, halign="center", valign="middle"):
        if isinstance(table, pl.DataFrame):
            data = [table.columns] + table.rows()
            table = Table(data, repeatRows=0)
        x, y, w, h = self._range_coordinates(row, col)
        # table.setStyle(
        #     TableStyle(
        #         [
        #             ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        #             ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        #             ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        #             ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        #             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        #         ]
        #     )
        # )
        tw, th = table.wrap(w, h)
        if tw > w or th > h:
            scale = min(w / tw, h / th)
            tw, th = tw * scale, th * scale
            self.c.saveState()
            self.c.scale(scale, scale)
            px, py = self._aligned_position(
                x / scale,
                y / scale,
                w / scale,
                h / scale,
                tw / scale,
                th / scale,
                halign,
                valign,
            )
            table.drawOn(self.c, px, py)
            self.c.restoreState()
        else:
            px, py = self._aligned_position(x, y, w, h, tw, th, halign, valign)
            table.drawOn(self.c, px, py)
        if self.show_grid:
            self.c.rect(x, y, w, h, stroke=1, fill=0)

    # ------------------------------------------------------------------
    def render(self, new_page=True):
        if new_page:
            self.c.showPage()


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
def build_report(
    chat_file_path: str, top_n_authors: int = 10, output_file: str = None
) -> None:
    """
    Build a PDF report for a WhatsApp chat.

    Args:
        chat_file_path: Path to the WhatsApp chat export file.
        top_n_authors: Number of top authors to show individually (default: 10).
        output_file: Path to the output PDF file (default: reports/{chat_name}_report.pdf).
    """
    from analyze import analyze, plot_charts
    from preprocess import preprocess

    info = preprocess(chat_file_path)
    info = analyze(info, top_n_authors=top_n_authors)
    plot_charts(info)
    chat_name = info["chat_name"]
    if not info["is_group"]:
        title_str = " & ".join(info["author_unique"])
    else:
        title_str = chat_name

    # Determine output path
    if output_file is None:
        os.makedirs("reports", exist_ok=True)
        output_file = f"reports/{chat_name}_report.pdf"
    else:
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    c = canvas.Canvas(output_file, pagesize=A4)
    c.setTitle(f"{chat_name} - Chat Report")
    grid = PageGrid(c, n_rows=12, n_cols=12, show_grid=False)

    # --- Title (rows 0–2, all columns)
    title_frame = grid.get_frame(row=(0, 2), col=(0, 12))
    # print(styles.list())

    datespan = f"{info['min_date'].strftime('%d %b %Y')} - {info['max_date'].strftime('%d %b %Y')}"

    if info["is_group"]:
        subtitle = f"Participants: {len(info['author_unique'])} | Total messages: {info['total_messages']} | Total days: {info['total_days']} | Total length: {info['total_length']} chars"
    else:
        subtitle = f"Total messages: {info['total_messages']} | Total days: {info['total_days']} | Total length: {info['total_length']} chars"

    title_frame.addFromList(
        [
            Paragraph(title_str, styles["Title"]),
            Paragraph(subtitle, styles["Center"]),
            # vspace
            Spacer(1, 5),
            Paragraph(datespan, styles["Center"]),
        ],
        c,
    )

    # --- Left: pie chart, Right: bar chart
    grid.add_png(f"plots/{chat_name}_message_share_pie.png", row=(1, 5), col=(0, 6))
    grid.add_png(
        f"plots/{chat_name}_avg_message_length_bar.png", row=(1, 5), col=(6, 12)
    )

    # left emoji bar plot, right table
    grid.add_png(f"plots/{chat_name}_common_emojis_bar.png", row=(5, 8), col=(0, 7))

    # --- Table with random data (placeholder)
    media_counts = sorted(
        info["media_counts"].items(), key=lambda x: x[1], reverse=True
    )
    data = [
        ["Metric", "", "Value"],
    ]
    # in order of count descending
    num_media_types = len(media_counts)
    if num_media_types > 0:
        for media_type, count in media_counts:
            data.append(["Media", f"{media_type}", f"{count}"])
    else:
        data.append(["Media", "All Types", "0"])
        num_media_types = 1

    table = Table(data, repeatRows=0)
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (1, 0)),
                ("SPAN", (0, 1), (0, num_media_types)),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEAFTER", (1, 0), (1, -1), 1, colors.black),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    grid.add_table(table, row=(5, 8), col=(7, 12), font_size=8)

    # -- hourly plot (full width)
    grid.add_png(
        f"plots/{chat_name}_hourly_message_distribution.png", row=(8, 10), col=(0, 7)
    )

    # --- Bottom: PNG heatmap (full width)
    grid.add_png(
        f"plots/{chat_name}_messages_per_day_heatmap.png", row=(10, 12), col=(0, 12)
    )

    grid.render(new_page=False)
    c.save()
    print(f"✅ Report generated successfully: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a PDF report for a WhatsApp chat export."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the WhatsApp chat export file (e.g., ./chats/c.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output PDF file (default: reports/{chat_name}_report.pdf)",
    )
    parser.add_argument(
        "--top_n_authors",
        type=int,
        default=7,
        help="Number of top authors to show individually (default: 10)",
    )
    args = parser.parse_args()

    build_report(args.file, top_n_authors=args.top_n_authors, output_file=args.output)
