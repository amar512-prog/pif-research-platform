from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class RenderSegment:
    text: str
    url: str | None = None
    color: tuple[float, float, float] | None = None


@dataclass(slots=True)
class RenderLine:
    segments: list[RenderSegment]
    font: str
    size: int
    x: float
    height: float
    is_justified: bool = False

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)


@dataclass(slots=True, frozen=True)
class LinkAnnotation:
    x1: float
    y1: float
    x2: float
    y2: float
    url: str


class SimplePDFExporter:
    """Single-column PDF exporter with research-paper-inspired typography."""

    PAGE_WIDTH = 612
    PAGE_HEIGHT = 792
    MARGIN = 54
    LEFT_MARGIN = 54
    TOP_MARGIN = 748
    BOTTOM_MARGIN = 60
    CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)
    ABSTRACT_INDENT = 30

    BODY_FONT = "F1"
    HEADING_FONT = "F2"
    TABLE_FONT = "F3"

    BODY_SIZE = 10
    BODY_LEADING = 13
    SUBTITLE_SIZE = 10
    SECTION_SIZE = 10
    SUBSECTION_SIZE = 10
    TABLE_SIZE = 8
    TABLE_LEADING = 10
    LINK_COLOR = (0.08, 0.25, 0.70)
    SUMMARY_SECTION_TITLES = {"executive summary", "abstract"}
    REFERENCE_SECTION_TITLE = "references"

    def export_markdown(self, markdown_text: str, output_path: str | Path) -> str:
        output = Path(output_path)
        rendered_lines = self._layout_markdown(markdown_text)
        pages = self._paginate(rendered_lines)
        page_payloads = [self._content_stream(lines, page_number=index + 1) for index, lines in enumerate(pages)]

        page_object_numbers: list[int] = []
        content_object_numbers: list[int] = []
        next_object = 3
        for _ in page_payloads:
            page_object_numbers.append(next_object)
            content_object_numbers.append(next_object + 1)
            next_object += 2

        font_object_numbers = {
            self.BODY_FONT: next_object,
            self.HEADING_FONT: next_object + 1,
            self.TABLE_FONT: next_object + 2,
        }
        next_object += 3

        annotation_object_numbers: list[list[int]] = []
        for _, annotations in page_payloads:
            refs = list(range(next_object, next_object + len(annotations)))
            annotation_object_numbers.append(refs)
            next_object += len(annotations)

        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        page_kids = " ".join(f"{index} 0 R" for index in page_object_numbers)
        objects.append(f"<< /Type /Pages /Count {len(page_payloads)} /Kids [{page_kids}] >>".encode("utf-8"))

        for page_index, (stream, _) in enumerate(page_payloads):
            content_obj_num = content_object_numbers[page_index]
            font_resource = " ".join(f"/{name} {obj_num} 0 R" for name, obj_num in font_object_numbers.items())
            annots = annotation_object_numbers[page_index]
            annots_clause = f" /Annots [{' '.join(f'{annot} 0 R' for annot in annots)}]" if annots else ""
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.PAGE_WIDTH} {self.PAGE_HEIGHT}] "
                    f"/Resources << /Font << {font_resource} >> >> /Contents {content_obj_num} 0 R"
                    f"{annots_clause} >>"
                ).encode("utf-8")
            )
            objects.append(
                b"<< /Length "
                + str(len(stream)).encode("utf-8")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream"
            )

        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

        for page_index, (_, annotations) in enumerate(page_payloads):
            for annotation in annotations:
                objects.append(
                    (
                        f"<< /Type /Annot /Subtype /Link /Rect "
                        f"[{annotation.x1:.2f} {annotation.y1:.2f} {annotation.x2:.2f} {annotation.y2:.2f}] "
                        f"/Border [0 0 0] /A << /S /URI /URI ({self._escape_pdf_text(annotation.url)}) >> >>"
                    ).encode("utf-8")
                )

        pdf = self._assemble_pdf(objects)
        output.write_bytes(pdf)
        return str(output)

    def _layout_markdown(self, markdown_text: str) -> list[RenderLine]:
        lines = markdown_text.splitlines()
        rendered: list[RenderLine] = []
        index = 0
        current_section = ""
        saw_title = False

        while index < len(lines):
            raw_line = lines[index].rstrip()
            stripped = raw_line.strip()
            if not stripped:
                rendered.append(self._blank_line(6))
                index += 1
                continue

            if self._is_table_row(raw_line):
                table_lines: list[str] = []
                while index < len(lines) and self._is_table_row(lines[index]):
                    table_lines.append(lines[index].rstrip())
                    index += 1
                rendered.extend(self._render_table(table_lines))
                rendered.append(self._blank_line(8))
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = self._inline_markdown_to_text(heading_match.group(2))
                if level == 1 and not saw_title:
                    rendered.extend(self._render_title_block(heading_text))
                    saw_title = True
                    current_section = ""
                elif level == 2:
                    current_section = heading_text.lower()
                    if current_section in self.SUMMARY_SECTION_TITLES:
                        rendered.extend(self._render_summary_heading(heading_text))
                    else:
                        rendered.extend(self._render_section_heading(heading_text))
                else:
                    rendered.extend(self._render_subsection_heading(heading_text))
                index += 1
                continue

            bullet_match = re.match(r"^(\-|\*)\s+(.*)$", stripped)
            numbered_match = re.match(r"^(\d+\.)\s+(.*)$", stripped)
            if bullet_match or numbered_match:
                marker = numbered_match.group(1) if numbered_match else "-"
                text = numbered_match.group(2) if numbered_match else bullet_match.group(2)
                rendered.extend(self._render_list_item(marker, text, current_section))
                index += 1
                continue

            paragraph_lines = [stripped]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate:
                    break
                if re.match(r"^(#{1,6})\s+", candidate):
                    break
                if re.match(r"^(\-|\*)\s+", candidate):
                    break
                if re.match(r"^\d+\.\s+", candidate):
                    break
                if self._is_table_row(lines[index]):
                    break
                paragraph_lines.append(candidate)
                index += 1
            rendered.extend(self._render_paragraph(" ".join(paragraph_lines), current_section))

        return rendered or [self._plain_line("", font=self.BODY_FONT, size=self.BODY_SIZE, x=self.LEFT_MARGIN, height=12)]

    def _render_title_block(self, text: str) -> list[RenderLine]:
        wrapped = self._wrap_text(text, self._chars_for_width(self.CONTENT_WIDTH, self.HEADING_FONT, 18))
        rendered = [self._blank_line(2)]
        for line in wrapped:
            rendered.append(self._centered_line(line, font=self.HEADING_FONT, size=18, height=24))
        rendered.append(
            self._centered_line(
                "Technical Report / Policy Brief",
                font=self.BODY_FONT,
                size=self.SUBTITLE_SIZE,
                height=14,
            )
        )
        rendered.append(self._blank_line(12))
        return rendered

    def _render_summary_heading(self, text: str) -> list[RenderLine]:
        return [
            self._blank_line(4),
            self._plain_line(text.upper(), font=self.HEADING_FONT, size=self.SECTION_SIZE, x=self.LEFT_MARGIN, height=14),
            self._blank_line(4),
        ]

    def _render_section_heading(self, text: str) -> list[RenderLine]:
        return [
            self._blank_line(8),
            self._plain_line(text.upper(), font=self.HEADING_FONT, size=self.SECTION_SIZE, x=self.LEFT_MARGIN, height=16),
            self._blank_line(3),
        ]

    def _render_subsection_heading(self, text: str) -> list[RenderLine]:
        return [
            self._blank_line(6),
            self._plain_line(text, font=self.HEADING_FONT, size=self.SUBSECTION_SIZE, x=self.LEFT_MARGIN, height=14),
            self._blank_line(2),
        ]

    def _render_paragraph(self, text: str, current_section: str) -> list[RenderLine]:
        x = self.LEFT_MARGIN
        width_points = self.CONTENT_WIDTH
        size = self.BODY_SIZE
        height = self.BODY_LEADING
        justify = True

        if current_section in self.SUMMARY_SECTION_TITLES:
            x = self.LEFT_MARGIN + self.ABSTRACT_INDENT
            width_points = self.CONTENT_WIDTH - (2 * self.ABSTRACT_INDENT)
        elif current_section == self.REFERENCE_SECTION_TITLE:
            x = self.LEFT_MARGIN + 16
            width_points = self.CONTENT_WIDTH - 16
            size = 9
            height = 11
            justify = False

        segments = self._inline_markdown_to_segments(text)
        has_link = any(segment.url for segment in segments)
        line_capacity = self._chars_for_width(width_points, self.BODY_FONT, size)
        plain_text = "".join(segment.text for segment in segments)
        if has_link and len(self._wrap_text(plain_text, line_capacity)) == 1:
            return [
                RenderLine(
                    segments=segments,
                    font=self.BODY_FONT,
                    size=size,
                    x=x,
                    height=height,
                    is_justified=False,
                ),
                self._blank_line(4),
            ]

        wrapped = self._wrap_text(self._inline_markdown_to_text(text), line_capacity)
        rendered = [
            self._plain_line(line, font=self.BODY_FONT, size=size, x=x, height=height, is_justified=justify)
            for line in wrapped
        ]
        rendered.append(self._blank_line(4))
        return rendered

    def _render_list_item(self, marker: str, text: str, current_section: str) -> list[RenderLine]:
        base_x = self.LEFT_MARGIN + 10
        width_points = self.CONTENT_WIDTH - 10
        size = self.BODY_SIZE
        height = self.BODY_LEADING

        if current_section in self.SUMMARY_SECTION_TITLES:
            base_x = self.LEFT_MARGIN + self.ABSTRACT_INDENT
            width_points = self.CONTENT_WIDTH - (2 * self.ABSTRACT_INDENT)
        elif current_section == self.REFERENCE_SECTION_TITLE:
            base_x = self.LEFT_MARGIN + 16
            width_points = self.CONTENT_WIDTH - 16
            size = 9
            height = 11

        prefix = f"{marker} "
        continuation_indent = base_x + 16
        text_capacity = self._chars_for_width(width_points - 16, self.BODY_FONT, size)
        segments = self._inline_markdown_to_segments(text)
        plain_text = "".join(segment.text for segment in segments)

        if any(segment.url for segment in segments) and len(self._wrap_text(plain_text, text_capacity)) == 1:
            rendered = [
                RenderLine(
                    segments=[RenderSegment(prefix)] + segments,
                    font=self.BODY_FONT,
                    size=size,
                    x=base_x,
                    height=height,
                    is_justified=False,
                )
            ]
        else:
            wrapped = self._wrap_text(self._inline_markdown_to_text(text), text_capacity)
            rendered = []
            for line_index, line in enumerate(wrapped):
                if line_index == 0:
                    rendered.append(
                        self._plain_line(
                            f"{prefix}{line}",
                            font=self.BODY_FONT,
                            size=size,
                            x=base_x,
                            height=height,
                            is_justified=False,
                        )
                    )
                else:
                    rendered.append(
                        self._plain_line(
                            line,
                            font=self.BODY_FONT,
                            size=size,
                            x=continuation_indent,
                            height=height,
                            is_justified=False,
                        )
                    )
        rendered.append(self._blank_line(3))
        return rendered

    def _render_table(self, lines: list[str]) -> list[RenderLine]:
        rows = [self._table_cells(line) for line in lines if not self._is_table_separator(line)]
        if not rows:
            return []

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]
        widths = self._table_column_widths(normalized_rows)

        rendered: list[RenderLine] = [self._blank_line(4)]
        for row_index, row in enumerate(normalized_rows):
            wrapped_cells = [self._wrap_text(cell, widths[column_index]) for column_index, cell in enumerate(row)]
            row_line_count = max(len(cell_lines) for cell_lines in wrapped_cells)
            for line_index in range(row_line_count):
                padded_cells = []
                for column_index, cell_lines in enumerate(wrapped_cells):
                    value = cell_lines[line_index] if line_index < len(cell_lines) else ""
                    padded_cells.append(value.ljust(widths[column_index]))
                rendered.append(
                    self._plain_line(
                        "  ".join(padded_cells).rstrip(),
                        font=self.TABLE_FONT,
                        size=self.TABLE_SIZE,
                        x=self.LEFT_MARGIN,
                        height=self.TABLE_LEADING,
                        is_justified=False,
                    )
                )
            if row_index == 0:
                rendered.append(
                    self._plain_line(
                        "  ".join("-" * width for width in widths).rstrip(),
                        font=self.TABLE_FONT,
                        size=self.TABLE_SIZE,
                        x=self.LEFT_MARGIN,
                        height=self.TABLE_LEADING,
                        is_justified=False,
                    )
                )
        return rendered

    def _paginate(self, lines: list[RenderLine]) -> list[list[RenderLine]]:
        if not lines:
            return [[self._plain_line("", font=self.BODY_FONT, size=self.BODY_SIZE, x=self.LEFT_MARGIN, height=12)]]

        pages: list[list[RenderLine]] = []
        page: list[RenderLine] = []
        used_height = 0.0
        available_height = self.TOP_MARGIN - self.BOTTOM_MARGIN

        for line in lines:
            if page and used_height + max(line.height, 1.0) > available_height:
                pages.append(page)
                page = []
                used_height = 0.0
            page.append(line)
            used_height += max(line.height, 1.0)

        if page:
            pages.append(page)
        return pages

    def _content_stream(self, lines: list[RenderLine], page_number: int) -> tuple[bytes, list[LinkAnnotation]]:
        body = ["BT"]
        current_y = self.TOP_MARGIN
        current_font: str | None = None
        current_size: int | None = None
        current_color = (0.0, 0.0, 0.0)
        current_char_spacing = 0.0
        annotations: list[LinkAnnotation] = []

        for line in lines:
            current_y -= line.height
            if not line.text:
                continue

            if current_font != line.font or current_size != line.size:
                body.append(f"/{line.font} {line.size} Tf")
                current_font = line.font
                current_size = line.size

            target_spacing = 0.18 if line.is_justified and len(line.text) > 50 and not any(seg.url for seg in line.segments) else 0.0
            if target_spacing != current_char_spacing:
                body.append(f"{target_spacing:.2f} Tc")
                current_char_spacing = target_spacing

            current_x = line.x
            for segment in line.segments:
                if not segment.text:
                    continue
                color = segment.color or (0.0, 0.0, 0.0)
                if color != current_color:
                    body.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")
                    current_color = color
                body.append(f"1 0 0 1 {current_x:.2f} {current_y:.2f} Tm")
                body.append(f"({self._escape_pdf_text(segment.text)}) Tj")
                segment_width = self._text_width(segment.text, line.font, line.size)
                if segment.url:
                    annotations.append(
                        LinkAnnotation(
                            x1=current_x,
                            y1=current_y - 2,
                            x2=current_x + segment_width,
                            y2=current_y + line.size + 2,
                            url=segment.url,
                        )
                    )
                current_x += segment_width

        if current_color != (0.0, 0.0, 0.0):
            body.append("0 0 0 rg")
        if current_char_spacing != 0.0:
            body.append("0 Tc")
        body.append(f"/{self.BODY_FONT} 8 Tf")
        body.append(f"1 0 0 1 {self._center_x(str(page_number), self.BODY_FONT, 8):.2f} 28 Tm")
        body.append(f"({page_number}) Tj")
        body.append("ET")
        return "\n".join(body).encode("utf-8"), annotations

    def _assemble_pdf(self, objects: list[bytes]) -> bytes:
        chunks = [b"%PDF-1.4\n"]
        offsets = [0]
        current = len(chunks[0])
        for index, payload in enumerate(objects, start=1):
            offsets.append(current)
            obj = f"{index} 0 obj\n".encode("utf-8") + payload + b"\nendobj\n"
            chunks.append(obj)
            current += len(obj)
        xref_offset = current
        xref = [f"xref\n0 {len(objects) + 1}\n".encode("utf-8"), b"0000000000 65535 f \n"]
        for offset in offsets[1:]:
            xref.append(f"{offset:010d} 00000 n \n".encode("utf-8"))
        trailer = (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode(
                "utf-8"
            )
        )
        return b"".join(chunks + xref + [trailer])

    def _wrap_text(self, text: str, width: int) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return [""]
        words: list[str] = []
        for raw_word in cleaned.split(" "):
            if len(raw_word) <= width:
                words.append(raw_word)
            else:
                words.extend(self._split_long_word(raw_word, width))

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _inline_markdown_to_segments(self, text: str) -> list[RenderSegment]:
        segments: list[RenderSegment] = []
        cursor = 0
        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
            prefix = self._clean_inline_fragment(text[cursor:match.start()])
            if prefix:
                segments.append(RenderSegment(prefix))
            label = self._clean_inline_fragment(match.group(1))
            url = match.group(2).strip()
            if label:
                segments.append(RenderSegment(label, url=url, color=self.LINK_COLOR))
            cursor = match.end()
        suffix = self._clean_inline_fragment(text[cursor:])
        if suffix:
            segments.append(RenderSegment(suffix))
        return self._normalize_segment_boundaries(segments)

    def _inline_markdown_to_text(self, text: str) -> str:
        cleaned = text.replace("**", "").replace("__", "").replace("`", "")
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
        return cleaned.strip()

    def _is_table_row(self, line: str) -> bool:
        stripped = line.strip()
        return "|" in stripped and stripped.count("|") >= 2

    def _is_table_separator(self, line: str) -> bool:
        cells = self._table_cells(line)
        return bool(cells) and all(re.fullmatch(r"[:\- ]+", cell or "-") for cell in cells)

    def _table_cells(self, line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [self._inline_markdown_to_text(cell.strip()) for cell in stripped.split("|")]

    def _table_column_widths(self, rows: list[list[str]]) -> list[int]:
        raw_widths = [max(max(len(row[index]) for row in rows), 8) for index in range(len(rows[0]))]
        spacing_budget = max(0, (len(raw_widths) - 1) * 2)
        max_total_width = self._chars_for_width(self.CONTENT_WIDTH, self.TABLE_FONT, self.TABLE_SIZE) - spacing_budget
        minimums = [8 for _ in raw_widths]
        widths = [min(width, 28) for width in raw_widths]
        while sum(widths) > max_total_width:
            candidates = [index for index, width in enumerate(widths) if width > minimums[index]]
            if not candidates:
                break
            shrink_index = max(candidates, key=lambda index: widths[index])
            widths[shrink_index] -= 1
        return widths

    def _split_long_word(self, word: str, width: int) -> list[str]:
        if width <= 1:
            return [word]
        return [word[index : index + width] for index in range(0, len(word), width)]

    def _centered_line(self, text: str, font: str, size: int, height: float) -> RenderLine:
        x = self._center_x(text, font, size)
        return self._plain_line(text, font=font, size=size, x=x, height=height, is_justified=False)

    def _center_x(self, text: str, font: str, size: int) -> float:
        return max(self.LEFT_MARGIN, (self.PAGE_WIDTH - self._text_width(text, font, size)) / 2)

    def _plain_line(self, text: str, font: str, size: int, x: float, height: float, is_justified: bool = False) -> RenderLine:
        if not text:
            return RenderLine(segments=[], font=font, size=size, x=x, height=height, is_justified=is_justified)
        return RenderLine(
            segments=[RenderSegment(text)],
            font=font,
            size=size,
            x=x,
            height=height,
            is_justified=is_justified,
        )

    def _blank_line(self, height: float) -> RenderLine:
        return self._plain_line("", font=self.BODY_FONT, size=self.BODY_SIZE, x=self.LEFT_MARGIN, height=height)

    def _clean_inline_fragment(self, text: str) -> str:
        cleaned = text.replace("**", "").replace("__", "").replace("`", "")
        return re.sub(r"\s+", " ", cleaned)

    def _normalize_segment_boundaries(self, segments: list[RenderSegment]) -> list[RenderSegment]:
        normalized: list[RenderSegment] = []
        for segment in segments:
            if not segment.text:
                continue
            text = segment.text
            if normalized:
                previous = normalized[-1]
                if not previous.text.endswith(" ") and not text.startswith(" "):
                    text = f" {text}"
            normalized.append(RenderSegment(text=text, url=segment.url, color=segment.color))
        return normalized

    def _chars_for_width(self, width_points: float, font: str, size: int) -> int:
        factor = self._font_width_factor(font)
        return max(12, int(width_points / (size * factor)))

    def _font_width_factor(self, font: str) -> float:
        if font == self.HEADING_FONT:
            return 0.54
        if font == self.TABLE_FONT:
            return 0.60
        return 0.50

    def _text_width(self, text: str, font: str, size: int) -> float:
        return max(1, len(text)) * size * self._font_width_factor(font)

    def _escape_pdf_text(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
