from __future__ import annotations

from pathlib import Path

from pif_research_platform.exporters import SimplePDFExporter


def test_pdf_exporter_layout_removes_markdown_tokens() -> None:
    exporter = SimplePDFExporter()
    markdown = """# Main Title

## Section

- First bullet item
- Second bullet item

| Indicator | Latest | Prior |
| --- | --- | --- |
| Tax collections growth | 6.7 % YoY | 5.4 % YoY |
"""

    lines = exporter._layout_markdown(markdown)
    visible_text = [line.text for line in lines if line.text]

    assert "Main Title" in visible_text
    assert "SECTION" in visible_text
    assert all(not text.startswith("#") for text in visible_text)
    assert any(text.startswith("- First bullet item") for text in visible_text)
    assert not any("| --- | --- | --- |" in text for text in visible_text)
    assert any("Indicator" in text and "Latest" in text and "Prior" in text for text in visible_text)


def test_pdf_exporter_wraps_long_table_cells() -> None:
    exporter = SimplePDFExporter()
    markdown = """| Indicator | Explanation |
| --- | --- |
| Tax collections growth | Tax collections growth remains elevated because compliance improvements and formal demand recovery have both strengthened over the current reporting cycle. |
"""

    lines = exporter._layout_markdown(markdown)
    visible_text = [line.text for line in lines if line.text]

    assert any("Tax collections growth  Tax collections growth" in text for text in visible_text)
    assert any("formal demand recovery have" in text for text in visible_text)
    assert any("both strengthened over the" in text for text in visible_text)
    assert any("current reporting cycle." in text for text in visible_text)
    assert not any("..." in text for text in visible_text)


def test_pdf_exporter_writes_pdf_without_raw_markdown_symbols(tmp_path: Path) -> None:
    exporter = SimplePDFExporter()
    markdown = """# Heading

## Summary

- Item one

| Name | Value |
| --- | --- |
| Alpha | 42 |
"""

    output_path = tmp_path / "report.pdf"
    exporter.export_markdown(markdown, output_path)
    payload = output_path.read_bytes()

    assert output_path.exists()
    assert b"# Heading" not in payload
    assert b"| --- | --- |" not in payload
    assert b"(Heading) Tj" in payload
    assert b"(Alpha" in payload
    assert b"/BaseFont /Times-Roman" in payload
    assert b"/BaseFont /Times-Bold" in payload


def test_pdf_exporter_emits_clickable_reference_links(tmp_path: Path) -> None:
    exporter = SimplePDFExporter()
    markdown = """## References

- [S1] Example Author (2025). Example paper. Example Publisher.
Source link: [Open article](https://example.com/paper)
"""

    output_path = tmp_path / "references.pdf"
    exporter.export_markdown(markdown, output_path)
    payload = output_path.read_bytes()

    assert b"(Open article) Tj" in payload
    assert b"/URI (https://example.com/paper)" in payload
