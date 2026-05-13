"""Create a tiny valid PDF (single page, Helvetica) without extra dependencies."""


def escape_pdf_literal(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_minimal_pdf_bytes(body_text: str) -> bytes:
    inner = (
        f"BT /F1 14 Tf 72 720 Td ({escape_pdf_literal(body_text)}) Tj ET\n"
    ).encode("latin-1")
    obj4 = (
        b"<< /Length "
        + str(len(inner)).encode("ascii")
        + b" >>\nstream\n"
        + inner
        + b"\nendstream"
    )
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        4: obj4,
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets: dict[int, int] = {}
    for i in range(1, 6):
        offsets[i] = sum(len(p) for p in parts)
        parts.append(f"{i} 0 obj\n".encode("ascii"))
        parts.append(objs[i])
        parts.append(b"\nendobj\n")
    body = b"".join(parts)
    xref_start = len(body)
    xref_lines = [b"xref\n0 6\n0000000000 65535 f \n"]
    for i in range(1, 6):
        xref_lines.append(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))
    xref = b"".join(xref_lines)
    trailer = (
        b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF\n"
    )
    return body + xref + trailer
