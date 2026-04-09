import lzma
from collections import Counter
from heapq import heappush, heappop
from struct import pack, unpack


class Compressor:
    """Compress quantized weight bytes."""

    @staticmethod
    def compress(data: bytes, method: str) -> bytes:
        if method == "brotli":
            import brotli
            return brotli.compress(data, quality=11)
        elif method == "zstd22":
            import zstandard as zstd
            return zstd.ZstdCompressor(level=22).compress(data)
        elif method == "lzma9":
            return lzma.compress(data, preset=9)
        elif method == "ans_huffman":
            return _huffman_encode(data)
        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def decompress(data: bytes, method: str) -> bytes:
        if method == "brotli":
            import brotli
            return brotli.decompress(data)
        elif method == "zstd22":
            import zstandard as zstd
            return zstd.ZstdDecompressor().decompress(data)
        elif method == "lzma9":
            return lzma.decompress(data)
        elif method == "ans_huffman":
            return _huffman_decode(data)
        raise ValueError(f"Unknown method: {method}")


# --- Minimal Huffman coder ---

def _build_tree(freqs):
    """Build Huffman tree from byte frequencies. Returns root node."""
    heap = []
    for byte, count in freqs.items():
        heappush(heap, (count, id(byte), byte))  # id() for tiebreaker
    if len(heap) == 1:
        c, _, sym = heappop(heap)
        return (c, None, (sym, None))
    while len(heap) > 1:
        a = heappop(heap)
        b = heappop(heap)
        heappush(heap, (a[0] + b[0], id(a), (a, b)))
    return heappop(heap)


def _build_codes(node, prefix=0, length=0, codes=None):
    """Extract code table from Huffman tree."""
    if codes is None:
        codes = {}
    _, _, children = node
    if isinstance(children, int):  # leaf
        codes[children] = (prefix, max(length, 1))
        return codes
    left, right = children
    _build_codes(left, prefix << 1, length + 1, codes)
    _build_codes(right, (prefix << 1) | 1, length + 1, codes)
    return codes


def _huffman_encode(data: bytes) -> bytes:
    """Encode bytes with Huffman coding. Format: [4B orig_len][2B num_symbols][symbol table][bitstream]."""
    if not data:
        return pack(">I", 0)
    freqs = Counter(data)
    tree = _build_tree(freqs)
    codes = _build_codes(tree)

    # Header: original length + code table
    out = bytearray(pack(">I", len(data)))
    out += pack(">H", len(codes))
    for sym, (code, nbits) in sorted(codes.items()):
        out += pack(">BBH", sym, nbits, code)  # max 16-bit codes sufficient for 256 symbols

    # Bitstream
    bits = 0
    nbits = 0
    buf = bytearray()
    for byte in data:
        code, clen = codes[byte]
        bits = (bits << clen) | code
        nbits += clen
        while nbits >= 8:
            nbits -= 8
            buf.append((bits >> nbits) & 0xFF)
    if nbits > 0:
        buf.append((bits << (8 - nbits)) & 0xFF)
    out += pack(">B", nbits % 8 if nbits % 8 else 0)  # trailing bits count
    out += buf
    return bytes(out)


def _huffman_decode(data: bytes) -> bytes:
    """Decode Huffman-encoded bytes."""
    if len(data) < 4:
        return b""
    orig_len = unpack(">I", data[:4])[0]
    if orig_len == 0:
        return b""
    num_symbols = unpack(">H", data[4:6])[0]

    # Rebuild code table
    pos = 6
    decode_map = {}
    for _ in range(num_symbols):
        sym, nbits, code = unpack(">BBH", data[pos:pos + 4])
        pos += 4
        decode_map[(code, nbits)] = sym

    trail_bits = unpack(">B", data[pos:pos + 1])[0]
    pos += 1
    bitstream = data[pos:]

    # Decode bitstream
    out = bytearray()
    bits = 0
    nbits = 0
    total_bits = len(bitstream) * 8 - (8 - trail_bits if trail_bits else 0)
    bit_idx = 0

    for byte in bitstream:
        for shift in range(7, -1, -1):
            if bit_idx >= total_bits:
                break
            bits = (bits << 1) | ((byte >> shift) & 1)
            nbits += 1
            bit_idx += 1
            key = (bits, nbits)
            if key in decode_map:
                out.append(decode_map[key])
                bits = 0
                nbits = 0
                if len(out) >= orig_len:
                    return bytes(out)
    return bytes(out)
