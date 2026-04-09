import struct
import io
import torch
import numpy as np


class ArtifactPacker:
    MAGIC = b"GOLF"
    VERSION = 1

    @staticmethod
    def pack(state_dict: dict, metadata: dict | None = None) -> bytes:
        buf = io.BytesIO()
        buf.write(ArtifactPacker.MAGIC)
        buf.write(struct.pack("<B", ArtifactPacker.VERSION))
        entries = []
        for name, tensor in state_dict.items():
            t = tensor.detach().cpu()
            if t.is_floating_point():
                data = t.to(torch.float16).numpy().tobytes()
                dtype_code = 1
            else:
                data = t.numpy().tobytes()
                dtype_code = 2
            entries.append((name.encode(), dtype_code, list(t.shape), data))
        buf.write(struct.pack("<I", len(entries)))
        for name_bytes, dtype_code, shape, data in entries:
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<B", dtype_code))
            buf.write(struct.pack("<B", len(shape)))
            for s in shape:
                buf.write(struct.pack("<I", s))
            buf.write(struct.pack("<I", len(data)))
            buf.write(data)
        return buf.getvalue()

    @staticmethod
    def unpack(data: bytes) -> dict:
        buf = io.BytesIO(data)
        assert buf.read(4) == ArtifactPacker.MAGIC
        version = struct.unpack("<B", buf.read(1))[0]
        n = struct.unpack("<I", buf.read(4))[0]
        state_dict = {}
        for _ in range(n):
            name_len = struct.unpack("<H", buf.read(2))[0]
            name = buf.read(name_len).decode()
            dtype_code = struct.unpack("<B", buf.read(1))[0]
            ndim = struct.unpack("<B", buf.read(1))[0]
            shape = [struct.unpack("<I", buf.read(4))[0] for _ in range(ndim)]
            data_len = struct.unpack("<I", buf.read(4))[0]
            raw = buf.read(data_len)
            if dtype_code == 1:
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                state_dict[name] = torch.from_numpy(arr.copy()).float()
            else:
                arr = np.frombuffer(raw, dtype=np.int64).reshape(shape)
                state_dict[name] = torch.from_numpy(arr.copy())
        return state_dict
