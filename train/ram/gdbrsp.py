#!/usr/bin/env python3
"""Minimal GDB Remote Serial Protocol client for mGBA's `-g` stub.

mGBA 0.10 exposes a GDB stub on TCP :2345 (`mgba-qt -g`). We don't need a real
gdb client -- the RSP `m<addr>,<len>` (read memory) command is architecture-
agnostic, so a tiny socket speaker is enough to peek GBA RAM headless.

This is a TRAINING+EVAL crutch only (F21). Runtime perception stays pixels-only.

GBA memory map (the regions game state lives in):
  EWRAM  0x02000000 - 0x0203FFFF  (256 KB, work RAM)
  IWRAM  0x03000000 - 0x03007FFF  (32 KB, fast RAM)
  IORAM  0x04000000 - 0x040003FF
"""
import socket
import time


class GdbRsp:
    def __init__(self, host="127.0.0.1", port=2345, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.buf = b""

    def connect(self, retries=40, delay=0.25):
        last = None
        for _ in range(retries):
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.settimeout(self.timeout)
                self.sock = s
                # Halt the target so memory is stable, then drain its stop reply.
                self.interrupt()
                return True
            except OSError as e:
                last = e
                time.sleep(delay)
        raise RuntimeError(f"could not connect to gdb stub at {self.host}:{self.port}: {last}")

    # --- low level packet I/O ---
    @staticmethod
    def _checksum(data: bytes) -> int:
        return sum(data) & 0xFF

    def _send_packet(self, payload: str):
        data = payload.encode("ascii")
        pkt = b"$" + data + b"#" + f"{self._checksum(data):02x}".encode("ascii")
        self.sock.sendall(pkt)
        self._wait_ack()

    def _recv_more(self):
        chunk = self.sock.recv(4096)
        if not chunk:
            raise RuntimeError("gdb stub closed connection")
        self.buf += chunk

    def _wait_ack(self):
        # Expect '+' (ack). Tolerate leading junk; '-' means resend (we don't).
        while True:
            if not self.buf:
                self._recv_more()
            c, self.buf = self.buf[:1], self.buf[1:]
            if c == b"+":
                return
            if c == b"-":
                raise RuntimeError("gdb stub NAK'd packet")
            # ignore anything else (e.g. stray notification bytes)

    def _recv_packet(self) -> str:
        # Find a full $...#xx frame in the buffer.
        while True:
            start = self.buf.find(b"$")
            if start != -1:
                hashpos = self.buf.find(b"#", start)
                if hashpos != -1 and len(self.buf) >= hashpos + 3:
                    payload = self.buf[start + 1:hashpos]
                    self.buf = self.buf[hashpos + 3:]
                    self.sock.sendall(b"+")  # ack
                    return payload.decode("ascii", "replace")
            self._recv_more()

    def interrupt(self):
        # Ctrl-C halts the target; stub replies with a stop packet (S05/T05...).
        self.sock.sendall(b"\x03")
        try:
            self._recv_packet()
        except Exception:
            pass

    # --- public ops ---
    def read_mem(self, addr: int, length: int) -> bytes:
        """Read `length` bytes at `addr`. Chunks large reads (stub caps packet size)."""
        out = bytearray()
        off = 0
        while off < length:
            n = min(512, length - off)
            self._send_packet(f"m{addr + off:x},{n:x}")
            resp = self._recv_packet()
            if resp.startswith("E") and len(resp) <= 3:
                raise RuntimeError(f"read error at {addr+off:#x}: {resp}")
            out += bytes.fromhex(resp)
            off += n
        return bytes(out)

    def read_u8(self, addr):
        return self.read_mem(addr, 1)[0]

    def read_u16(self, addr):
        b = self.read_mem(addr, 2)
        return b[0] | (b[1] << 8)

    def read_u32(self, addr):
        b = self.read_mem(addr, 4)
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    def cont(self):
        # Resume execution. The stub acks 'c' immediately, then sends nothing
        # until the next stop (which we trigger later via interrupt()).
        self._send_packet("c")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


if __name__ == "__main__":
    import sys
    g = GdbRsp()
    g.connect()
    print("connected + halted")
    ewram = g.read_mem(0x02000000, 256)
    iwram = g.read_mem(0x03000000, 256)
    nz_e = sum(1 for b in ewram if b)
    nz_i = sum(1 for b in iwram if b)
    print(f"EWRAM@0x02000000 first 32B: {ewram[:32].hex()}  (nonzero {nz_e}/256)")
    print(f"IWRAM@0x03000000 first 32B: {iwram[:32].hex()}  (nonzero {nz_i}/256)")
    g.cont()
    g.close()
    print("OK" if (nz_e or nz_i) else "WARN: all-zero (game may not have booted)")
