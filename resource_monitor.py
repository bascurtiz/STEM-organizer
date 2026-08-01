"""Lightweight CPU / GPU / RAM / disk activity sampling (Windows-first, no extra deps)."""
from __future__ import annotations

import ctypes
import struct
import sys
import time
from dataclasses import dataclass

_DISK_REF_BPS = 50 * 1024 * 1024  # 50 MB/s ~= 100% on the tiny bars
_NT_STATUS_SUCCESS = 0
_SYSTEM_PERFORMANCE_INFORMATION = 2
_NT_PERF_BUF_SIZE = 512


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu: float
    gpu: float
    ram: float
    disk_read: float
    disk_write: float


class _WinCpuSampler:
    def __init__(self) -> None:
        self._last_idle = 0
        self._last_kernel = 0
        self._last_user = 0
        self._ready = False

    def sample(self) -> float:
        idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user),
        ):
            return 0.0
        idle_v, kernel_v, user_v = idle.value, kernel.value, user.value
        if not self._ready:
            self._last_idle, self._last_kernel, self._last_user = idle_v, kernel_v, user_v
            self._ready = True
            return 0.0
        idle_d = idle_v - self._last_idle
        total_d = (kernel_v + user_v) - (self._last_kernel + self._last_user)
        self._last_idle, self._last_kernel, self._last_user = idle_v, kernel_v, user_v
        if total_d <= 0:
            return 0.0
        return max(0.0, min(100.0, (1.0 - idle_d / total_d) * 100.0))


class _WinRamSampler:
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    def sample(self) -> float:
        stat = self._MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return 0.0
        return float(stat.dwMemoryLoad)


class _NvmlSampler:
    def __init__(self) -> None:
        self._ok = False
        self._handle = ctypes.c_void_p()
        if sys.platform != 'win32':
            return
        try:
            nvml = ctypes.WinDLL('nvml.dll')
        except OSError:
            return
        nvml_init = getattr(nvml, 'nvmlInit_v2', None) or getattr(nvml, 'nvmlInit', None)
        nvml_shutdown = getattr(nvml, 'nvmlShutdown', None)
        nvml_get_count = getattr(nvml, 'nvmlDeviceGetCount_v2', None) or getattr(
            nvml, 'nvmlDeviceGetCount', None,
        )
        nvml_get_handle = getattr(nvml, 'nvmlDeviceGetHandleByIndex_v2', None) or getattr(
            nvml, 'nvmlDeviceGetHandleByIndex', None,
        )
        nvml_get_util = getattr(nvml, 'nvmlDeviceGetUtilizationRates', None)
        if not all((nvml_init, nvml_shutdown, nvml_get_count, nvml_get_handle, nvml_get_util)):
            return
        if nvml_init() != 0:
            return
        count = ctypes.c_uint()
        if nvml_get_count(ctypes.byref(count)) != 0 or count.value < 1:
            nvml_shutdown()
            return
        if nvml_get_handle(0, ctypes.byref(self._handle)) != 0:
            nvml_shutdown()
            return

        class _Util(ctypes.Structure):
            _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

        self._util_struct = _Util
        self._nvml_get_util = nvml_get_util
        self._nvml_shutdown = nvml_shutdown
        self._ok = True

    def sample(self) -> float:
        if not self._ok:
            return 0.0
        util = self._util_struct()
        if self._nvml_get_util(self._handle, ctypes.byref(util)) != 0:
            return 0.0
        return float(util.gpu)

    def close(self) -> None:
        if self._ok:
            self._nvml_shutdown()
            self._ok = False


class _WinDiskSampler:
    """System-wide disk throughput via NtQuerySystemInformation (no psutil)."""

    def __init__(self) -> None:
        self._ok = False
        self._ntdll = None
        self._last_read = 0
        self._last_write = 0
        self._last_at = 0.0
        if sys.platform != 'win32':
            return
        try:
            self._ntdll = ctypes.WinDLL('ntdll')
        except OSError:
            return
        status, read_c, write_c = self._query()
        if status != _NT_STATUS_SUCCESS:
            return
        self._last_read = read_c
        self._last_write = write_c
        self._last_at = time.monotonic()
        self._ok = True

    def _query(self) -> tuple[int, int, int]:
        assert self._ntdll is not None
        buf = (ctypes.c_ubyte * _NT_PERF_BUF_SIZE)()
        ret_len = ctypes.c_ulong()
        status = self._ntdll.NtQuerySystemInformation(
            _SYSTEM_PERFORMANCE_INFORMATION,
            ctypes.byref(buf),
            _NT_PERF_BUF_SIZE,
            ctypes.byref(ret_len),
        )
        if status != _NT_STATUS_SUCCESS:
            return status, 0, 0
        raw = bytes(buf[: ret_len.value])
        read_count = struct.unpack_from('<q', raw, 8)[0]
        write_count = struct.unpack_from('<q', raw, 16)[0]
        return _NT_STATUS_SUCCESS, read_count, write_count

    @staticmethod
    def _to_pct(bytes_per_sec: float) -> float:
        return min(100.0, max(0.0, bytes_per_sec) / _DISK_REF_BPS * 100.0)

    def sample(self) -> tuple[float, float]:
        if not self._ok:
            return 0.0, 0.0
        status, read_c, write_c = self._query()
        if status != _NT_STATUS_SUCCESS:
            return 0.0, 0.0
        now = time.monotonic()
        dt = now - self._last_at
        if dt <= 0:
            return 0.0, 0.0
        read_bps = max(0, read_c - self._last_read) / dt
        write_bps = max(0, write_c - self._last_write) / dt
        self._last_read = read_c
        self._last_write = write_c
        self._last_at = now
        return self._to_pct(read_bps), self._to_pct(write_bps)


class _PsutilDiskSampler:
    def __init__(self) -> None:
        import psutil  # type: ignore[import-untyped]

        self._psutil = psutil
        self._last_read = 0
        self._last_write = 0
        self._last_at = 0.0
        counters = psutil.disk_io_counters()
        if counters is not None:
            self._last_read = counters.read_bytes
            self._last_write = counters.write_bytes
            self._last_at = time.monotonic()

    def sample(self) -> tuple[float, float]:
        counters = self._psutil.disk_io_counters()
        if counters is None:
            return 0.0, 0.0
        now = time.monotonic()
        dt = now - self._last_at
        if dt <= 0:
            return 0.0, 0.0
        read_bps = max(0, counters.read_bytes - self._last_read) / dt
        write_bps = max(0, counters.write_bytes - self._last_write) / dt
        self._last_read = counters.read_bytes
        self._last_write = counters.write_bytes
        self._last_at = now
        return (
            _WinDiskSampler._to_pct(read_bps),
            _WinDiskSampler._to_pct(write_bps),
        )


class _DiskSampler:
    def __init__(self) -> None:
        self._backend = None
        try:
            self._backend = _PsutilDiskSampler()
        except ImportError:
            if sys.platform == 'win32':
                win = _WinDiskSampler()
                if win._ok:
                    self._backend = win

    def sample(self) -> tuple[float, float]:
        if self._backend is None:
            return 0.0, 0.0
        return self._backend.sample()


class ResourceMonitor:
    def __init__(self) -> None:
        if sys.platform == 'win32':
            self._cpu = _WinCpuSampler()
            self._ram = _WinRamSampler()
            self._gpu = _NvmlSampler()
        else:
            self._cpu = None
            self._ram = None
            self._gpu = _NvmlSampler()
        self._disk = _DiskSampler()
        self._psutil = None
        if sys.platform != 'win32':
            try:
                import psutil  # type: ignore[import-untyped]
            except ImportError:
                pass
            else:
                self._psutil = psutil
                psutil.cpu_percent(interval=None)

    def close(self) -> None:
        if self._gpu is not None:
            self._gpu.close()

    def sample(self) -> ResourceSnapshot:
        if self._psutil is not None:
            cpu = float(self._psutil.cpu_percent(interval=None))
            ram = float(self._psutil.virtual_memory().percent)
            gpu = self._gpu.sample() if self._gpu is not None else 0.0
        elif sys.platform == 'win32':
            cpu = self._cpu.sample() if self._cpu is not None else 0.0
            ram = self._ram.sample() if self._ram is not None else 0.0
            gpu = self._gpu.sample() if self._gpu is not None else 0.0
        else:
            cpu = ram = gpu = 0.0
        disk_read, disk_write = self._disk.sample()
        return ResourceSnapshot(cpu, gpu, ram, disk_read, disk_write)
